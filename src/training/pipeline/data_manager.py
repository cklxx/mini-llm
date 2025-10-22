"""Utilities for resolving training data and building loaders."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data import load_tokenizer
from data.high_performance_loader import (
    DataLoadingConfig,
    IntelligentDataCache,
    ParallelDataProcessor,
    StreamingJsonLoader,
)

from ..datasets import ConversationDataset, DPODataset, LanguageModelingDataset


@dataclass
class DatasetStats:
    dataset: str
    path: str
    original_samples: int
    sampled_samples: int
    train_samples: int
    val_samples: int
    sampling_config: dict[str, Any]


class DataResolver:
    """Resolve dataset paths for a given training mode."""

    def __init__(self, config, mode: str):
        self.config = config
        self.mode = mode
        self._manifest_cache: dict[str, list[str] | None] = {}
        self._fast_cache: dict[tuple[str, str], list[Any]] = {}

    # ------------------------------------------------------------------
    def _manifest_filename(self) -> str | None:
        if os.environ.get("MINIGPT_DISABLE_MANIFEST", "0") == "1":
            return None

        mapping = {
            "pretrain": "pretrain_manifest.json",
            "sft": "sft_manifest.json",
            "dpo": "dpo_manifest.json",
            "rlhf": "rlhf_manifest.json",
        }
        return mapping.get(self.mode)

    def _manifest_dir(self) -> str:
        if hasattr(self.config, "manifest_dir"):
            return self.config.manifest_dir
        if hasattr(self.config, "project_root"):
            return os.path.join(self.config.project_root, "configs", "data")
        return os.path.join(os.getcwd(), "configs", "data")

    def _load_manifest_dataset_paths(self) -> list[str] | None:
        if self.mode in self._manifest_cache:
            return self._manifest_cache[self.mode]

        filename = self._manifest_filename()
        if not filename:
            self._manifest_cache[self.mode] = None
            return None

        manifest_path = os.path.join(self._manifest_dir(), filename)
        if not os.path.exists(manifest_path):
            print(f"⚠️  未找到 manifest 文件: {manifest_path}")
            self._manifest_cache[self.mode] = None
            return None

        try:
            with open(manifest_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️  读取 manifest 失败 {manifest_path}: {exc}")
            self._manifest_cache[self.mode] = None
            return None

        datasets = data.get("datasets", []) if isinstance(data, dict) else []
        resolved_paths: list[str] = []
        missing_entries: list[str] = []

        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not isinstance(path, str):
                continue

            resolved = self._maybe_resolve_direct_path(path)
            if not resolved:
                # ``resolve_data_path`` treats the input as a filename. When manifest
                # entries already include directory components (e.g. ``data/final``),
                # fall back to checking whether the raw manifest path exists relative
                # to the project root before giving up.
                resolved = self.resolve_data_path(os.path.basename(path))
            if resolved and os.path.exists(resolved):
                resolved_paths.append(resolved)
            else:
                missing_entries.append(path)

        if missing_entries:
            missing_list = ", ".join(missing_entries)
            print(
                "⚠️  manifest {path} 中的部分数据文件缺失，"
                "将回退到默认候选: {missing}".format(
                    path=manifest_path, missing=missing_list
                )
            )

        if not resolved_paths:
            self._manifest_cache[self.mode] = None
            return None

        self._manifest_cache[self.mode] = resolved_paths
        return resolved_paths

    def resolve_data_path(self, filename: str) -> str | None:
        search_dirs = []
        if hasattr(self.config, "data_search_dirs"):
            search_dirs.extend(self.config.data_search_dirs)
        else:
            search_dirs.append(self.config.data_dir)

        dataset_dir = os.path.join(self.config.data_dir, "dataset")
        if os.path.exists(dataset_dir) and dataset_dir not in search_dirs:
            search_dirs.append(dataset_dir)

        extra_dir = os.environ.get("MINIGPT_DATA_DIR")
        if extra_dir and extra_dir not in search_dirs:
            search_dirs.append(extra_dir)

        for root in search_dirs:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                return candidate
        return None

    def _maybe_resolve_direct_path(self, name: str) -> str | None:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        # Treat path with directory separators as potentially relative to project or data dir
        if os.path.sep in name or (os.path.altsep and os.path.altsep in name):
            if hasattr(self.config, "project_root"):
                project_candidate = os.path.join(self.config.project_root, name)
                if os.path.exists(project_candidate):
                    return project_candidate
            if hasattr(self.config, "data_dir"):
                data_candidate = os.path.join(self.config.data_dir, name)
                if os.path.exists(data_candidate):
                    return data_candidate
            cwd_candidate = os.path.join(os.getcwd(), name)
            if os.path.exists(cwd_candidate):
                return cwd_candidate
        return None

    def resolve_dataset_file(self, *candidates: str) -> str | None:
        for name in candidates:
            direct = self._maybe_resolve_direct_path(name)
            if direct:
                if name != os.path.basename(direct):
                    print(f"✅ 使用数据文件 {os.path.basename(direct)} (匹配路径 {name})")
                return direct
            path = self.resolve_data_path(name)
            if path:
                if name != os.path.basename(path):
                    print(f"✅ 使用数据文件 {os.path.basename(path)} (匹配别名 {name})")
                return path
        print(f"⚠️  未找到数据文件候选: {', '.join(candidates)}")
        return None

    def dataset_candidates(self) -> Sequence[Sequence[str]]:
        manifest_paths = self._load_manifest_dataset_paths()
        if manifest_paths:
            return [(path,) for path in manifest_paths]
        if self.mode == "pretrain":
            candidates: list[tuple[str, ...]] = []
            explicit = getattr(self.config, "pretrain_data_path", None)
            if explicit:
                candidates.append((explicit,))
            candidates.extend(
                [
                    (
                        "wiki_zh_full.simdedup.jsonl",
                        "wiki_zh_full.cleaned.jsonl",
                        "wiki_pretrain_part1.json",
                    ),
                    ("chinacorpus_full.simdedup.jsonl", "chinacorpus_full.cleaned.jsonl"),
                    ("pretrain_hq.cleaned.jsonl", "pretrain_hq.jsonl"),
                ]
            )
            return candidates
        if self.mode == "sft":
            candidates = []
            explicit = getattr(self.config, "sft_data_path", None)
            if explicit:
                candidates.append((explicit,))
            candidates.extend(
                [
                    ("sft_mini_512.cleaned.jsonl", "sft_mini_512.jsonl"),
                    ("alex_identity.jsonl", "minigpt_identity.jsonl"),
                    ("ultra_think.jsonl",),
                ]
            )
            return candidates
        if self.mode == "dpo":
            candidates = []
            explicit = getattr(self.config, "dpo_data_path", None)
            if explicit:
                candidates.append((explicit,))
            candidates.append(("dpo.jsonl",))
            return candidates
        if self.mode == "rlhf":
            candidates = []
            explicit = getattr(self.config, "rlhf_data_path", None)
            if explicit:
                candidates.append((explicit,))
            candidates.extend(
                [
                    ("alex_identity.jsonl", "minigpt_identity.jsonl"),
                    ("ultra_think.jsonl",),
                ]
            )
            return candidates
        raise ValueError(f"不支持的训练模式: {self.mode}")

    def get_data_paths(self) -> list[str]:
        seen_paths = set()
        resolved: list[str] = []
        for option in self.dataset_candidates():
            path = self.resolve_dataset_file(*option)
            if not path:
                continue
            canonical = os.path.realpath(path)
            if canonical in seen_paths:
                print(
                    f"⚠️  数据文件 {os.path.basename(path)} 已经在其它候选中被引用，跳过重复加载以避免过拟合。"
                )
                continue
            seen_paths.add(canonical)
            resolved.append(path)
        return resolved

    def load_records(self, path: str) -> list[Any]:
        if (
            getattr(self.config, "use_high_performance_data_loading", False)
            and self.mode in {"sft", "pretrain", "dpo", "rlhf"}
        ):
            try:
                return self._load_records_fast(path)
            except Exception as exc:  # pragma: no cover - 安全回退
                print(
                    f"⚠️  高性能数据加载失败，回退到标准模式: {exc}"
                )
        return self._load_records_standard(path)

    def _load_records_standard(self, path: str) -> list[Any]:
        records: list[Any] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _load_records_fast(self, path: str) -> list[Any]:
        cache_key = (os.path.realpath(path), self.mode)
        if cache_key in self._fast_cache:
            return self._fast_cache[cache_key]

        config = self._build_fast_loading_config(path)
        cache: IntelligentDataCache | None = None
        processed: list[Any] | None = None

        if config.enable_cache:
            cache = IntelligentDataCache(config.cache_dir)
            if not config.force_rebuild_cache and cache.is_cache_valid(config):
                cached = cache.load_cache(config)
                if cached is not None:
                    processed = cached

        start_time = time.perf_counter()
        if processed is None:
            loader = StreamingJsonLoader(path, config.chunk_size)
            data_chunks = list(loader.get_chunks())
            tokenizer = None
            tokenizer_path = getattr(self.config, "tokenizer_json_path", None)
            if tokenizer_path:
                try:
                    tokenizer = load_tokenizer(tokenizer_path)
                except Exception as exc:  # pragma: no cover - 回退到字符长度
                    print(
                        f"⚠️  无法加载分词器 {tokenizer_path}，高性能数据预处理将使用字符长度: {exc}"
                    )
            processor = ParallelDataProcessor(
                config.max_parallel_workers, tokenizer=tokenizer
            )
            if self.mode in {"sft", "rlhf"}:
                processed = processor.process_conversations(
                    data_chunks, config.max_length
                )
            elif self.mode == "pretrain":
                processed = processor.process_pretrain_texts(
                    data_chunks, config.max_length
                )
                processed = [
                    {"text": text}
                    for text in processed
                    if isinstance(text, str) and text
                ]
            elif self.mode == "dpo":
                processed = processor.process_preference_pairs(
                    data_chunks, config.max_length
                )
            else:  # pragma: no cover - 不支持的模式
                raise ValueError(f"Unsupported fast loading mode: {self.mode}")

            if cache and processed is not None:
                cache.save_cache(config, processed)

        processed = processed or []
        duration = time.perf_counter() - start_time
        sample_count = len(processed)
        print(
            f"⚡️  使用高性能数据加载 {os.path.basename(path)}: "
            f"{sample_count} 条样本，耗时 {duration:.2f}s"
        )

        self._fast_cache[cache_key] = processed
        return self._fast_cache[cache_key]

    def _build_fast_loading_config(self, path: str) -> DataLoadingConfig:
        return DataLoadingConfig(
            data_path=path,
            max_length=getattr(self.config, "max_seq_len", 512),
            batch_size=getattr(self.config, "batch_size", 32),
            num_workers=max(1, getattr(self.config, "num_workers", 1)),
            prefetch_factor=getattr(self.config, "prefetch_factor", 2),
            pin_memory=getattr(self.config, "pin_memory", False),
            enable_cache=getattr(self.config, "data_cache_enabled", True),
            cache_dir=getattr(self.config, "data_cache_dir", "data_cache"),
            force_rebuild_cache=getattr(
                self.config, "data_cache_force_rebuild", False
            ),
            streaming=getattr(self.config, "data_streaming_enabled", False),
            chunk_size=getattr(self.config, "data_chunk_size", 10000),
            buffer_size=getattr(self.config, "data_buffer_size", 50000),
            parallel_processing=True,
            max_parallel_workers=max(
                1, getattr(self.config, "data_max_parallel_workers", 4)
            ),
            tokenizer_path=getattr(self.config, "tokenizer_json_path", None),
        )


class DatasetPreparer:
    """Build train/validation dataloaders with sampling controls."""

    def __init__(self, config, mode: str, tokenizer, resolver: DataResolver, seed: int):
        self.config = config
        self.mode = mode
        self.tokenizer = tokenizer
        self.resolver = resolver
        self.rng = random.Random(seed)
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self._prefetch_factor = getattr(self.config, "prefetch_factor", 2)

    def build(self) -> tuple[DataLoader, DataLoader | None, list[dict[str, Any]]]:
        data_paths = self.resolver.get_data_paths()
        if not data_paths:
            raise RuntimeError("未能解析到任何数据集路径，请检查配置或数据文件是否存在。")

        train_records: list[Any] = []
        val_records: list[Any] = []
        stats: list[dict[str, Any]] = []

        default_sampling = (
            self.config.dataset_sampling.get("default", {})
            if hasattr(self.config, "dataset_sampling")
            else {}
        )

        global_ratio = max(0.0, getattr(self.config, "dataset_global_sample_ratio", 1.0))
        if global_ratio != 1.0:
            print(f"⚙️  应用全局采样比例: {global_ratio:.3f}")

        for data_path in data_paths:
            dataset_name = os.path.basename(data_path)
            sampling_cfg = (
                self.config.dataset_sampling.get(dataset_name, default_sampling)
                if hasattr(self.config, "dataset_sampling")
                else {}
            )

            file_data = self.resolver.load_records(data_path)
            original_count = len(file_data)
            if original_count == 0:
                print(f"⚠️  数据文件 {dataset_name} 未加载到有效样本，跳过")
                continue

            if global_ratio == 0.0:
                print(
                    f"⚠️  全局采样比例为 0，跳过数据集 {dataset_name} 以避免空样本。"
                )
                continue

            sample_ratio = sampling_cfg.get("sample_ratio", 1.0) or 1.0
            max_samples = sampling_cfg.get("max_samples")
            val_split = sampling_cfg.get("val_split", getattr(self.config, "validation_split", 0.0))

            sample_size = int(round(original_count * sample_ratio))
            if max_samples is not None:
                sample_size = min(sample_size, max_samples)
            sample_size = max(1, min(sample_size, original_count))

            if global_ratio != 1.0:
                scaled_size = max(1, int(math.ceil(sample_size * global_ratio)))
                if global_ratio < 1.0:
                    sample_size = min(sample_size, scaled_size)
                else:
                    sample_size = min(original_count, max(sample_size, scaled_size))

            if sample_size < original_count:
                sampled_data = self.rng.sample(file_data, sample_size)
            else:
                sampled_data = list(file_data)

            self.rng.shuffle(sampled_data)

            min_val_samples = getattr(self.config, "validation_min_samples", 0)
            val_ratio = max(0.0, min(1.0, val_split or 0.0))
            val_count = int(round(sample_size * val_ratio)) if val_ratio > 0 else 0

            if sample_size < max(min_val_samples, 3):
                val_count = 0
            if val_count >= sample_size:
                val_count = max(0, sample_size - 1)

            val_subset = sampled_data[:val_count]
            train_subset = sampled_data[val_count:]
            if not train_subset:
                train_subset = sampled_data
                val_subset = []

            train_records.extend(train_subset)
            if val_subset:
                val_records.extend(val_subset)

            stats.append(
                {
                    "dataset": dataset_name,
                    "path": data_path,
                    "original_samples": original_count,
                    "sampled_samples": sample_size,
                    "train_samples": len(train_subset),
                    "val_samples": len(val_subset),
                    "sampling_config": sampling_cfg,
                }
            )

            print(
                f"📦 {dataset_name}: 原始 {original_count} 条 → 采样 {sample_size} 条 | "
                f"训练 {len(train_subset)} 条, 验证 {len(val_subset)} 条"
            )

        if not train_records:
            raise RuntimeError("采样后训练集为空，请调整数据配额或检查数据文件内容。")

        train_dataset, val_dataset = self._build_mode_specific_datasets(train_records, val_records)
        train_loader = self._build_train_loader(train_dataset)
        val_loader = self._build_val_loader(val_dataset)

        print(f"✅ 最终训练样本数: {len(train_dataset)}")
        print(f"训练数据批次数: {len(train_loader)}")
        if val_loader:
            print(f"✅ 最终验证样本数: {len(val_dataset)}")
            print(f"验证数据批次数: {len(val_loader)}")
        else:
            print("✅ 最终验证样本数: 0")

        return train_loader, val_loader, stats

    def _build_mode_specific_datasets(self, train_records, val_records):
        if self.mode == "pretrain":
            return (
                self._create_pretrain_dataset(train_records),
                self._create_pretrain_dataset(val_records) if val_records else None,
            )
        if self.mode in {"sft", "rlhf"}:
            return (
                self._create_sft_dataset(
                    train_records,
                    augmentation=getattr(self.config, "conversation_augmentation", None),
                ),
                self._create_sft_dataset(val_records, augmentation=None) if val_records else None,
            )
        if self.mode == "dpo":
            train_dataset = self._create_dpo_dataset(train_records)
            if train_dataset is None:
                raise RuntimeError("DPO训练需要包含 'chosen'/'rejected' 成对样本的数据集")
            return (
                train_dataset,
                None,
            )
        raise ValueError(f"不支持的训练模式: {self.mode}")

    def _build_train_loader(self, dataset):
        drop_last = len(dataset) >= self.config.batch_size
        sampler = None
        shuffle = True
        if self._is_distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=drop_last,
            )
            shuffle = False

        loader_kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": self.config.persistent_workers,
            "drop_last": drop_last,
            "sampler": sampler,
        }
        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self._prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    def _build_val_loader(self, dataset):
        if not dataset:
            return None
        sampler = None
        shuffle = False
        if self._is_distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=False,
                drop_last=False,
            )

        num_workers = max(1, self.config.num_workers // 2) if self.config.num_workers > 0 else 0
        loader_kwargs: dict[str, Any] = {
            "batch_size": min(self.config.batch_size, 8),
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": False,
            "drop_last": False,
            "sampler": sampler,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(2, self._prefetch_factor // 2)

        return DataLoader(dataset, **loader_kwargs)

    def _create_pretrain_dataset(self, data: Iterable[Any]):
        texts = []
        for item in data:
            text = self._extract_text(item)
            if text and len(text.strip()) > 10:
                texts.append(text)
        return LanguageModelingDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_len,
        )

    def _create_sft_dataset(self, data: Iterable[Any], augmentation=None):
        conversations = []
        for item in data:
            if "conversations" in item:
                conversations.append(item["conversations"])
            elif "input" in item and "output" in item:
                conversations.append({"input": item["input"], "output": item["output"]})
        print(f"📊 SFT数据集包含 {len(conversations)} 个对话")
        return ConversationDataset(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_len,
            role_tokens=getattr(self.config, "role_tokens", None),
            augmentation=augmentation,
            seed=getattr(self.config, "random_seed", 42),
        )

    def _create_dpo_dataset(self, data: Iterable[Any]):
        if not data:
            return None

        records: list[Any] = []
        for item in data:
            if isinstance(item, dict) and "chosen" in item and "rejected" in item:
                records.append(item)

        if not records:
            print("⚠️  DPO 数据集中缺少有效的偏好样本，已跳过。")
            return None

        print(f"📊 DPO数据集包含 {len(records)} 组偏好对")
        return DPODataset(
            records=records,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_len,
            role_tokens=getattr(self.config, "role_tokens", None),
            seed=getattr(self.config, "random_seed", 42),
        )

    @staticmethod
    def _extract_text(data: Any) -> str | None:
        if "text" in data:
            return data["text"]
        if "conversations" in data:
            text = ""
            for turn in data["conversations"]:
                if "content" in turn:
                    text += turn["content"] + " "
            return text.strip()
        if "input" in data and "output" in data:
            return f"{data['input']} {data['output']}"
        if "chosen" in data and "rejected" in data:
            return data["chosen"]
        return None
