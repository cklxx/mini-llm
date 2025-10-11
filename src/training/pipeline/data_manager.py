"""Utilities for resolving training data and building loaders."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from ..datasets import ConversationDataset, LanguageModelingDataset


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

    def resolve_data_path(self, filename: str) -> str | None:
        search_dirs = [self.config.data_dir]

        dataset_dir = os.path.join(self.config.data_dir, "dataset")
        if os.path.exists(dataset_dir):
            search_dirs.append(dataset_dir)

        extra_dir = os.environ.get("MINIGPT_DATA_DIR")
        if extra_dir:
            search_dirs.append(extra_dir)

        for root in search_dirs:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                return candidate
        return None

    def resolve_dataset_file(self, *candidates: str) -> str | None:
        for name in candidates:
            path = self.resolve_data_path(name)
            if path:
                if name != os.path.basename(path):
                    print(f"✅ 使用数据文件 {os.path.basename(path)} (匹配别名 {name})")
                return path
        print(f"⚠️  未找到数据文件候选: {', '.join(candidates)}")
        return None

    def dataset_candidates(self) -> Sequence[Sequence[str]]:
        if self.mode == "pretrain":
            return [
                ("pretrain_hq.jsonl",),
                ("sft_mini_512.jsonl", "minigpt_identity.jsonl"),
            ]
        if self.mode == "sft":
            return [
                ("sft_mini_512.jsonl",),
                ("alex_identity.jsonl", "minigpt_identity.jsonl"),
                ("ultra_think.jsonl",),
            ]
        if self.mode == "dpo":
            return [("dpo.jsonl",)]
        if self.mode == "rlhf":
            return [
                ("alex_identity.jsonl", "minigpt_identity.jsonl"),
                ("ultra_think.jsonl",),
            ]
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

    @staticmethod
    def load_records(path: str) -> list[Any]:
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


class DatasetPreparer:
    """Build train/validation dataloaders with sampling controls."""

    def __init__(self, config, mode: str, tokenizer, resolver: DataResolver, seed: int):
        self.config = config
        self.mode = mode
        self.tokenizer = tokenizer
        self.resolver = resolver
        self.rng = random.Random(seed)

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

            sample_ratio = sampling_cfg.get("sample_ratio", 1.0) or 1.0
            max_samples = sampling_cfg.get("max_samples")
            val_split = sampling_cfg.get("val_split", getattr(self.config, "validation_split", 0.0))

            sample_size = int(round(original_count * sample_ratio))
            if max_samples is not None:
                sample_size = min(sample_size, max_samples)
            sample_size = max(1, min(sample_size, original_count))

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
            return (
                self._create_dpo_dataset(train_records),
                self._create_dpo_dataset(val_records) if val_records else None,
            )
        raise ValueError(f"不支持的训练模式: {self.mode}")

    def _build_train_loader(self, dataset):
        drop_last = len(dataset) >= self.config.batch_size
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=getattr(self.config, "prefetch_factor", 2),
            drop_last=drop_last,
        )

    def _build_val_loader(self, dataset):
        if not dataset:
            return None
        return DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, 8),
            shuffle=False,
            num_workers=max(1, self.config.num_workers // 2),
            pin_memory=self.config.pin_memory,
            persistent_workers=False,
            drop_last=False,
        )

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
        texts = [item["chosen"] for item in data if "chosen" in item]
        return LanguageModelingDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_len,
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
