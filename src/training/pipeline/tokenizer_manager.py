"""Tokenizer management for the training pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.rust_bpe_tokenizer import RustBPETokenizer

from .data_manager import DataResolver


class TokenizerManager:
    """Handle tokenizer loading and training across stages."""

    def __init__(self, config, mode: str, output_dir: str, resolver: DataResolver):
        self.config = config
        self.mode = mode
        self.output_dir = output_dir
        self.resolver = resolver

    @property
    def _use_hf_tokenizer(self) -> bool:
        return getattr(self.config, "tokenizer_type", "huggingface") == "huggingface"

    @property
    def tokenizer_dir(self) -> Path:
        return Path(self.output_dir) / "tokenizer"

    def setup(self, retrain: bool = False):
        print("🔤 设置分词器...")

        if self._use_hf_tokenizer:
            return self._setup_hf_tokenizer(retrain=retrain)
        if getattr(self.config, 'tokenizer_type', 'huggingface') == 'rust-bpe':
            return self._setup_rust_tokenizer(retrain=retrain)
        raise RuntimeError("不支持的分词器类型，请检查配置。")

    def _setup_rust_tokenizer(self, retrain: bool = False):
        candidate_sources: list[Path] = []
        if self.mode in ["sft", "dpo", "rlhf"]:
            pretrain_dir = Path(self.config.checkpoint_dir) / f"pretrain_{self.config.model_size}"
            candidate_sources.append(pretrain_dir / "tokenizer")

        output_dir = self.tokenizer_dir
        if output_dir.exists() and not retrain:
            candidate_sources.insert(0, output_dir)

        source_dir: Path | None = None
        if not retrain:
            for candidate in candidate_sources:
                if candidate and candidate.exists():
                    if (candidate / "tokenizer.pkl").exists():
                        source_dir = candidate
                        break

        tokenizer = RustBPETokenizer(vocab_size=self.config.vocab_size)

        if source_dir is None:
            print("⚠️  未找到现有 Rust BPE 分词器，将从数据训练新的词表。")
            texts = self._collect_texts_for_tokenizer()
            if not texts:
                raise RuntimeError("没有可用于训练分词器的文本数据。")
            tokenizer.train(texts)
            tokenizer.save(str(output_dir))
            print(f"分词器已保存: {output_dir}")
            print(f"词汇表大小: {tokenizer.vocab_size}")
            return tokenizer

        tokenizer.load(str(source_dir))
        if source_dir != output_dir:
            self._copy_rust_tokenizer(source_dir, output_dir)

        print(f"词汇表大小: {tokenizer.vocab_size}")
        return tokenizer

    def _setup_hf_tokenizer(self, retrain: bool = False) -> BPETokenizer:
        candidate_sources: list[Path] = []
        if self.mode in ["sft", "dpo", "rlhf"]:
            pretrain_dir = Path(self.config.checkpoint_dir) / f"pretrain_{self.config.model_size}"
            candidate_sources.append(pretrain_dir / "tokenizer")

        config_json = getattr(self.config, "tokenizer_json_path", None)
        if config_json:
            candidate_sources.append(Path(config_json).parent)

        output_dir = self.tokenizer_dir
        if output_dir.exists() and not retrain:
            candidate_sources.insert(0, output_dir)

        source_dir: Path | None = None
        if not retrain:
            for candidate in candidate_sources:
                if candidate and candidate.exists():
                    json_path = candidate / "tokenizer.json"
                    if json_path.exists():
                        source_dir = candidate
                        break

        tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)

        if source_dir is None:
            print("⚠️  未找到现有 HuggingFace 分词器，将从数据训练新的词表。")
            texts = self._collect_texts_for_tokenizer()
            if not texts:
                raise RuntimeError("没有可用于训练分词器的文本数据。")
            tokenizer.train(texts)
            tokenizer.save(str(output_dir))
            print(f"分词器已保存: {output_dir}")
            print(f"词汇表大小: {tokenizer.vocab_size}")
            return tokenizer

        tokenizer.load(str(source_dir))

        if source_dir != output_dir:
            self._copy_hf_tokenizer(source_dir, output_dir)

        print(f"词汇表大小: {tokenizer.vocab_size}")
        return tokenizer

    def _collect_texts_for_tokenizer(self) -> Iterable[str]:
        texts = []
        data_paths = self.resolver.get_data_paths()
        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"⚠️  数据文件不存在，跳过: {data_path}")
                continue
            with open(data_path, encoding="utf-8") as handle:
                for line in handle:
                    if len(texts) >= 50000:
                        break
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    text = self._extract_text(data)
                    if text:
                        texts.append(text)
            if len(texts) >= 50000:
                break
        print(f"收集了 {len(texts)} 条文本用于训练分词器")
        return texts

    @staticmethod
    def _extract_text(data) -> str | None:
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

    def _copy_hf_tokenizer(self, source_dir: Path, dest_dir: Path) -> None:
        import shutil

        dest_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            src_file = source_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dest_dir / filename)

    def _copy_rust_tokenizer(self, source_dir: Path, dest_dir: Path) -> None:
        import shutil

        dest_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("tokenizer.pkl", "tokenizer_meta.json"):
            src_file = source_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dest_dir / filename)

