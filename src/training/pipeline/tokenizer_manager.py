"""Tokenizer management for the training pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer

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
        return getattr(self.config, "tokenizer_type", "bpe") == "huggingface"

    @property
    def tokenizer_path(self) -> str:
        if self._use_hf_tokenizer:
            return os.path.join(self.output_dir, "tokenizer", "tokenizer.json")
        return os.path.join(self.output_dir, "tokenizer.pkl")

    @property
    def tokenizer_dir(self) -> str:
        if self._use_hf_tokenizer:
            return os.path.join(self.output_dir, "tokenizer")
        return self.output_dir

    def setup(self, retrain: bool = False) -> BPETokenizer:
        print("🔤 设置分词器...")

        if self._use_hf_tokenizer:
            return self._setup_hf_tokenizer(retrain=retrain)

        pretrain_tokenizer_path = None
        if self.mode in ["sft", "dpo", "rlhf"]:
            pretrain_dir = os.path.join(
                self.config.checkpoint_dir, f"pretrain_{self.config.model_size}"
            )
            pretrain_tokenizer_path = os.path.join(pretrain_dir, "tokenizer.pkl")
            if os.path.exists(pretrain_tokenizer_path):
                print(f"✅ 从 pretrain checkpoint 加载分词器: {pretrain_tokenizer_path}")
                tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
                tokenizer.load(pretrain_tokenizer_path)
                self._copy_tokenizer(pretrain_tokenizer_path)
                print(f"📋 分词器已复制到: {self.tokenizer_path}")
                print(f"词汇表大小: {tokenizer.vocab_size}")
                return tokenizer
            print(f"⚠️  未找到 pretrain 分词器: {pretrain_tokenizer_path}")
            print("   将训练新的分词器（建议先运行 pretrain 模式）")

        if os.path.exists(self.tokenizer_path) and not retrain:
            print(f"加载现有分词器: {self.tokenizer_path}")
            tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
            tokenizer.load(self.tokenizer_path)
        else:
            print("训练新的分词器...")
            texts = self._collect_texts_for_tokenizer()
            tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
            tokenizer.train(texts)
            tokenizer.save(self.tokenizer_path)
            print(f"分词器已保存: {self.tokenizer_path}")

        print(f"词汇表大小: {tokenizer.vocab_size}")
        return tokenizer

    def _setup_hf_tokenizer(self, retrain: bool = False) -> BPETokenizer:
        tokenizer = BPETokenizer()

        candidate_sources: list[Path] = []
        if self.mode in ["sft", "dpo", "rlhf"]:
            pretrain_dir = Path(self.config.checkpoint_dir) / f"pretrain_{self.config.model_size}"
            candidate_sources.append(pretrain_dir / "tokenizer")

        config_json = getattr(self.config, "tokenizer_json_path", None)
        if config_json:
            candidate_sources.append(Path(config_json).parent)

        output_dir = Path(self.tokenizer_dir)
        if output_dir.exists() and not retrain:
            candidate_sources.insert(0, output_dir)

        source_dir: Path | None = None
        for candidate in candidate_sources:
            if candidate and candidate.exists():
                json_path = candidate / "tokenizer.json"
                if json_path.exists():
                    source_dir = candidate
                    break

        if source_dir is None:
            raise FileNotFoundError(
                "未找到 HuggingFace 分词器文件。请确保 tokenizer.json 已复制到项目中。"
            )

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

    def _copy_tokenizer(self, pretrain_tokenizer_path: str) -> None:
        import shutil

        if self._use_hf_tokenizer:
            source_dir = Path(pretrain_tokenizer_path)
            if source_dir.is_file():
                source_dir = source_dir.parent
            self._copy_hf_tokenizer(source_dir, Path(self.tokenizer_dir))
            return

        os.makedirs(self.output_dir, exist_ok=True)
        shutil.copy2(pretrain_tokenizer_path, self.tokenizer_path)

    def _copy_hf_tokenizer(self, source_dir: Path, dest_dir: Path) -> None:
        import shutil

        dest_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            src_file = source_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dest_dir / filename)
