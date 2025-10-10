"""
手写实现BPE (Byte Pair Encoding) Tokenizer
用于新手教学理解分词原理，优化了中文处理
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


CJK_RANGES: List[Tuple[str, str]] = [
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u31f0", "\u31ff"),  # Katakana Phonetic Extensions
    ("\uAC00", "\uD7AF"),  # Hangul syllables
]


def is_chinese_char(char: str) -> bool:
    """判断是否为中日韩字符（兼容原函数名称）。"""

    return any(start <= char <= end for start, end in CJK_RANGES)


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 40,
    fill: str = "█",
    empty: str = "░",
) -> None:
    """打印动态进度条，会覆盖上一行。"""

    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + empty * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="", flush=True)
    if current == total:
        print()


def show_progress_with_stats(
    current: int,
    total: int,
    start_time: float,
    prefix: str = "",
    extra_info: str = "",
) -> None:
    """统一的进度显示函数，包含速度和预计完成时间。"""

    elapsed = time.time() - start_time
    speed = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / speed if speed > 0 else 0

    suffix = f"({current:,}/{total:,}) {speed:.0f}/秒"
    if eta > 0:
        suffix += f" ETA: {eta:.0f}秒"
    if extra_info:
        suffix += f" | {extra_info}"

    print_progress_bar(current, total, prefix=prefix, suffix=suffix)


class BPETokenizer:
    """BPE分词器实现 - 优化了中文处理并支持字节回退。"""

    def __init__(
        self,
        vocab_size: int = 30000,
        *,
        lowercase: bool = True,
        normalize_nfkc: bool = True,
        strip_spaces: bool = True,
        enable_byte_fallback: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.normalize_nfkc = normalize_nfkc
        self.strip_spaces = strip_spaces
        self.enable_byte_fallback = enable_byte_fallback

        self.word_freqs: Dict[str, int] = {}
        self.splits: Dict[str, List[str]] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}

        # 特殊token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # 特殊token的ID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # Byte fallback配置
        self.byte_eow_token = "<BYTE_EOW>"
        self._byte_token_prefix = "<0x"
        self._byte_token_pattern = re.compile(r"<0x([0-9A-Fa-f]{2})>")
        self._reserved_special_tokens = 4
        self._checksum: str = ""

        self._update_reserved_token_count()
        self._sync_internal_state()

    # ------------------------------------------------------------------
    def _normalizer_signature(self) -> str:
        """Return a stable signature that describes the normalization pipeline."""

        stages: List[str] = []
        if self.normalize_nfkc:
            stages.append("nfkc")
        if self.strip_spaces:
            stages.append("strip_spaces")
        if self.lowercase:
            stages.append("lowercase")
        return "+".join(stages) if stages else "identity"

    # ------------------------------------------------------------------
    def _update_reserved_token_count(self) -> None:
        if self.enable_byte_fallback:
            minimum_vocab = self._reserved_special_tokens + 257
            if self.vocab_size < minimum_vocab:
                print(
                    f"⚠️  vocab_size={self.vocab_size} 太小，无法启用 byte fallback，自动禁用。"
                )
                self.enable_byte_fallback = False
        self._reserved_byte_tokens = 257 if self.enable_byte_fallback else 0
        self._reserved_token_count = self._reserved_special_tokens + self._reserved_byte_tokens

    def _format_byte_token(self, value: int) -> str:
        return f"<0x{value:02X}>"

    def _normalize(self, text: str) -> str:
        if self.normalize_nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.strip_spaces:
            text = re.sub(r"\s+", " ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def _sync_internal_state(self) -> None:
        """重建内部映射信息。"""

        self._id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        if self.enable_byte_fallback and self.vocab:
            self.byte_token_to_id = {}
            self.id_to_byte = {}
            for byte in range(256):
                token = self._format_byte_token(byte)
                if token in self.vocab:
                    idx = self.vocab[token]
                    self.byte_token_to_id[byte] = idx
                    self.id_to_byte[idx] = byte
            self.byte_eow_id = self.vocab.get(self.byte_eow_token)
        else:
            self.byte_token_to_id = {}
            self.id_to_byte = {}
            self.byte_eow_id = None

    def _validate_special_tokens(self, require_presence: bool = False, *, strict: bool = True) -> None:
        """Ensure special tokens exist at the expected ids."""

        expected = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
            self.bos_token: self.bos_id,
            self.eos_token: self.eos_id,
        }
        if self.enable_byte_fallback:
            if self.byte_eow_id is None:
                if require_presence:
                    raise ValueError("启用了 byte fallback 但缺少 BYTE_EOW token")
            else:
                expected[self.byte_eow_token] = self.byte_eow_id

        for token, expected_id in expected.items():
            actual_id = self.vocab.get(token)
            if actual_id is None:
                if require_presence and strict:
                    raise ValueError(f"缺少特殊token: {token}")
                continue
            if actual_id != expected_id:
                if strict:
                    raise ValueError(
                        f"特殊token {token} 的ID不匹配: 期望 {expected_id}, 实际 {actual_id}"
                    )

    def _checksum_payload(self) -> Dict[str, Any]:
        return {
            "vocab": sorted(self.vocab.items()),
            "merges": sorted(self.merges.items()),
            "config": self.get_config(),
        }

    def _update_checksum(self) -> None:
        payload = pickle.dumps(self._checksum_payload(), protocol=4)
        self._checksum = hashlib.sha256(payload).hexdigest()

    def _is_cjk_string(self, text: str) -> bool:
        return bool(text) and all(is_chinese_char(ch) for ch in text)

    def _should_add_space(self, text: str) -> bool:
        if not text:
            return False
        if self._is_cjk_string(text):
            return False
        return any(ch.isalnum() for ch in text)

    def _postprocess_text(self, text: str) -> str:
        if self.strip_spaces:
            text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
        text = re.sub(r"([\u4e00-\u9fff])\s+([\u3000-\u303F\uFF00-\uFFEF.,!?;:，。！？；：])", r"\1\2", text)
        text = re.sub(r"([\u3000-\u303F\uFF00-\uFFEF.,!?;:，。！？；：])\s+([\u4e00-\u9fff])", r"\1\2", text)
        text = re.sub(r"\s+([,.!?;:，。！？；：])", r"\1", text)
        return text.strip()

    def _encode_as_bytes(self, text: str) -> List[int]:
        if not self.enable_byte_fallback:
            return [self.unk_id]
        try:
            data = text.encode("utf-8")
        except Exception:
            data = text.encode("utf-8", errors="ignore")
        token_ids: List[int] = []
        for byte in data:
            token = self._format_byte_token(byte)
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # 如果缺失fallback token，退化为UNK
                return [self.unk_id]
        if self.byte_eow_id is not None:
            token_ids.append(self.byte_eow_id)
        return token_ids or [self.unk_id]

    # ------------------------------------------------------------------
    def pre_tokenize(self, text: str) -> List[str]:
        """预分词：改进中文处理"""

        text = self._normalize(text)
        words: List[str] = []
        current_word = ""

        for char in text:
            if is_chinese_char(char):
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            elif char.isalnum():
                current_word += char
            elif char.isspace():
                if current_word:
                    words.append(current_word)
                    current_word = ""
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                if char.strip():
                    words.append(char)

        if current_word:
            words.append(current_word)

        return [w for w in words if w.strip()]

    def compute_word_frequencies(self, texts: List[str]) -> None:
        """计算词频"""

        word_freqs = defaultdict(int)
        print("正在计算词频...")
        start_time = time.time()
        total_texts = len(texts)

        for i, text in enumerate(texts):
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
            if (i + 1) % 1000 == 0 or i == total_texts - 1:
                show_progress_with_stats(i + 1, total_texts, start_time, prefix="词频计算")

        original_vocab_size = len(word_freqs)
        chinese_chars = {
            word: freq
            for word, freq in word_freqs.items()
            if len(word) == 1 and is_chinese_char(word)
        }
        other_words = {
            word: freq
            for word, freq in word_freqs.items()
            if not (len(word) == 1 and is_chinese_char(word))
        }

        min_freq_chinese = 1
        min_freq_other = 2 if len(other_words) <= 30000 else max(2, len(other_words) // 15000)

        filtered_word_freqs: Dict[str, int] = {}
        filtered_word_freqs.update({word: freq for word, freq in chinese_chars.items() if freq >= min_freq_chinese})
        filtered_word_freqs.update({word: freq for word, freq in other_words.items() if freq >= min_freq_other})

        self.word_freqs = filtered_word_freqs
        elapsed = time.time() - start_time
        chinese_kept = len(
            [w for w in filtered_word_freqs if len(w) == 1 and is_chinese_char(w)]
        )
        print(
            f"✅ 词频计算完成! {len(filtered_word_freqs):,}词汇 (中文:{chinese_kept:,}/原始:{original_vocab_size:,}) 耗时:{elapsed:.1f}秒"
        )

    def initialize_splits(self) -> None:
        """初始化分割：将每个词分割成字符"""

        splits: Dict[str, List[str]] = {}
        for word in self.word_freqs:
            if len(word) == 1 and is_chinese_char(word):
                splits[word] = [word, "</w>"]
            else:
                splits[word] = [c for c in word] + ["</w>"]
        self.splits = splits
        print(f"✅ 初始化分割完成! {len(splits):,}个词")

    def compute_pair_frequencies(self) -> Dict[Tuple[str, str], int]:
        """计算相邻字符对的频率"""

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    def merge_vocab(self, pair: Tuple[str, str]) -> int:
        """合并词汇表中的字符对"""

        new_splits: Dict[str, List[str]] = {}
        merge_count = 0
        for word in self.word_freqs:
            split = self.splits[word]
            new_split: List[str] = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    merge_count += 1
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        self.splits = new_splits
        return merge_count

    def train(self, texts: List[str]) -> None:
        """训练BPE分词器"""

        print("🚀 开始训练BPE分词器（中文优化版）...")
        original_count = len(texts)
        max_texts = 150000
        if original_count > max_texts:
            print(f"⚠️  数据量过大({original_count:,})，采样到{max_texts:,}条")
            random.seed(42)
            texts = random.sample(texts, max_texts)

        total_start_time = time.time()
        self.compute_word_frequencies(texts)
        self.initialize_splits()

        merges: Dict[Tuple[str, str], int] = {}
        vocab_target = self.vocab_size - self._reserved_token_count
        if vocab_target <= 0:
            raise ValueError(
                f"vocab_size={self.vocab_size} 太小，至少需要 {self._reserved_token_count + 1} 以容纳特殊token和byte fallback。"
            )

        chinese_words = len([w for w in self.word_freqs if len(w) == 1 and is_chinese_char(w)])
        print(
            f"🔄 开始BPE训练: 目标{vocab_target:,}次合并, {len(self.word_freqs):,}词汇(中文:{chinese_words:,})"
        )

        merge_start_time = time.time()
        last_best_pair = None
        repeated_pair_count = 0
        no_progress_count = 0

        for i in range(vocab_target):
            pair_freqs = self.compute_pair_frequencies()
            if not pair_freqs:
                print(f"\n⚠️  没有更多的字符对可以合并，提前结束于第 {i + 1} 次合并")
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_pair == last_best_pair:
                repeated_pair_count += 1
                if repeated_pair_count >= 3:
                    break
            else:
                repeated_pair_count = 0

            if best_pair in merges:
                no_progress_count += 1
                if no_progress_count >= 10:
                    break
                continue
            else:
                no_progress_count = 0

            merge_count = self.merge_vocab(best_pair)
            if merge_count == 0:
                no_progress_count += 1
                if no_progress_count >= 10:
                    break
                continue
            else:
                no_progress_count = 0

            merges[best_pair] = i
            last_best_pair = best_pair

            if (i + 1) % 100 == 0 or i == vocab_target - 1:
                show_progress_with_stats(
                    i + 1,
                    vocab_target,
                    merge_start_time,
                    prefix="BPE训练",
                    extra_info=f"最新: '{best_pair[0]}'+'{best_pair[1]}'({best_freq:,})",
                )

        self.merges = merges
        merge_elapsed = time.time() - merge_start_time
        print(f"\n✅ BPE合并完成！实际合并次数: {len(merges):,} (耗时: {merge_elapsed:.2f}秒)")

        self.build_vocab()
        total_elapsed = time.time() - total_start_time
        chinese_tokens = len(
            [
                token
                for token in self.vocab
                if (len(token) == 1 and is_chinese_char(token))
                or (token.endswith("</w>") and len(token) > 4 and is_chinese_char(token[0]))
            ]
        )
        fallback_state = "启用" if self.enable_byte_fallback else "关闭"
        print(
            f"🎉 训练完成! 词汇表:{len(self.vocab):,} 中文:{chinese_tokens:,} ByteFallback:{fallback_state} 总耗时:{total_elapsed:.1f}秒"
        )

    def build_vocab(self) -> None:
        """构建词汇表"""

        vocab: Dict[str, int] = {}
        vocab[self.pad_token] = self.pad_id
        vocab[self.unk_token] = self.unk_id
        vocab[self.bos_token] = self.bos_id
        vocab[self.eos_token] = self.eos_id

        if self.enable_byte_fallback:
            for byte in range(256):
                token = self._format_byte_token(byte)
                if token not in vocab:
                    vocab[token] = len(vocab)
            vocab[self.byte_eow_token] = len(vocab)

        for word in self.word_freqs:
            for token in self.splits[word]:
                if token not in vocab:
                    vocab[token] = len(vocab)

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._sync_internal_state()
        self._validate_special_tokens(require_presence=True)
        self._update_checksum()

    def encode_word(self, word: str) -> List[str]:
        """编码单个词"""

        if len(word) == 1 and is_chinese_char(word):
            return [word + "</w>"]

        tokens = [c for c in word] + ["</w>"]
        for pair, _ in sorted(self.merges.items(), key=lambda x: x[1]):
            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本为token ID列表"""

        words = self.pre_tokenize(text)
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.bos_id)

        for word in words:
            try:
                tokens = self.encode_word(word)
            except Exception:
                tokens = []

            mapped_ids: List[int] = []
            for token in tokens:
                token_id = self.vocab.get(token)
                if token_id is None:
                    mapped_ids = []
                    break
                mapped_ids.append(token_id)

            if not mapped_ids:
                fallback_ids = self._encode_as_bytes(word)
                token_ids.extend(fallback_ids)
            else:
                token_ids.extend(mapped_ids)

        if add_special_tokens:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """解码token ID列表为文本"""

        if not token_ids:
            return ""

        id_to_token = self._id_to_token or {v: k for k, v in self.vocab.items()}
        output: List[str] = []
        byte_buffer: bytearray = bytearray()

        def flush_byte_buffer() -> Optional[str]:
            if not byte_buffer:
                return None
            try:
                decoded = byte_buffer.decode("utf-8")
            except UnicodeDecodeError:
                decoded = byte_buffer.decode("latin-1")
            byte_buffer.clear()
            output.append(decoded)
            if self._should_add_space(decoded):
                output.append(" ")
            return decoded

        for token_id in token_ids:
            token = id_to_token.get(token_id)
            if token is None:
                continue
            if token in {self.pad_token, self.bos_token}:
                continue
            if token == self.eos_token:
                flush_byte_buffer()
                break

            if self.enable_byte_fallback and token_id in self.id_to_byte:
                byte_buffer.append(self.id_to_byte[token_id])
                continue

            if self.enable_byte_fallback and token == self.byte_eow_token:
                flush_byte_buffer()
                continue

            flush_byte_buffer()

            if token in {self.unk_token}:
                continue

            if token.endswith("</w>"):
                surface = token[:-4]
                output.append(surface)
                if self._should_add_space(surface):
                    output.append(" ")
            else:
                output.append(token)

        flush_byte_buffer()

        text = "".join(output)
        return self._postprocess_text(text)

    def save(self, path: str) -> None:
        """保存分词器"""

        self._validate_special_tokens(require_presence=True)
        payload_without_checksum = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "config": self.get_config(),
        }
        checksum = hashlib.sha256(pickle.dumps(payload_without_checksum, protocol=4)).hexdigest()
        payload_without_checksum["checksum"] = checksum

        with open(path, "wb") as f:
            pickle.dump(payload_without_checksum, f)

        self._checksum = checksum
        print(f"分词器已保存到: {path}")

    def load(self, path: str) -> None:
        """加载分词器"""

        with open(path, "rb") as f:
            data = pickle.load(f)

        expected_checksum = data.get("checksum")
        payload_for_checksum = {k: data[k] for k in data if k != "checksum"}
        if expected_checksum:
            actual_checksum = hashlib.sha256(pickle.dumps(payload_for_checksum, protocol=4)).hexdigest()
            if actual_checksum != expected_checksum:
                raise ValueError("Tokenizer checksum mismatch，文件可能已损坏或被篡改。")
            self._checksum = expected_checksum
        else:
            # 兼容旧版本
            self._checksum = ""

        self.vocab = data["vocab"]
        self.merges = data.get("merges", {})
        self.vocab_size = data.get("vocab_size", len(self.vocab))

        config = data.get("config") or {}
        self.lowercase = config.get("lowercase", True)
        self.normalize_nfkc = config.get("normalize_nfkc", True)
        self.strip_spaces = config.get("strip_spaces", True)
        self.enable_byte_fallback = config.get("enable_byte_fallback", False)
        self._update_reserved_token_count()

        if self.enable_byte_fallback and self.byte_eow_token not in self.vocab:
            # 旧分词器不包含字节token，自动禁用
            self.enable_byte_fallback = False
            self._update_reserved_token_count()

        self._sync_internal_state()
        expected_special = config.get("special_tokens") if config else None
        if expected_special:
            mismatches = self.diff_special_tokens(expected_special)
            if mismatches:
                raise ValueError(
                    "Tokenizer special tokens mismatch: "
                    + ", ".join(
                        f"{name}: ckpt={exp} vs file={act}" for name, (exp, act) in mismatches.items()
                    )
                )
        self._validate_special_tokens(require_presence=True)
        if not self._checksum:
            self._update_checksum()

        print(f"分词器已加载: {path}")

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""

        return len(self.vocab)

    def get_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "lowercase": self.lowercase,
            "normalize_nfkc": self.normalize_nfkc,
            "strip_spaces": self.strip_spaces,
            "enable_byte_fallback": self.enable_byte_fallback,
            "normalizer_signature": self._normalizer_signature(),
            "special_tokens": self.special_tokens_map(require_presence=bool(self.vocab)),
        }

    def checksum(self) -> str:
        return self._checksum

    def special_tokens_map(self, *, require_presence: bool = True) -> Dict[str, Dict[str, Any]]:
        self._validate_special_tokens(require_presence=require_presence, strict=require_presence)
        mapping: Dict[str, Dict[str, Any]] = {}
        for name, token, fallback_id in [
            ("pad", self.pad_token, self.pad_id),
            ("unk", self.unk_token, self.unk_id),
            ("bos", self.bos_token, self.bos_id),
            ("eos", self.eos_token, self.eos_id),
        ]:
            vocab_id = self.vocab.get(token)
            mapping[name] = {
                "token": token,
                "id": vocab_id if vocab_id is not None else fallback_id,
                "present": vocab_id is not None,
            }
        if self.enable_byte_fallback:
            vocab_id = self.vocab.get(self.byte_eow_token)
            mapping["byte_eow"] = {
                "token": self.byte_eow_token,
                "id": vocab_id if vocab_id is not None else self.byte_eow_id,
                "present": vocab_id is not None,
            }
        return mapping

    def diff_special_tokens(self, expected: Dict[str, Dict[str, Any]]) -> Dict[str, tuple[Any, Any]]:
        if not expected:
            return {}
        actual = self.special_tokens_map(require_presence=False)
        mismatches: Dict[str, tuple[Any, Any]] = {}
        for name, expected_info in expected.items():
            actual_info = actual.get(name)
            if actual_info is None:
                mismatches[name] = (expected_info, None)
                continue
            keys_to_compare = {
                key for key in ("token", "id", "present") if key in expected_info
            }
            if not keys_to_compare:
                keys_to_compare = {"token", "id"}
            comparison = {k: expected_info.get(k) for k in keys_to_compare}
            actual_comp = {k: actual_info.get(k) for k in keys_to_compare}
            if comparison != actual_comp:
                mismatches[name] = (expected_info, actual_info)
        return mismatches

    def compute_unk_statistics(
        self, texts: Iterable[str], sample_size: int = 1000
    ) -> Dict[str, Any]:
        total_tokens = 0
        unk_tokens = 0
        sampled = 0
        for text in texts:
            if sample_size and sampled >= sample_size:
                break
            sampled += 1
            token_ids = self.encode(text, add_special_tokens=False)
            total_tokens += len(token_ids)
            unk_tokens += sum(1 for tid in token_ids if tid == self.unk_id)
        unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0
        return {
            "sampled_texts": sampled,
            "total_tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "unk_rate": unk_rate,
        }

    def compute_unk_rate(self, texts: Iterable[str], sample_size: int = 1000) -> float:
        return self.compute_unk_statistics(texts, sample_size)["unk_rate"]


def train_tokenizer_from_data(data_path: str, vocab_size: int = 30000) -> BPETokenizer:
    """从数据文件训练分词器"""

    print(f"📁 加载数据: {data_path}, 目标词汇表: {vocab_size:,}")

    texts: List[str] = []
    start_time = time.time()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "conversations" in data:
                for conv in data["conversations"]:
                    texts.append(conv["content"])
            elif "text" in data:
                texts.append(data["text"])

    elapsed = time.time() - start_time
    chinese_char_count = sum(len([c for c in text if is_chinese_char(c)]) for text in texts)
    total_char_count = sum(len(text) for text in texts)
    chinese_ratio = chinese_char_count / total_char_count if total_char_count > 0 else 0

    print(f"✅ 读取 {len(texts):,} 条文本，中文占比 {chinese_ratio:.1%} (耗时: {elapsed:.2f}秒)")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer


if __name__ == "__main__":
    sample_texts = [
        "Hello world! How are you?",
        "I am learning about BPE tokenization.",
        "This is a sample text for training.",
        "BPE stands for Byte Pair Encoding.",
    ]

    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)

    test_text = "Hello! How are you doing?"
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)

    print(f"原文: {test_text}")
    print(f"编码: {token_ids}")
    print(f"解码: {decoded_text}")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")
