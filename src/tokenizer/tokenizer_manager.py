"""
通用Tokenizer管理系统

实现智能缓存机制，避免重复训练，提供统一API接口
- 基于数据文件hash和配置自动判断是否需要重新训练
- 支持多种tokenizer配置的管理
- 提供标准化的加载和训练接口
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bpe_tokenizer import BPETokenizer, train_tokenizer_from_data


@dataclass
class TokenizerConfig:
    """Tokenizer配置"""

    vocab_size: int = 30000
    tokenizer_type: str = "bpe"  # 支持扩展其他类型
    max_samples: int = 150000  # 训练时最大样本数

    def __post_init__(self):
        """验证配置"""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.tokenizer_type not in ["bpe"]:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def get_cache_key(self) -> str:
        """获取缓存键"""
        return f"{self.tokenizer_type}_vocab{self.vocab_size}_max{self.max_samples}"


class TokenizerManager:
    """
    Tokenizer智能管理系统

    功能：
    - 自动缓存已训练的tokenizer
    - 基于数据hash判断是否需要重新训练
    - 提供统一的API接口
    - 支持多种配置管理
    """

    def __init__(self, cache_dir: str = "tokenizers"):
        """
        初始化TokenizerManager

        Args:
            cache_dir: tokenizer缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # 创建子目录
        (self.cache_dir / "models").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)

        print(f"🔧 TokenizerManager initialized: {self.cache_dir}")

    def _get_data_hash(self, data_path: str, max_lines: int = 10000) -> str:
        """
        计算数据文件的hash值（采样前N行以提高效率）

        Args:
            data_path: 数据文件路径
            max_lines: 最大采样行数

        Returns:
            数据hash值
        """
        hash_obj = hashlib.md5()

        try:
            with open(data_path, encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    if line_count >= max_lines:
                        break
                    hash_obj.update(line.encode("utf-8"))
                    line_count += 1

            # 添加文件大小和修改时间信息
            stat = os.stat(data_path)
            hash_obj.update(str(stat.st_size).encode())
            hash_obj.update(str(stat.st_mtime).encode())

        except Exception as e:
            print(f"⚠️  Warning: Error computing data hash: {e}")
            # 如果出错，使用文件路径作为fallback
            hash_obj.update(data_path.encode("utf-8"))

        return hash_obj.hexdigest()[:12]  # 使用前12位

    def _get_tokenizer_path(self, config: TokenizerConfig, data_hash: str) -> tuple[Path, Path]:
        """
        获取tokenizer模型文件和元数据文件路径

        Args:
            config: tokenizer配置
            data_hash: 数据hash值

        Returns:
            (model_path, metadata_path)
        """
        cache_key = config.get_cache_key()
        filename = f"{cache_key}_{data_hash}"

        model_path = self.cache_dir / "models" / f"{filename}.pkl"
        metadata_path = self.cache_dir / "metadata" / f"{filename}.json"

        return model_path, metadata_path

    def _save_metadata(
        self, metadata_path: Path, config: TokenizerConfig, data_path: str, data_hash: str
    ):
        """保存tokenizer元数据"""
        metadata = {
            "config": config.to_dict(),
            "data_path": str(data_path),
            "data_hash": data_hash,
            "created_at": __import__("time").time(),
            "version": "1.0",
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any] | None:
        """加载tokenizer元数据"""
        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _is_cache_valid(
        self, model_path: Path, metadata_path: Path, config: TokenizerConfig, data_hash: str
    ) -> bool:
        """检查缓存是否有效"""
        if not model_path.exists() or not metadata_path.exists():
            return False

        metadata = self._load_metadata(metadata_path)
        if not metadata:
            return False

        # 检查配置是否匹配
        cached_config = TokenizerConfig(**metadata["config"])
        if cached_config != config:
            return False

        # 检查数据hash是否匹配
        if metadata.get("data_hash") != data_hash:
            return False

        return True

    def get_or_train_tokenizer(
        self,
        data_path: str,
        config: TokenizerConfig | dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> BPETokenizer:
        """
        获取或训练tokenizer（核心API）

        Args:
            data_path: 训练数据路径
            config: tokenizer配置
            force_retrain: 是否强制重新训练

        Returns:
            训练好的tokenizer
        """
        # 处理配置
        if config is None:
            config = TokenizerConfig()
        elif isinstance(config, dict):
            config = TokenizerConfig(**config)

        print(f"📝 Tokenizer request: {config.get_cache_key()}")

        # 计算数据hash
        print("🔍 Computing data fingerprint...")
        data_hash = self._get_data_hash(data_path)
        print(f"   Data fingerprint: {data_hash}")

        # 获取文件路径
        model_path, metadata_path = self._get_tokenizer_path(config, data_hash)

        # 检查缓存
        if not force_retrain and self._is_cache_valid(model_path, metadata_path, config, data_hash):
            print(f"✅ Loading cached tokenizer: {model_path.name}")
            tokenizer = BPETokenizer(vocab_size=config.vocab_size)
            tokenizer.load(str(model_path))

            # 显示缓存信息
            metadata = self._load_metadata(metadata_path)
            if metadata:
                created_time = metadata.get("created_at", 0)
                print(
                    f"   Cached on: {__import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime(created_time))}"
                )
                print(f"   Vocab size: {tokenizer.get_vocab_size()}")

            return tokenizer

        # 需要重新训练
        print(f"🚀 Training new tokenizer: {config.get_cache_key()}")
        if force_retrain:
            print("   Reason: Force retrain requested")
        else:
            print("   Reason: No valid cache found")

        # 训练tokenizer
        tokenizer = train_tokenizer_from_data(data_path, config.vocab_size)

        # 保存到缓存
        print(f"💾 Caching tokenizer: {model_path.name}")
        tokenizer.save(str(model_path))
        self._save_metadata(metadata_path, config, data_path, data_hash)

        print(f"✅ Tokenizer ready: vocab_size={tokenizer.get_vocab_size()}")

        return tokenizer

    def list_cached_tokenizers(self) -> list[dict[str, Any]]:
        """列出所有缓存的tokenizer"""
        cached_tokenizers = []

        metadata_dir = self.cache_dir / "metadata"
        if not metadata_dir.exists():
            return cached_tokenizers

        for metadata_file in metadata_dir.glob("*.json"):
            metadata = self._load_metadata(metadata_file)
            if metadata:
                # 检查对应的模型文件是否存在
                model_file = self.cache_dir / "models" / f"{metadata_file.stem}.pkl"
                if model_file.exists():
                    cached_tokenizers.append(
                        {
                            "name": metadata_file.stem,
                            "config": metadata["config"],
                            "data_path": metadata["data_path"],
                            "created_at": metadata.get("created_at", 0),
                            "model_size": model_file.stat().st_size,
                        }
                    )

        return sorted(cached_tokenizers, key=lambda x: x["created_at"], reverse=True)

    def clean_cache(self, keep_latest: int = 5) -> int:
        """
        清理旧的缓存文件

        Args:
            keep_latest: 保留最新的N个tokenizer

        Returns:
            删除的文件数量
        """
        cached = self.list_cached_tokenizers()

        if len(cached) <= keep_latest:
            print(f"💾 Cache clean: {len(cached)} tokenizers, nothing to clean")
            return 0

        to_delete = cached[keep_latest:]
        deleted_count = 0

        for item in to_delete:
            name = item["name"]

            # 删除模型文件
            model_file = self.cache_dir / "models" / f"{name}.pkl"
            if model_file.exists():
                model_file.unlink()
                deleted_count += 1

            # 删除元数据文件
            metadata_file = self.cache_dir / "metadata" / f"{name}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                deleted_count += 1

        print(
            f"🧹 Cache cleaned: removed {deleted_count} files, kept latest {keep_latest} tokenizers"
        )
        return deleted_count


# 全局tokenizer管理器实例
_global_manager = None


def get_global_tokenizer_manager() -> TokenizerManager:
    """获取全局tokenizer管理器"""
    global _global_manager
    if _global_manager is None:
        _global_manager = TokenizerManager()
    return _global_manager


def get_tokenizer(
    data_path: str,
    vocab_size: int = 30000,
    tokenizer_type: str = "bpe",
    force_retrain: bool = False,
    cache_dir: str | None = None,
) -> BPETokenizer:
    """
    通用tokenizer获取API（推荐使用）

    Args:
        data_path: 训练数据路径
        vocab_size: 词汇表大小
        tokenizer_type: tokenizer类型
        force_retrain: 是否强制重新训练
        cache_dir: 缓存目录（None使用全局管理器）

    Returns:
        训练好的tokenizer

    Example:
        # 简单使用
        tokenizer = get_tokenizer("data/train.jsonl", vocab_size=30000)

        # 强制重新训练
        tokenizer = get_tokenizer("data/train.jsonl", force_retrain=True)
    """
    config = TokenizerConfig(vocab_size=vocab_size, tokenizer_type=tokenizer_type)

    if cache_dir is not None:
        # 使用指定的缓存目录
        manager = TokenizerManager(cache_dir)
    else:
        # 使用全局管理器
        manager = get_global_tokenizer_manager()

    return manager.get_or_train_tokenizer(data_path, config, force_retrain)


def list_tokenizers() -> list[dict[str, Any]]:
    """列出所有缓存的tokenizer"""
    manager = get_global_tokenizer_manager()
    return manager.list_cached_tokenizers()


def clean_tokenizer_cache(keep_latest: int = 5) -> int:
    """清理tokenizer缓存"""
    manager = get_global_tokenizer_manager()
    return manager.clean_cache(keep_latest)


if __name__ == "__main__":
    # 测试代码
    print("🧪 Testing TokenizerManager...")

    # 创建测试数据
    test_data = [
        '{"conversations": [{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi there!"}]}',
        '{"conversations": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I am fine"}]}',
    ]

    test_file = "test_data.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for line in test_data:
            f.write(line + "\n")

    try:
        # 测试通用API
        print("\n1. Testing get_tokenizer API...")
        tokenizer = get_tokenizer(test_file, vocab_size=1000)
        print(f"   Tokenizer vocab size: {tokenizer.get_vocab_size()}")

        # 第二次调用应该使用缓存
        print("\n2. Testing cache functionality...")
        tokenizer2 = get_tokenizer(test_file, vocab_size=1000)
        print(f"   Tokenizer2 vocab size: {tokenizer2.get_vocab_size()}")

        # 列出缓存
        print("\n3. Listing cached tokenizers...")
        cached = list_tokenizers()
        for item in cached:
            print(f"   - {item['name']}: vocab={item['config']['vocab_size']}")

        print("\n✅ TokenizerManager test completed!")

    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
