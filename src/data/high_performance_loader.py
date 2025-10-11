"""
高性能数据加载系统
支持流式加载、智能缓存、并行处理、内存优化
"""
import hashlib
import json
import os
import pickle
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


@dataclass
class DataLoadingConfig:
    """高性能数据加载配置"""
    # 基础配置
    data_path: str
    max_length: int = 512
    batch_size: int = 32

    # 性能配置
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "data_cache"
    force_rebuild_cache: bool = False

    # 流式加载配置
    streaming: bool = True
    chunk_size: int = 10000  # 每个chunk的样本数
    buffer_size: int = 50000  # 内存中最大样本数

    # 预处理配置
    parallel_processing: bool = True
    max_parallel_workers: int = 8


class StreamingJsonLoader:
    """流式JSON加载器，支持大文件高效处理"""

    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.file_size = os.path.getsize(file_path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """流式迭代JSON行"""
        with open(self.file_path, encoding='utf-8') as f:
            buffer = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    buffer.append(data)

                    if len(buffer) >= self.chunk_size:
                        yield from buffer
                        buffer = []

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    continue

            # 处理最后的缓冲区
            if buffer:
                yield from buffer

    def get_chunks(self) -> Iterator[list[dict[str, Any]]]:
        """以chunk形式返回数据"""
        buffer = []
        for item in self:
            buffer.append(item)
            if len(buffer) >= self.chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer


class IntelligentDataCache:
    """智能数据缓存系统"""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # 元数据文件
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_metadata(self):
        """保存缓存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _get_file_hash(self, file_path: str, sample_lines: int = 1000) -> str:
        """计算文件hash值，使用采样提高效率"""
        hash_obj = hashlib.md5()

        # 添加文件基本信息
        stat = os.stat(file_path)
        hash_obj.update(f"{stat.st_size}_{stat.st_mtime}".encode())

        # 采样文件内容
        with open(file_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                hash_obj.update(line.encode('utf-8'))

        return hash_obj.hexdigest()[:16]

    def _get_cache_key(self, config: DataLoadingConfig) -> str:
        """生成缓存键"""
        file_hash = self._get_file_hash(config.data_path)
        config_str = f"{config.max_length}_{config.chunk_size}"
        return f"{file_hash}_{hashlib.md5(config_str.encode()).hexdigest()[:8]}"

    def get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def is_cache_valid(self, config: DataLoadingConfig) -> bool:
        """检查缓存是否有效"""
        if config.force_rebuild_cache:
            return False

        cache_key = self._get_cache_key(config)
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return False

        # 检查元数据
        if cache_key not in self.metadata:
            return False

        metadata = self.metadata[cache_key]

        # 检查源文件是否修改
        current_hash = self._get_file_hash(config.data_path)
        if metadata.get('file_hash') != current_hash:
            return False

        return True

    def load_cache(self, config: DataLoadingConfig) -> list[dict[str, Any]] | None:
        """加载缓存数据"""
        if not self.is_cache_valid(config):
            return None

        cache_key = self._get_cache_key(config)
        cache_path = self.get_cache_path(cache_key)

        try:
            print(f"📦 Loading cached data: {cache_path.name}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            metadata = self.metadata[cache_key]
            print(f"✅ Cache loaded: {len(data)} samples, "
                  f"created {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_at']))}")

            return data

        except Exception as e:
            print(f"⚠️  Cache loading failed: {e}")
            return None

    def save_cache(self, config: DataLoadingConfig, data: list[dict[str, Any]]):
        """保存数据到缓存"""
        cache_key = self._get_cache_key(config)
        cache_path = self.get_cache_path(cache_key)

        try:
            print(f"💾 Saving data to cache: {cache_path.name}")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 更新元数据
            self.metadata[cache_key] = {
                'file_hash': self._get_file_hash(config.data_path),
                'data_path': config.data_path,
                'max_length': config.max_length,
                'sample_count': len(data),
                'created_at': time.time(),
                'cache_size': cache_path.stat().st_size
            }
            self._save_metadata()

            print(f"✅ Cache saved: {len(data)} samples, "
                  f"size: {cache_path.stat().st_size / 1024 / 1024:.1f}MB")

        except Exception as e:
            print(f"❌ Cache saving failed: {e}")

    def clean_old_cache(self, keep_recent: int = 5):
        """清理旧缓存"""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        if len(cache_files) <= keep_recent:
            return

        # 按修改时间排序
        cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # 删除旧文件
        for cache_file in cache_files[keep_recent:]:
            try:
                cache_file.unlink()
                print(f"🗑️  Removed old cache: {cache_file.name}")
            except Exception as e:
                print(f"⚠️  Failed to remove {cache_file.name}: {e}")

        # 清理元数据
        valid_keys = {f.stem for f in cache_files[:keep_recent]}
        self.metadata = {k: v for k, v in self.metadata.items() if k in valid_keys}
        self._save_metadata()


class ParallelDataProcessor:
    """并行数据处理器"""

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers

    def process_conversations(self, data_chunks: list[list[dict]],
                            max_length: int) -> list[dict[str, Any]]:
        """并行处理对话数据"""
        if len(data_chunks) == 1:
            # 单个chunk，直接处理
            return self._process_conversation_chunk(data_chunks[0], max_length)

        # 多个chunk，并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_conversation_chunk, chunk, max_length)
                for chunk in data_chunks
            ]

            results = []
            for future in tqdm(futures, desc="Processing chunks"):
                results.extend(future.result())

            return results

    def _process_conversation_chunk(self, chunk: list[dict],
                                  max_length: int) -> list[dict[str, Any]]:
        """处理单个对话数据chunk"""
        processed = []

        for item in chunk:
            if 'conversations' in item:
                conversations = item['conversations']
                if len(conversations) >= 2:
                    user_input = ""
                    assistant_output = ""

                    for conv in conversations:
                        if conv['role'] == 'user':
                            user_input = conv['content']
                        elif conv['role'] == 'assistant':
                            assistant_output = conv['content']

                    if user_input and assistant_output:
                        total_length = len(user_input) + len(assistant_output)
                        if total_length <= max_length:
                            processed.append({
                                'input': user_input,
                                'output': assistant_output,
                                'length': total_length,
                                'type': 'conversation'
                            })

        return processed

    def process_pretrain_texts(self, data_chunks: list[list[dict]],
                             max_length: int) -> list[str]:
        """并行处理预训练文本数据"""
        if len(data_chunks) == 1:
            return self._process_pretrain_chunk(data_chunks[0], max_length)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_pretrain_chunk, chunk, max_length)
                for chunk in data_chunks
            ]

            results = []
            for future in tqdm(futures, desc="Processing text chunks"):
                results.extend(future.result())

            return results

    def _process_pretrain_chunk(self, chunk: list[dict],
                              max_length: int) -> list[str]:
        """处理单个预训练文本chunk"""
        texts = []

        for item in chunk:
            if 'text' in item:
                text = item['text']
                if len(text) <= max_length:
                    texts.append(text)

        return texts


class HighPerformanceDataset(IterableDataset):
    """高性能可迭代数据集"""

    def __init__(self, config: DataLoadingConfig, tokenizer, data_type: str = "sft"):
        self.config = config
        self.tokenizer = tokenizer
        self.data_type = data_type

        # 初始化缓存和处理器
        self.cache = IntelligentDataCache(config.cache_dir) if config.enable_cache else None
        self.processor = ParallelDataProcessor(config.max_parallel_workers)

        # 加载或处理数据
        self.data = self._load_or_process_data()

    def _load_or_process_data(self) -> list[dict[str, Any]]:
        """加载或处理数据"""
        # 尝试从缓存加载
        if self.cache and self.cache.is_cache_valid(self.config):
            cached_data = self.cache.load_cache(self.config)
            if cached_data is not None:
                return cached_data

        # 处理原始数据
        print(f"🔄 Processing data from: {self.config.data_path}")
        loader = StreamingJsonLoader(self.config.data_path, self.config.chunk_size)

        # 收集数据chunks
        data_chunks = list(loader.get_chunks())
        print(f"📊 Loaded {len(data_chunks)} chunks for processing")

        # 并行处理
        if self.data_type == "sft":
            processed_data = self.processor.process_conversations(data_chunks, self.config.max_length)
        elif self.data_type == "pretrain":
            processed_data = self.processor.process_pretrain_texts(data_chunks, self.config.max_length)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        print(f"✅ Processed {len(processed_data)} samples")

        # 保存到缓存
        if self.cache:
            self.cache.save_cache(self.config, processed_data)

        return processed_data

    def __iter__(self):
        """迭代数据"""
        if self.config.streaming and len(self.data) > self.config.buffer_size:
            # 流式模式：随机采样数据避免内存溢出
            indices = torch.randperm(len(self.data))
            for i in indices:
                yield self._process_item(self.data[i])
        else:
            # 普通模式：直接迭代
            for item in self.data:
                yield self._process_item(item)

    def _process_item(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        """处理单个数据项"""
        if self.data_type == "sft":
            return self._process_conversation_item(item)
        elif self.data_type == "pretrain":
            return self._process_pretrain_item(item)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def _process_conversation_item(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        """处理对话数据项"""
        input_text = item['input']
        output_text = item['output']

        # 编码文本
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)

        # 构造完整序列
        full_sequence = [self.tokenizer.bos_id] + input_ids + output_ids + [self.tokenizer.eos_id]

        # 截断和填充
        if len(full_sequence) > self.config.max_length:
            full_sequence = full_sequence[:self.config.max_length]

        labels = full_sequence[1:] + [self.tokenizer.pad_id]

        # 填充到固定长度
        while len(full_sequence) < self.config.max_length:
            full_sequence.append(self.tokenizer.pad_id)
            labels.append(self.tokenizer.pad_id)

        return {
            'input_ids': torch.tensor(full_sequence, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1 if x != self.tokenizer.pad_id else 0 for x in full_sequence], dtype=torch.long)
        }

    def _process_pretrain_item(self, text: str) -> torch.Tensor:
        """处理预训练文本项"""
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # 确保至少有2个token
        if len(token_ids) < 2:
            token_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]

        # 截断或填充
        if len(token_ids) > self.config.max_length:
            token_ids = token_ids[:self.config.max_length]
        else:
            token_ids.extend([self.tokenizer.pad_id] * (self.config.max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)


def create_high_performance_dataloader(
    config: DataLoadingConfig,
    tokenizer,
    data_type: str = "sft"
) -> DataLoader:
    """创建高性能数据加载器"""

    dataset = HighPerformanceDataset(config, tokenizer, data_type)

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=config.num_workers > 0
    )

    return dataloader


def benchmark_data_loading(config: DataLoadingConfig, tokenizer, iterations: int = 100):
    """基准测试数据加载性能"""
    print("🔬 Benchmarking data loading performance...")
    print(f"   Config: batch_size={config.batch_size}, num_workers={config.num_workers}")
    print(f"   Cache: {'enabled' if config.enable_cache else 'disabled'}")

    # 创建数据加载器
    dataloader = create_high_performance_dataloader(config, tokenizer, "sft")

    # 预热
    print("🔥 Warming up...")
    for i, _batch in enumerate(dataloader):
        if i >= 5:
            break

    # 基准测试
    print(f"⏱️  Running {iterations} iterations...")
    start_time = time.time()
    samples_processed = 0

    for i, batch in enumerate(dataloader):
        if i >= iterations:
            break
        samples_processed += len(batch['input_ids'])

    end_time = time.time()
    elapsed = end_time - start_time

    # 结果
    throughput = samples_processed / elapsed
    print("📊 Benchmark Results:")
    print(f"   Processed: {samples_processed} samples in {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    print(f"   Avg batch time: {elapsed / iterations * 1000:.1f}ms")

    return {
        'throughput': throughput,
        'elapsed_time': elapsed,
        'samples_processed': samples_processed,
        'avg_batch_time': elapsed / iterations
    }


if __name__ == "__main__":
    # 测试高性能数据加载系统
    print("🧪 Testing high-performance data loading system...")

    # 配置
    config = DataLoadingConfig(
        data_path="data/dataset/minimind_dataset/sft_mini_512.jsonl",
        max_length=512,
        batch_size=16,
        num_workers=4,
        enable_cache=True,
        streaming=True,
        parallel_processing=True
    )

    print(f"Config: {asdict(config)}")
