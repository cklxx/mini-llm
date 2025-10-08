# 第 03 章 · 预训练与数据流程

mini-llm 的数据流水线兼顾了实验便利性与工程可扩展性。本章会从数据格式、Tokenizer 管理、到高性能加载逐步展开，配合 `src/data` 与 `src/tokenizer` 目录阅读效果更佳。

## 3.1 数据配置与拆分
- `DatasetConfig` 管理常用参数：`data_path`、`max_length`、`train_split`，初始化后可直接传入各类 loader。
- `ConversationDataLoader`、`PretrainDataLoader`、`DPODataLoader` 分别处理 SFT、纯文本预训练、偏好数据。每个 loader 都会在 `load_*` 方法里打印样本统计，方便快速校验数据质量。
- 训练/验证划分使用 `get_train_test_split`，默认随机打乱并按 `train_split` 切分。

## 3.2 数据格式约定
- **SFT 对话**：JSONL 中需包含 `conversations` 字段，mini-llm 会自动抓取最后一次 user/assistant 配对。
- **预训练文本**：JSONL 的 `text` 字段会被读取，并在 `max_length` 处截断。
- **DPO 数据**：要求 `prompt`、`chosen`、`rejected` 三字段，便于 RLHF 阶段直接使用。

## 3.3 Tokenizer 智能管理
- `TokenizerManager` 通过数据文件的哈希值判断是否需要重训 tokenizer，并将模型与元数据缓存在 `tokenizers/models`、`tokenizers/metadata`。
- `TokenizerConfig` 暴露 `vocab_size`、`max_samples` 等关键信息，`get_cache_key()` 会拼出缓存命名规则。
- 训练入口 `get_or_train_tokenizer`：传入数据路径即可自动复用缓存或触发 `train_tokenizer_from_data`。

## 3.4 高性能数据加载
- `DataLoadingConfig` 支持缓存、流式读取和并行预处理等选项，适合大规模预训练场景。
- `StreamingJsonLoader` 提供 chunk 化迭代器，避免一次性读入超大 JSONL。
- `IntelligentDataCache` 利用文件 hash + 配置组合成缓存键，自动维护缓存元数据，能显著缩短多次实验的准备时间。

## 深度拆解章节
- [3.1 语言建模的概率建模](01-语言建模概率基础/README.md)
- [3.2 自回归与因果掩码](02-自回归建模与因果掩码/README.md)
- [3.3 分词策略与信息压缩](03-分词策略与信息压缩/README.md)
- [3.4 数据加载与批处理](04-数据加载与批处理/README.md)

## 3.5 与训练循环协同
- `LanguageModelingDataset` 和 `ConversationDataset` 在 `src/training/trainer.py` 中实现，可直接接入 `torch.utils.data.DataLoader`。
- 预训练场景下，`__getitem__` 会输出固定长度的 `torch.long` 张量，并自动补齐 `<pad>` token，确保批次对齐。
- 上游数据加载器返回的 Python 对象可直接喂入上述 Dataset，构成“加载 → 切分 → tokenize → DataLoader” 的闭环。

> 实践提示：在首次实验时先运行 `TokenizerManager.get_or_train_tokenizer(..., force_retrain=True)` 生成干净缓存，后续即可享受快速复用的好处。
