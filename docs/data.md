# 🗂️ 数据与分词

`src/data` 与 `src/tokenizer` 提供了训练 Mini-LLM 所需的数据加载与分词工具。本篇文档介绍数据格式、加载器以及分词器管理方式。

## 数据集配置 (`DatasetConfig`)
- `data_path`：JSONL 数据路径
- `max_length`：过滤或截断的最大字符数/Token 数
- `train_split`：训练/验证拆分比例
- `shuffle`：是否在划分前打乱

示例：
```python
from src.data.dataset_loader import DatasetConfig
config = DatasetConfig(data_path="data/sft_mini.jsonl", max_length=512)
```

## 对话与 SFT 数据 (`ConversationDataLoader`)
- 期望 JSONL 中每行包含 `{"conversations": [...]}`
- conversations 至少包含一条 `role="user"` 与一条 `role="assistant"`
- 输出字典 `{"input": user_text, "output": assistant_text, "length": ...}`
- `get_train_test_split` 会依据 `train_split` 进行拆分

## 预训练文本 (`PretrainDataLoader`)
- JSONL 每行包含 `{"text": "..."}`
- 根据 `max_length` 过滤过长样本
- 返回纯文本列表，可直接传入 `LanguageModelingDataset`

## 偏好/对比数据 (`DPODataLoader`)
- JSONL 每行包含 `prompt`、`chosen`、`rejected`
- 常用于 Direct Preference Optimization (DPO) 或奖励模型训练

## 分词器 (`tokenizer` 模块)
### BPETokenizer
- 提供中文友好的预分词与 BPE 合并策略
- 默认特殊 token：`<PAD>/<UNK>/<BOS>/<EOS>`
- `train(texts)`：基于内存中的语料训练词表
- `encode(text)` / `decode(ids)`：在训练与推理流程中使用

### TokenizerManager
- 自动为给定数据路径计算 hash，避免重复训练
- 缓存路径默认位于 `tokenizers/`
- `get_or_train_tokenizer(data_path, config)`：若缓存存在则直接加载，否则调用 `train_tokenizer_from_data`

示例：
```python
from src.tokenizer.tokenizer_manager import TokenizerManager, TokenizerConfig

manager = TokenizerManager()
tokenizer = manager.get_or_train_tokenizer(
    data_path="data/sft_mini.jsonl",
    config=TokenizerConfig(vocab_size=20000)
)
```

## 与训练循环的配合
- `LanguageModelingDataset` 会自动补齐 PAD，并确保最少两个 token
- `ConversationDataset` 返回 `input_ids` 与 `labels`，适配 `SFTTrainer`
- 训练时请确保分词器的特殊 token ID 与 `MiniGPTConfig` 中保持一致

若数据格式发生变化，只需扩展相应 DataLoader，并在文档中同步说明字段含义，即可保持代码与文档一致。
