# 🗂️ 数据与分词

`src/data` 与 `src/tokenizer` 提供了训练 Mini-LLM 所需的数据加载与分词工具。本篇文档介绍数据格式、加载器、分词器管理方式，并给出常见坑位与排查建议。

## 数据集配置 (`DatasetConfig`)

`DatasetConfig` 以 dataclass 形式封装路径、长度截断及拆分策略，构造时只需给出数据文件即可按默认策略过滤和打乱样本。【F:src/data/dataset_loader.py†L10-L96】

| 字段 | 含义 | 默认值 | 备注 |
| ---- | ---- | ------ | ---- |
| `data_path` | JSONL 文件路径 | 必填 | 可为相对路径或绝对路径 |
| `max_length` | 过滤或截断时允许的最大字符/Token 数 | `512` | `ConversationDataLoader` 会用 `input+output` 长度过滤，`PretrainDataLoader` 会按文本长度过滤 |
| `train_split` | 训练/验证拆分比例 | `0.9` | `get_train_test_split` 基于该比例划分 |
| `shuffle` | 划分前是否先打乱样本 | `True` | 当样本自带排序时务必保留随机打乱 |

示例：

```python
from src.data.dataset_loader import DatasetConfig

config = DatasetConfig(data_path="data/sft_mini.jsonl", max_length=512)
```

## JSONL 数据结构速查

| 任务 | 必备字段 | 示例 |
| ---- | -------- | ---- |
| 监督微调/对话 | `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}` | 确保至少包含一轮 user→assistant；若存在 system 消息也会被保留 |
| 预训练语言建模 | `{"text": "纯文本"}` | 仅保留 `text` 字段，超长样本会被过滤 |
| 偏好/奖励 | `{"prompt": "...", "chosen": "...", "rejected": "..."}` | `chosen/rejected` 均需可被分词器编码 |

## 对话与 SFT 数据 (`ConversationDataLoader`)

`ConversationDataLoader` 会遍历 JSONL 文件，抽取每行里的 user/assistant 内容拼接成输入输出对，并记录样本长度；超出 `max_length` 的样本会直接丢弃，最终返回 `[{"input": ..., "output": ..., "length": ...}, ...]` 结构的列表。【F:src/data/dataset_loader.py†L33-L85】

常见注意事项：

- 如果 conversations 中缺少任一角色，会被过滤掉；可在数据预处理阶段补齐空 assistant 回复避免浪费样本。【F:src/data/dataset_loader.py†L55-L76】
- 在调用 `get_train_test_split` 前若想保持时间序列，请将 `shuffle=False`；否则默认会打乱后再按比例切分，保证验证集覆盖不同主题。【F:src/data/dataset_loader.py†L86-L96】

## 预训练文本 (`PretrainDataLoader`)

预训练数据读取逻辑会逐行解析 JSONL，只有包含 `text` 且长度不超过 `max_length` 的样本才会保留下来，确保后续语言建模数据集不会超出上下文窗口。【F:src/data/dataset_loader.py†L99-L130】

## 偏好/对比数据 (`DPODataLoader`)

当需要进行 Direct Preference Optimization 或奖励模型训练时，可使用 `DPODataLoader` 获取成对对比样本；加载器会校验 `prompt/chosen/rejected` 是否全部存在，缺失字段会被自动跳过。【F:src/data/dataset_loader.py†L132-L162】

## 与训练管线的衔接

- `DataResolver` 会按照训练模式（`pretrain`/`sft`/`dpo`/`rlhf`）从 `config.data_dir` 及环境变量指定目录中搜索候选文件，避免由于别名或软链接导致的重复加载。【F:src/training/pipeline/data_manager.py†L28-L99】
- `DatasetPreparer` 根据 `dataset_sampling` 配置对每个数据源执行采样、验证集切分与统计汇总，并最终构造 `ConversationDataset` 或 `LanguageModelingDataset` 以供训练循环使用。【F:src/training/pipeline/data_manager.py†L115-L214】
- 通过设置 `MINIGPT_GLOBAL_SAMPLE_RATIO`（默认 `0.5`）可统一缩放所有数据集的采样量，便于在教学或调试场景中快速减少预处理样本数。【F:config/training_config.py†L124-L182】【F:src/training/pipeline/data_manager.py†L140-L214】
- `ConversationDataset` 在构造时会自动插入角色标记、对非 assistant 位置填充 `pad_id` 掩码，并支持按概率截断尾部若干轮回复，以提升 SFT 泛化能力。【F:src/training/datasets/conversation.py†L12-L178】

> **排查建议**：当发现样本数量与期望不符时，优先检查 `max_length` 是否设置过小、`dataset_sampling.max_samples` 是否生效，以及原始 JSONL 是否存在空行或非法 JSON 格式。

### DatasetSampling 配置详解

`BaseConfig.dataset_sampling` 允许针对不同数据文件设置采样比例与验证集拆分，便于兼顾小众数据与主语料：

| 字段 | 作用 | 触发位置 |
| ---- | ---- | -------- |
| `sample_ratio` | 按原始样本数的比例进行随机采样，未指定时默认 1.0 | `DatasetPreparer.build` 会根据比例与 `max_samples` 共同决定采样量。【F:config/training_config.py†L124-L182】【F:src/training/pipeline/data_manager.py†L146-L214】 |
| `max_samples` | 对采样数量增加上限，常用于控制长尾数据集的权重 | 采样后会在日志中打印“原始→采样”对照，便于观察阈值是否生效。【F:src/training/pipeline/data_manager.py†L150-L200】 |
| `val_split` | 针对单一数据文件覆盖默认验证比例 | 若采样后不足 `validation_min_samples` 会自动关闭验证集，避免极小验证集导致指标不稳定。【F:config/training_config.py†L124-L206】【F:src/training/pipeline/data_manager.py†L170-L189】 |

所有采样统计会被写入输出目录的 `dataset_stats.json`，方便复现实验或排查数据配额变动。【F:src/training/pipeline/environment.py†L12-L63】

### 对话增强与掩码策略

- `ConversationDataset` 会为 `user/system` 轮次写入 `pad_id` 标签，使得损失仅关注 `assistant` 回复部分；若开启 `turn_separator`，不同轮次之间会插入额外 token 进一步强化轮次边界。【F:src/training/datasets/conversation.py†L38-L116】
- 通过 `conversation_augmentation.turn_truncate_prob/max_turn_truncate` 可按概率裁剪尾部若干轮，从而模拟不完整对话，提高模型在短上下文下的稳健性。【F:config/training_config.py†L147-L159】【F:src/training/datasets/conversation.py†L18-L120】
- 若原始 `conversations` 缺少 assistant 回复，数据集会自动补上一条空回复，避免因标签缺失导致训练循环报错。【F:src/training/datasets/conversation.py†L95-L111】

## 分词器管理

`TokenizerManager` 会对原始数据文件计算哈希并缓存训练好的分词器，避免重复训练；若指定 `--retrain-tokenizer` 则会忽略缓存重新生成。配置项（如 `vocab_size`、`min_frequency`、特殊 token）统一通过 `TokenizerConfig` 传入。【F:src/training/pipeline/tokenizer_manager.py†L14-L118】

典型用法：

```python
from src.tokenizer.tokenizer_manager import TokenizerManager, TokenizerConfig

manager = TokenizerManager()
tokenizer = manager.get_or_train_tokenizer(
    data_path="data/sft_mini.jsonl",
    config=TokenizerConfig(vocab_size=20_000, min_frequency=2)
)
```

分词器的 `bos_id/eos_id/pad_id` 会在训练配置中读取并传入模型，务必保持一致，以免训练时的标签掩码与推理生成出现偏差。【F:src/model/config.py†L20-L144】【F:src/training/pipeline/app.py†L137-L204】
