# MiniLLM 全流程小册子

> 本小册子详细记录了 MiniLLM 的完整训练流程：环境初始化、数据准备、RustBPE 分词、嵌入生成、预训练 (Pretrain)、监督微调 (SFT) 以及直接偏好优化 (DPO)。内容基于 `scripts/run.sh` 提供的一键脚本，并补充每一步的原理和手动操作指引。

> ⚠️ **RustBPE 为可选流程**：应社区反馈，脚本默认不会构建或调用 RustBPE；仅当设置 `ENABLE_RUSTBPE=1` 时才会启用相关步骤。以下章节仍保留 RustBPE 的详细说明，便于后续接入。

## 1. 使用 pip 初始化环境

1. **检测 Python**：脚本会读取 `PYTHON` 环境变量（缺省为 `python3`），并提示正在使用的解释器。
2. **创建虚拟环境**：默认在项目根目录创建 `.venv`，避免污染系统环境。
3. **安装依赖**：通过 `pip install -r requirements.txt` 安装依赖，并在 `.venv/.deps_installed` 写入标记。下次执行时若检测到标记则跳过安装，满足“环境准备好后直接训练”的诉求。
4. **保留 RustBPE 可选入口**：如需启用 RustBPE，可手动执行 `pip install maturin` 后运行 `ENABLE_RUSTBPE=1 bash scripts/run_cn_pipeline.sh`，新脚本保持默认关闭。

> 📌 *手动执行*
>
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
> ```

## 2. 数据准备与中国源数据配比

项目新增 `scripts/build_chinese_mix.py`，用于按照 nanochat 的数据配比构建中文任务混合：

| 数据类别 | 默认数量 | 数据描述 |
| --- | --- | --- |
| 通用对话 (`general`) | 460K | 例如 Belle, ShareGPT 中文清洗后的多轮对话 |
| 知识测评 (`knowledge`) | 100K | 例如 CLUE、CMMLU 选择题（转化为问答） |
| 数学推理 (`math`) | 8K | 例如 `math401` 中文版本、CMath |
| 身份认同 (`identity`) | 2 × 1K | 提升模型人设与自我描述能力 |
| 偏好对 (`preference`) | 60K | 例如 UltraFeedback 中文增强或自建偏好数据 |

脚本会自动读取 `data/chinese/*.jsonl` 中的源数据（如未提供会使用示例样本），并输出到 `data/processed/`：

- `pretrain_chinese.jsonl`：用于预训练的统一文本格式。
- `sft_chinese.jsonl`：用于 SFT 的 Chat 格式。
- `dpo_chinese.jsonl`：用于 DPO 的偏好对格式。

> 📌 *数据加载逻辑*
>
> - 采用惰性迭代器逐行读取 JSONL，避免一次性占用大量内存。
> - 每个类别会根据目标数量自动抽样或重复采样，确保比例稳定。
> - 身份认同数据默认重复两次，模仿 nanochat 在 mid-training 与 SFT 中的做法。
> - 结果写入时保留原字段，并附带 `source` 标签方便追溯。

## 3. RustBPE 分词与词表训练

`scripts/train_rustbpe_tokenizer.py` 复刻 nanochat 的 RustBPE 工作流：

1. **输入管线**：支持 `pretrain`、`sft`、`dpo` 三种数据格式。脚本会自动把 JSONL 样本转换为纯文本序列，传给 RustBPE。
2. **词表规模**：默认 6,400（扣除特殊符号后满足 RustBPE 至少 256 基础词元的约束）。
3. **模型导出**：生成 `tokenizer.pkl` 和 `tokenizer_meta.json`，包含分词模式、特殊符号与词表统计。

> 📌 *训练原理*
>
> - RustBPE 会先按照 GPT-4 的正则模式 (`SPLIT_PATTERN`) 切分文本，再并行执行 BPE 统计和合并。
> - 生成的 `mergeable_ranks` 会与特殊符号结合，交由 tiktoken 执行快速推理。
> - 训练完成的 `RustBPETokenizer` 支持 `encode`、`decode`、`render_conversation` 等接口，方便后续任务。

## 4. 将数据转换为嵌入 (Embedding) 表示

`scripts/export_embeddings.py` 提供统一的数据 -> token/嵌入流程：

1. **预训练 (`pretrain`)**：
   - 为每个文本添加 `<|bos|>` 起始符。
   - 输出 `(input_ids, labels)`，并写入 `Torch` 张量文件，默认 2048 序列长度。

2. **SFT (`sft`)**：
   - 使用 `RustBPETokenizer.render_conversation` 渲染对话，将 `<|user_start|>` / `<|assistant_start|>` 等标签插入。
   - 同时生成 `loss_mask`，仅对助手回复部分计算损失。

3. **DPO (`dpo`)**：
   - 对 `chosen` 与 `rejected` 两条对话分别渲染，保存为成对的 `input_ids`/`loss_mask`。

> 📌 *注意事项*
>
> - 所有张量均保存为半精度兼容的 `torch.int32` / `torch.int64`，便于直接加载到 DataLoader 中。
> - `--max-length` 控制截断与填充长度，默认 2048；脚本会在超出长度时发出警告。
> - 通过 `--bos-token`、`--append-eos` 等参数可自定义特殊符号处理。

## 5. 默认训练配置

- `hidden_size = 512`
- `num_hidden_layers = 8`
- `num_attention_heads = 8`
- `vocab_size = 6400`

该组合与仓库历史版本保持一致，`trainer/train_*.py` 默认采用这一配置。脚本会打印模型可训练参数量，方便检查。

## 6. 一键运行脚本

`scripts/run.sh` 串联上述步骤，完成以下工作：

1. 环境初始化（pip + venv）并兼容 `PYTHON`/`VENV_DIR` 自定义；
2. 构建中文数据混合 (`build_chinese_mix.py`)，同时确认 `data/chinese/identity_conversations.jsonl` 存在；
3. 依次运行预训练（2 epoch）、SFT、DPO，三阶段均使用默认配置；
4. 每个阶段结束后调用 `scripts/evaluate_stage.py` 进行快速验证，在终端打印分数并把 JSON 结果追加写入 `../tf_dir/eval_results.jsonl`；
5. 保持 `PRETRAIN_ARGS` / `SFT_ARGS` / `DPO_ARGS` 环境变量作为额外参数通道，方便自定义 batch size、学习率等训练细节。

脚本默认输出权重到 `out/`，并自动创建上一级目录中的 `tf_dir/` 以兼容历史流水线。如果希望重用 uv 方案，仍可运行 `bash scripts/run_cn_pipeline.sh`，并按照上一节说明手动安装 `maturin` 后开启 RustBPE。

> 💡 **本地 CPU 冒烟测试**：运行 `bash scripts/run.sh --smoke-test`，脚本会自动裁剪 JSONL 数据、改用 CPU + float32，并限制每阶段仅执行数个迭代，几分钟内即可验证整条流水线是否可用。

## 7. 身份认同数据

- 在 `dataset/identity_cn_sample.jsonl` 中提供示例对话，真实场景请替换为自定义身份设定生成的数据。
- 构建脚本会自动下载/引用 `identity_conversations.jsonl`（若存在），并在 mid-training/SFT 阶段重复采样两遍。
- 你可以按照 [nanochat 身份定制指南](https://github.com/karpathy/nanochat/discussions/139) 生成更多中文身份数据，然后放置在 `data/chinese/identity_conversations.jsonl`。

## 8. 手动执行提示

- **仅预训练阶段**：运行 `python trainer/train_pretrain.py --data_path data/processed/pretrain_chinese.jsonl`。
- **仅 SFT 阶段**：确认预训练权重存在后，执行 `python trainer/train_full_sft.py --data_path data/processed/sft_chinese.jsonl`。
- **仅 DPO 阶段**：`python trainer/train_dpo.py --data_path data/processed/dpo_chinese.jsonl`，脚本会自动加载最新的 SFT 权重。

## 9. 常见问题

1. **rustbpe 无法导入**：确认已设置 `ENABLE_RUSTBPE=1` 并执行 `uv tool install maturin`，随后运行 `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`，确保在同一虚拟环境下执行 Python。
2. **数据集太小导致抽样失败**：`build_chinese_mix.py` 会在数据不足时循环采样，仍建议提供足够语料。
3. **序列过长被截断**：在 `export_embeddings.py` 中调整 `--max-length` 或在构建数据集时进行预截断。
4. **GPU 显存不足**：降低 `--batch_size` 或增加 `--accumulation_steps`；脚本默认启用梯度累积。

---

通过本小册子，你可以快速了解并复现 MiniLLM 在中文语境下的完整训练与调优流程，也可以在每个阶段进行个性化定制。
