# nanochat 调研纪要

## 核心特点概览
- **端到端单脚本体验**：`speedrun.sh` 在一次执行中串联虚拟环境初始化、数据抓取、分词训练、预训练、mid-train、SFT、可选 RL、评估与报告生成，保障“4 小时速通”故事完整落地。
- **算力与成本透明**：默认目标是 8×H100 的 $100 预算，但脚本内直接给出扩展到 d26/d41 等更大规模时需要的分片与 batch 调整，降低试错成本。
- **强制轻依赖**：仓库依赖 `uv`、RustBPE、纯 PyTorch 组件，强调可复制的最小可行环境，兼顾单机多卡与单卡梯度累积。

## 全流程管线拆解
1. **环境与日志**：speedrun 先检查 `uv`、创建 `.venv` 并激活，统一 pip 依赖来源；同一脚本开关 `WANDB_RUN` 控制是否写入 wandb。
2. **报告初始化**：调用 `python -m nanochat.report reset` 收集硬件/依赖信息写入 `report/header.md`，贯穿后续训练阶段的指标记录。
3. **分词阶段**：
   - 安装 Rust toolchain，构建 `rustbpe`。
   - 前台下载 8 片 FineWeb-Edu 预热 tokenizer，后台异步拉取 240 片供预训练。
   - 使用 `scripts.tok_train`（2B 字符）训练 65k 词表，再运行 `scripts.tok_eval` 统计压缩率、词频等指标，并写入报告。
4. **预训练 (BASE)**：
   - 下载 `eval_bundle`（CORE 评测数据），并等待后台分片全部完成。
   - `scripts.base_train` 依据 depth 派生 `n_embd/n_head/n_kv_head`，在 `torch.compile` 模型上使用 Muon + AdamW 组合优化、自动推算梯度累积步数，并周期性评估 CORE/val bpb。
   - 后续 `scripts.base_loss`、`scripts.base_eval` 汇总损失曲线、样本与 CORE 指标写入报告。
5. **Mid-train / SFT / RL**：
   - `scripts.mid_train` 通过多任务对话数据注入工具调用与多选任务能力。
   - `scripts.chat_sft` 针对单轮对话增强回答质量，`scripts.chat_rl`（可选）在 GSM8K 上运行 RLHF。
   - 每一步之后 `scripts.chat_eval` 会针对对应阶段评测（ChatCORE、GSM8K 等），补充到报告。
6. **交付**：默认提示使用 `scripts.chat_web` 提供 FastAPI + HTMX Web UI，`python -m nanochat.report generate` 生成最终 `report.md` 并复制到仓库根目录。

## 数据与分词设计
- **流式分片下载**：`nanochat/dataset.py` 允许指定分片数量与起始偏移，在速通脚本中前后台两阶段下载，兼顾冷启动速度与训练吞吐。
- **分布式流式 DataLoader**：`tokenizing_distributed_data_loader` 基于 `deque` 滚动缓存，将 `parquets_iter_batched` 产出的行批分批送入 tokenizer，动态拼接成 `(B, T)` 张量，并支持 DDP rank 间切分。
- **双栈 tokenizer**：`nanochat/tokenizer.py` 同时提供 HuggingFace BPE 与 RustBPE+tiktoken 实现，统一维护 `<|python_start|>`、`<|output_end|>` 等特殊 token，兼容 mid/SFT 工具流。
- **Tokenizer 训练评估工具链**：`scripts/tok_train` 能限定训练字符数、线程数，`scripts/tok_eval` 计算压缩比、频率直方图与极端样本，全部写入报告以便比较多次实验。

## 模型结构与优化策略
- **轻量高效 Transformer**：`nanochat/gpt.py` 选用 ReLU² MLP、无参数 RMSNorm、无 bias Linear、MQA、多头 KV cache，兼顾推理效率与实现简洁。
- **Rotary Embedding 管理**：模型在 `init_weights` 内预生成 10× sequence_len 的 RoPE 缓存，推理期间根据缓存加速位置编码，避免频繁重算。
- **混合优化**：embedding/unembedding 使用 `DistAdamW`，其余 Linear 层用 `Muon/DistMuon`，配合自定义动量调度 `get_muon_momentum` 与 warmdown 学习率曲线。
- **自动算力规划**：训练脚本根据 `total_batch_size`、`device_batch_size` 与 DDP 世界规模自动推导梯度累积；若传入 `target_flops` 或 data:param 比例，则反算迭代步数，帮助锁定预算。

## 推理引擎与工具体系
- **KVCache 抽象**：`nanochat/engine.py` 的 `KVCache` 支持前缀预填、动态扩容与批内复制，实现“一次前缀，多次采样”加速。
- **工具调用状态机**：`RowState` 追踪 `<|python_start|>`、`<|python_end|>`、`<|output_end|>`，遇到 Python 块时触发 `use_calculator` 执行受限表达式，支持多轮算术工具调用。
- **采样策略**：`sample_next_token` 支持温度=0 greedy、top-k，自定义随机数生成器保证多卡一致性。
- **服务接口**：`scripts.chat_web` 使用 FastAPI + HTMX + SSE 提供流式响应，`scripts.chat_cli` / `chat_eval` 则复用 `Engine` 完成离线推理或批量评测。

## 评估与可观测性
- **CORE 评测集成**：`scripts/base_eval` 解压 `eval_bundle` 后执行 ARC、GSM8K、HumanEval、ChatCORE 等任务，按阶段记录得分。
- **报告流水线**：`nanochat/report.py` 维护 `Report.log/Report.generate`，每个训练脚本通过 `get_report().log` 追加 Markdown 片段，最后汇总系统信息、训练超参、指标表格与样例输出。
- **监控钩子**：训练脚本在关键节点输出 tokens/step、FLOPs、EMA loss，并在报告中保留梯度裁剪、学习率、采样样例，为复现实验提供上下文。

## 工程化体验
- **配置覆盖器**：`nanochat/configurator.py` 允许从 CLI 或 `config.py` 覆盖 `base_train` 等脚本头部的超参声明，在不引入复杂配置系统的情况下保持透明。
- **任务与数据组织**：`tasks/` 目录分离 CORE、ChatCORE、SFT 列表，便于扩展新评测；`dev/` 提供 `repackage_data_reference.py` 等数据准备脚本。
- **测试与验证**：`tests/test_rustbpe.py` 等聚焦 tokenizer 正确性，配合 `uv` 的轻量测试命令保持 repo 干净。

## Mini-LLM 可借鉴要点
1. **提供“一键速通”脚本**：将 `make_dataset`、`train`、`eval`、`demo` 组合为 `scripts/run_all.sh` 或 Python Orchestrator，增加 wandb/本地日志开关，降低入门门槛。
2. **构建自动化报告体系**：仿照 `nanochat.report` 设计统一 `Report.log` 接口，让各阶段脚本将指标、样例、配置写入 Markdown，便于教学分享。
3. **引入数据分片配方**：在文档或配置内提供“参数量→所需 tokens → manifest 分片”换算表，提醒用户准备足够数据以满足目标 Chinchilla 比例。
4. **强化 tokenizer 评估**：借鉴 `scripts.tok_eval` 的压缩比、极端样例统计，构建 Mini-LLM 的 tokenizer QA 报告与回归测试。
5. **增加工具态采样接口**：参考 `RowState` 状态机，将 Mini-LLM 的 `TextGenerator` 扩展为支持工具起止标记、强制 token 队列，便于未来加入计算器或代码执行。
6. **提前规划 FLOPs/Token 预算**：在训练入口支持 `target_flops`、`data_param_ratio` 参数，让教学实验明确预算与算力消耗。
7. **混合优化器模板**：探索在 Mini-LLM 中集成 Muon 或类 Muon 优化器，并提供动量 schedule Hook，示范“不同参数块用不同优化策略”。
8. **交互式 Web Demo**：借用 FastAPI+HTMX 范式快速落地一个默认 Web UI，搭配 `uvicorn` 启动脚本，方便课程展示。
9. **轻量配置覆盖**：采用类似 `configurator.py` 的模式，在脚本顶层定义默认常量，再允许命令行覆盖，实现“少而清晰”的可配置性。
10. **数据下载后台化**：在数据准备阶段引入异步/后台下载（如 `subprocess.Popen`），提升长流程的资源利用率与总体耗时体验。
