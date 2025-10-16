# 小尺寸代码模型训练调研报告

## 1. 训练目标与整体流程概览
- **模型规模**：针对 1B 级别以内的自回归代码模型（如 350M/700M/1.3B 参数）。
- **训练阶段**：
  1. 原始代码预训练，覆盖多语言代码主体与注释、README。
  2. 指令/对话微调，用自然语言与代码混合提示增强问题解决能力。
  3. 可选的人类偏好对齐（DPO/RLHF）强化交互质量。
- **关键挑战**：数据质量与许可、长上下文效率、资源预算约束。

## 2. 语料来源与许可考量
| 语料来源 | 特点 | 许可 | 适用策略 |
| --- | --- | --- | --- |
| [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) | 缩减版 The Stack，覆盖 30+ 语言，清洗后约 40GB | Apache-2.0 | 作为主力多语言代码语料，可按语言子集下载。
| [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | Python/JavaScript 为主，已去除常见许可证冲突仓库 | MIT | 适合作为 Python 重点预训练补充。
| [code-search-net](https://huggingface.co/datasets/code_search_net) | 代码 + docstring 对齐，适合指令化微调 | Apache-2.0 | 提取函数/描述对用于监督微调。
| [openai/humaneval](https://huggingface.co/datasets/openai_humaneval) | 小规模评测基准 | MIT | 仅用于验证，不进入训练集。
| 企业/团队私有代码 | 高相关性，高质量 | 需内部许可 | 注意脱敏、合规与安全扫描。

- **许可守则**：
  - 预训练阶段遵循 source repo 许可证；对不兼容许可证（GPL-3.0 等）进行过滤。
  - 保留版权声明、LICENSE 与 NOTICE 元数据。
  - 在生成式场景中提供模型来源与数据说明，避免侵权风险。

## 3. 数据清洗与处理流程
1. **仓库级过滤**：基于 StarCoder 数据工作流，排除含个人信息或敏感词的文件；屏蔽 `test/`, `third_party/`, `vendor/` 等冗余目录。
2. **文件级规则**：
   - 使用 [starcoder/dedup](https://huggingface.co/spaces/bigcode/code-deduplication) 或 MinHash 去重。
   - 过滤大文件（>1 MB）与纯二进制内容。
   - 检查高重复率代码片段（>70% 重复行）。
3. **语言均衡**：借助 [Pygments](https://pygments.org/) 或 `fasttext` 自动识别语言，控制主语言（Python、C++、JavaScript、Java）占比不超过 70%。
4. **结构化存储**：转换为 JSONL 格式字段 `{ "repo": ..., "path": ..., "content": ... }`；附带 `license`, `token_count` 等元信息，便于后续采样与温度混合策略。

## 4. 模型架构与配置建议
| 项目 | 建议值 |
| --- | --- |
| 词表 | 32K BPE（基于 StarCoder 或 tiktoken cl100k）
| 上下文窗口 | 4K～8K token，小显存可采用滑动窗口填充与 FlashAttention-2
| Transformer 深度 | 24 层以内；宽度 2048；多头 16～24；FFN 扩展 4×
| 正则化 | RMSNorm + SwiGLU；使用 Dropout 0.1 + Attention Dropout 0.1
| 优化器 | AdamW (β1=0.9, β2=0.95, weight decay=0.1)
| 学习率策略 | 线性 warmup (2% 训练步数) + cosine decay
| 混合精度 | bfloat16 + GradScaler；ZeRO-2 / ZeRO-3 以节省显存
| 长上下文 | 采用 RoPE 缩放（YaRN / NTK-aware）以扩展上下文到 16K

## 5. 200M 级模型的可行性分析
- **定位与适用场景**：200M 左右的模型介于传统代码补全模型（<100M）与主流开源模型（>350M）之间，适合作为本地 IDE 智能补全、轻量代码审查机器人或离线推理的候选方案。
- **性能预期**：
  - 由于参数规模减半，语言理解与跨语言泛化能力会比 350M/700M 模型弱，但在特定语言（如 Python/TypeScript）上通过针对性数据增强仍可达到实用水平。
  - 需准备更高质量、更聚焦的训练语料（例如 15～25B token，以主语言为核心），并强化指令微调覆盖常用 IDE 场景。
  - 对评测指标的预期应聚焦在 HumanEval/MBPP 通过率的稳定提升（相较于 100M 级模型提升 5～10 个百分点），而非追求与 1B 模型同级表现。
- **资源与效率**：
  - 训练显存需求显著下降，单机 4×A100-40G 或 8×A6000 即可完成全量预训练，约 1～1.5 天可遍历 20B token。
  - 推理端在单张消费级 GPU（RTX 3090/4090）即可部署，甚至可通过量化（INT4/INT8）在 CPU 上提供基本补全。
- **风险与缓解**：
  - 模型容量有限，容易过拟合重复模式，需加强去重与正则化，并考虑知识蒸馏（从 350M/700M 教师模型蒸馏）以提升泛化。
  - 长上下文能力较弱，可通过窗口并行、检索增强或特化任务（如函数级补全）限定输入长度。
- **结论**：200M 级模型在资源受限或需要嵌入式部署的场景下可行，但若追求通用编程助手体验，建议起步 350M 以上；可先训练 200M 验证流水线，再扩展到更大规模。

## 6. 小尺寸模型渐进式实验设计
- **阶段 0：Tokenizer 与数据流水线验证（≤20M 参数）**
  - 结构：12 层解码器、隐藏维度 512、注意力头数 8、FFN 2048，词表 32K。
  - 数据：抽样 1～2B token（Python/TypeScript 为主），验证数据导入、去重、许可标记是否正常。
  - 目标：单机 1×A100-40G 半天内完成 1 epoch，确认损失曲线、推理端代码补全质量可用。
- **阶段 1：轻量可复现基线（60M～80M 参数）**
  - 结构：16 层、隐藏维度 768、注意力头数 12、FFN 3072；上下文窗口 2K，使用 FlashAttention。
  - 数据：4～6B token，覆盖主要语言与 README 注释；混入 20K 条指令/问答样本做短暂 SFT。
  - 目标：在 4×A6000 或 2×A100-40G 下 12 小时内完成 150K 步，获得稳定的 HumanEval/MBPP 改进。
- **阶段 2：200M 量级性能冲刺（180M～220M 参数）**
  - 结构：24 层、隐藏维度 1536、注意力头数 16、FFN 6144，窗口扩展至 4K，并开启 RoPE 缩放。
  - 数据：15～25B token，主语言 + 高质量项目；指令集扩充至 80K～120K 样本并加入执行反馈。
  - 目标：在 4×A100-40G/8×A6000 环境 1～1.5 天内遍历数据，评测指标相较阶段 1 再提升 5～10 百点。
- **阶段 3：向 350M+ 扩展的试点**
  - 基于阶段 2 的流水线，增加层数（32 层）与隐藏维度（2048），复用已验证的优化超参。
  - 在确认 200M 模型满足 IDE 插件或代码审查最小可用标准后，再投入额外资源扩展规模。
- **配套策略**：
  - 所有阶段均保留知识蒸馏与 LoRA 微调接口，以便快速复现或适配垂直场景。
  - 通过增量式检查点（每 20K 步快照）与轻量自动评测，及时终止表现欠佳的设置，降低算力浪费。

## 7. 训练计划与资源预估
- **数据规模**：预训练语料 50～80B token；指令微调 100K～300K 样本。
- **资源**：
  - 350M 模型：8×A100-40G，训练 ~2 天（batch size 0.5B token）。
  - 1.3B 模型：8×A100-80G，训练 ~6 天（batch size 1.5B token）。
  - 若硬件有限，可采用 `gradient_accumulation_steps` + `FlashAttention-2` + `sequence parallel`。
- **监控指标**：
  - 训练损失（code vs docstring）；
  - Token throughput / sec；
  - GPU memory 占用。
- **评测基准**：HumanEval、MBPP、CodeContests；额外加入 lint/单测通过率统计。

## 8. 指令微调与对齐策略
1. **监督微调（SFT）**：
   - 构造 `prompt -> solution` 数据，覆盖 bug 修复、算法题、代码解释。
   - 结合 Stack Overflow、LeetCode 解析、Docstring QA。
2. **偏好优化（DPO/RLHF）**：
   - 使用 Pairwise 代码审查数据（good vs bad fix）。
   - 奖励模型关注可运行性（单测通过）与风格规范。
3. **工具增强**：
   - 引入执行反馈（如 EvalPlus）收集自动标签。
   - 在训练阶段随机插入 `# TODO`、`## Explanation` 任务提高生成可读性。

## 9. 项目落地建议
- **配置模板**：在 `config/` 目录下新增针对 350M/700M/1.3B 的 YAML 配置，便于切换。
- **数据流水线脚本**：
  - `scripts/download_code_corpus.py`：自动化下载与导出语料（见本次新增脚本）。
  - `scripts/progressive_train.py`：按阶段执行渐进式训练计划，默认提供 dry-run 说明，可结合 `--execute` 真正跑预训练/SFT，并支持覆盖超参或衔接 checkpoint。
    - `--stages` 可筛选需要重复迭代的阶段，便于在单一环节内多次实验学习率、数据混合或蒸馏策略。
    - Dry-run 输出包含“迭代建议”，提醒每个阶段至少进行 2 次以上的主动复盘与重训，而非一次通过。
    - 建议在获得新评测结果或修正配置后再次运行脚本，以保持渐进式训练的多轮反馈闭环。
  - 后续可追加清洗脚本（语言识别、去重、许可证过滤）。
- **实验记录**：使用 Weights & Biases 或 TensorBoard 记录损失曲线与评测分数。
- **版本管理**：保持数据处理代码可重现，记录随机种子和数据快照。

## 10. 后续工作
- 构建专用的代码安全扫描器（检查秘钥/密码）。
- 针对领域代码（如嵌入式、金融）收集额外私有数据。
- 研究 `speculative decoding` 或 `Medusa` 以提升推理性能。

