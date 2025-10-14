# 2023-2025年业界主流大模型与基础设施调研

> **说明**：本调研聚焦于2023-2025年期间公开披露或经多方渠道确认的业界代表性模型与训练基础设施。由于商业机密限制，部分信息来自公开演讲、学术论文、媒体采访或工程团队分享，可能存在一定误差，但尽量给出经过交叉验证的区间或量级估计。

## 目录
1. 2025 年新增重点进展速览
2. Transformer-Decoder 类全参数模型
3. 混合专家（Mixture-of-Experts, MoE）模型
4. 多模态与指令增强模型
5. 轻量化与蒸馏方向
6. 基础设施（Infra）建设要点

---

## 1. 2025 年新增重点进展速览

| 模型/项目 | 架构与规模亮点 | 训练硬件与规模 | Infra/工具链更新 |
| --- | --- | --- | --- |
| **OpenAI o4 系列**（o4、o4-mini、o4-high） | 面向推理强化的 Transformer-Decoder，主干保持全参设计并引入长期思维链缓存；主模型总参数推测在 1T+，活跃参数随路由调整。 | 继续依托万卡级 H100/B100 集群；推测使用自研 FPGA/ASIC 协处理器加速 RLHF 与评估。 | 发布新的 **Reasoning Runtime**、Trust & Safety 自动红队平台，强化安全守护与多模态实时对话。 |
| **Google Gemini 2.0 / 1.5 Pro Ultra** | Pathways 多骨干融合，实现 2M tokens 长上下文；多模态在统一骨干上共享参数。 | TPU v5p/v6e Pods，单次训练使用 2 万片以上 TPU；推理侧结合 GPU SuperPOD。 | 发布 **MaxText** 长序列优化、Spanner/AlloyDB 统一数据治理方案。 |
| **Meta Llama 3.1 & 3.2** | 扩展到 405B 参数，并提供 11B/90B 对齐版本；引入自适应压缩注意力与 1M tokens 预览级上下文。 | 405B 训练披露使用约 24,000 张 H100，历时 ~30 天；中型模型在 3,000 张级别。 | LlamaStack 升级支持 vLLM、SGLang、LoRA Serve，一键部署企业工作负载。 |
| **DeepSeek-V3 & DeepSeek-R1** | V3 采用分层 MoE + 长上下文缓存；R1 面向推理，强化 RLHFlow 自反馈。 | 声称使用 2,048 张 H800（国产化集群）+ 3,000 张 A100；R1 强调高效并行策略。 | 自研 **DeepEP** 推理框架、Sparse Attention 内核及国产化算力调度平台。 |
| **Databricks DBRX Instruct 2025** | MoE-Decoder，16 专家（4 激活），上下文扩展至 256k tokens；强化企业私域调优。 | 采用 20,000 张 H100（含部分 B100）混合集群；使用 MosaicML 自动化训练管线。 | 推出企业级 AI Gateway，支持私有数据治理、审计与算力监控。 |
| **Anthropic Claude 3.5 / Claude 3 Opus 2025** | 推理强化 Transformer，补充工具使用与代理框架；长上下文扩展至 2M tokens。 | Google Cloud TPU v5p Pods + H100，重视安全与合规。 | 扩展 Claude Workbench、Red Team Studio，自动化策略审查。 |

---

## 2. Transformer-Decoder 类全参数模型

### 2.1 OpenAI GPT-4 / o 系列
- **架构**：多层深度 Transformer-Decoder，内部包含稀疏 MoE 组件（官方未披露，但多方分析认为存在专家路由）。2025 年发布的 o4 系列强调推理强化，引入更深层专家路由与跨会话记忆缓存。
- **模型尺寸**：官方未公开；推测主模型参数量在万亿级别，存在多种规模的专家分支以匹配不同产品（o4、o4-mini、o4-high）。
- **训练 GPU**：早期基于 NVIDIA A100 80GB SXM4 集群，现已迁移至 H100/B100 万卡级训练，并尝试引入专用加速器协同优化 RLHF。
- **基础设施**：
  - 高速 InfiniBand 网络（400-800 Gbps）+ 自研光互联
  - 深度定制 Megatron-LM / DeepSpeed，并叠加 Reasoning Runtime
  - 大规模数据治理、RLHF/RAHF 管线与 Trust & Safety 自动化红队体系

### 2.2 Anthropic Claude 3（Opus/Sonnet/Haiku）
- **架构**：标准 Transformer Decoder，强调对齐与安全性；Opus 版本可能在 MoE 基础上优化。2025 年 Claude 3.5 在工具使用、推理与长上下文方面显著增强。
- **模型尺寸**：未公开；推测 Opus 在数千亿到万亿级，Sonnet/Haiku 为中小型衍生。
- **训练 GPU**：主要采用 NVIDIA H100/A100，并扩展到 TPU v5p；披露采购超过 20,000 张 H100，形成混合算力池。
- **基础设施**：
  - 与 Google Cloud 合作，使用 TPU v5e/v4/v5p Pods 以及自建 GPU 超算
  - 大规模 RLHF、人类反馈数据平台与 Claude Workbench
  - 强化的内容过滤、自动化红队与对齐流水线

### 2.3 Google Gemini 1.5 Pro / Ultra
- **架构**：Transformer-Decoder 结合多模态编码器，支持长上下文（>1M tokens）；2025 年 Gemini 2.0 加入多骨干协同与自适应路由。
- **模型尺寸**：Ultra 为万亿级参数；Pro 在数千亿级；Gemini 2.0 进一步扩大至万亿+总参数并提供 2M tokens 上下文。
- **训练 GPU/TPU**：主力在 TPU v5e/v5p/v4 MegaPods，并扩展到 v6e；兼容 GPU 推理。
- **基础设施**：
  - TPU 互联超算（超万片 TPU）
  - Pathways 框架 + MaxText，实现异构调度与长上下文优化
  - 多模态数据治理、自动化评估基准与安全策略库

### 2.4 Meta Llama 3 系列
- **架构**：标准 Transformer Decoder，使用 RMSNorm、SwiGLU、分组查询注意力（GQA），训练时应用多尺度学习率与延伸上下文技巧；2025 年 Llama 3.1/3.2 引入自适应压缩注意力与更长上下文。
- **模型尺寸**：8B、70B、400B（传闻）多个版本；公开权重最大为 70B；405B/90B 版本在内部服务中试运行。
- **训练 GPU**：Meta 披露 70B 版本使用约 16,000 张 NVIDIA H100 训练 30+ 天；8B 使用约 2,000 张；405B 推测使用 24,000 张 H100。
- **基础设施**：
  - 自研 LlamaStack：包括 LLaMA-cpp、LLaMA-Serve、Llama Deploy
  - 内部 FSDP + Tensor Parallel 混合并行训练，结合 MaxText/FlashAttention-3
  - 广泛的合成与过滤数据管线（含 15T+ tokens）与安全评估基准

### 2.5 Mistral Large & Mixtral 衍生
- **架构**：
  - Mistral 7B/Small：Transformer Decoder + Sliding Window Attention + GQA
  - Mistral Large：推测为多头注意力 + MoE 组合；Mixtral 8x22B 在 2025 年进入测试
- **模型尺寸**：7B、8x7B（Mixtral 8x7B）、8x22B、Large 约 70B+。
- **训练 GPU**：主要使用 NVIDIA A100/H100；据透露训练 7B 使用约 1,000-2,000 张 A100 数周，Mixtral 8x22B 使用 3,000+ 张 H100。
- **基础设施**：
  - 依赖自研优化的 FlashAttention、xFormers 内核
  - 使用 FairScale/FSDP 与 DeepSpeed ZeRO 优化，叠加自研路由负载均衡
  - 高速 NVLink 互联集群与法国内部液冷数据中心

### 2.6 百度文心4.0（ERNIE 4.0）
- **架构**：改进的 Transformer Decoder，结合跨模态模块与检索增强（RAG），2025 年在企业服务中增加 Agent 平台与插件生态。
- **模型尺寸**：官方称“千亿参数级”；具体未披露，公开信息显示存在 100B+ 主干与多模态分支。
- **训练 GPU**：基于昆仑芯片及 NVIDIA A/H 系列混合集群；重点部署在“飞桨+昆仑”平台，并建设国产化算力中心。
- **基础设施**：
  - 飞桨（PaddlePaddle）深度优化版本
  - 内部数据治理、知识图谱与对齐工具链
  - 面向政企的安全审计与本地化部署套件

---

## 3. 混合专家（MoE）模型

### 3.1 Google Gemini 2.0 MoE / Gemini 1.5 Ultra
- **架构**：在 Pathways 框架下的分层稀疏专家 Transformer，结合多模态骨干，Top-2 路由并引入专家容量自适应。2025 年 Gemini 2.0 MoE 将视觉、语音与文本统一到同一专家池中，实现跨模态共享专家。
- **模型尺寸**：Gemini 1.5 Ultra 稀疏总参数推测在 1T+，活跃参数约 80-120B；Gemini 2.0 MoE 进一步扩展专家数量并提升长上下文至 2M tokens。
- **训练硬件**：TPU v5p/v6e MegaPods，单次训练使用 20,000 片以上 TPU，并在推理侧引入 GPU SuperPOD 混合集群。
- **基础设施**：Pathways + MaxText 组合进行稀疏激活调度，配合自动负载均衡、专家健康监控与跨模态数据流水线。

### 3.2 DeepSeek-V2/V2.5/V3
- **架构**：DeepSeekMoE，分层稀疏专家（64 专家，活跃 8 专家），引入无路由器负载的均衡训练策略；2025 年 V3 在此基础上增加分层路由与 KV Cache 共享，并上线 R1 推理增强版本。
- **模型尺寸**：V2 236B 总参数，活跃 21B；V2.5 延续此设计并增强推理效率；V3 提出 671B 总参数、活跃 37B 的配置。
- **训练 GPU**：使用约 2,048 张 NVIDIA A100 80GB；通过 FP8 混合精度与流水线并行降低成本；V3 转向 2,048 张 H800 + 异构 A100 集群，并针对国产化算力进行推理优化。
- **基础设施**：
  - 基于 Megatron-LM + DeepSpeed 改造的流水线、张量并行
  - 训练期间采用自研容错与算力调度平台与 DeepEP 通信库
  - 大规模中英双语与代码数据过滤体系，并集成 RLHFlow 数据闭环

### 3.3 Databricks DBRX 2025 / Mosaic RouteFormer
- **架构**：Decoder-only + MoE，8 专家（2 活跃）；使用多查询注意力（MQA）和 FlashAttention-2；2025 年企业版扩展至 16 专家（4 激活），叠加 256k tokens 上下文；Mosaic RouteFormer Demo 则在 MoE 基础上加入稀疏注意力和分层路由以提升对话与代码表现。
- **模型尺寸**：132B 总参数，活跃参数约 36B；2025 企业版提升至 280B 总参数、活跃 68B；RouteFormer Demo 提供 64B 总参数、活跃 12B 的高效配置。
- **训练 GPU**：12,384 张 NVIDIA H100；训练 3 个月，消耗 80 万 GPU 小时；2025 年版本使用 20,000 张 H100/B100，GPU 小时超过 120 万；RouteFormer Demo 在 4,096 张 H100 上完成预训练。
- **基础设施**：
  - MosaicML 训练平台（现属 Databricks）
  - Streaming 数据管线（平均 12T tokens）与 Delta Lake 数据湖整合
  - Composer + FSDP + 自动混合精度 + 企业级 AI Gateway

### 3.4 xAI Grok-2 / Grok-3 规划
- **架构**：稀疏专家 + 长上下文路由框架，结合压缩注意力与动态 KV Cache；2025 年 Grok-2 推理版强调工具调用和代码代理能力。
- **模型尺寸**：Grok-2 推测总参数 480B，活跃 48B；内部路线图指向 Grok-3 采用 1T+ 稀疏参数与多代理协作。
- **训练 GPU**：基于 20,000 张 NVIDIA H100 与部分 B100 的混合集群；借助 xAI 自研的 GrokChain 编排在推理侧进行多代理协同。
- **基础设施**：Starlink 低延迟骨干 + 自研通信库，结合 Tesla Dojo 试验性算力；构建端到端数据采集、内容审核与自动评估平台。

---

## 4. 多模态与指令增强模型

### 4.1 OpenAI GPT-4o / Omni 系列
- **架构**：统一多模态 Transformer（音频/视觉/文本共享骨干），端到端训练；2025 年 Omni 进一步引入多模态工具调用与实时语音循环。
- **模型尺寸**：未公开，推测与 GPT-4 同级但权重共享导致参数量更低；实时模型可能采用分层专家或共享权重以降低延迟。
- **训练 GPU**：A100/H100/B100 集群，需高带宽用于实时音频处理，并依赖低延迟分布式训练。
- **基础设施**：
  - 低延迟推理栈（Triton、自研调度器、Reasoning Runtime）
  - 多模态数据采集、对齐与隐私过滤体系，强化实时合规监控

### 4.2 Google Gemini 1.5 Flash/Pro / Gemini 2.0 多模态
- **架构**：统一多模态 Transformer + 长上下文缓存；Flash 面向低延迟推理；Gemini 2.0 支持原生视频/音频流式理解。
- **模型尺寸**：Flash 为中等规模（数百亿级），Pro 为数千亿级；Gemini 2.0 将上下文扩展到 2M tokens，并提供轻量化 Flash-S。
- **训练硬件**：TPU v5p/v4 Pods、多模态预处理加速器，2025 年引入 TPU v6e。
- **基础设施**：
  - 训练阶段使用分布式数据存储（Spanner + Colossus）与多模态数据清洗流水线
  - 推理阶段结合 Vertex AI、Gemini API 提供弹性扩缩与合规审计

### 4.3 Meta LLaVA-NeXT, ImageBind, Audiocraft 路线
- **架构**：文本骨干接入 CLIP/Segment Anything 等视觉模型，通过适配器/投影层实现多模态融合；2025 年推出 LLaVA-Next-Omni，整合音频生成。
- **模型尺寸**：基于 Llama 3/2（7B/13B/70B）与外部视觉编码器，并衍生出 90B/405B 内部版本。
- **训练 GPU**：开源项目一般使用 256-512 张 A100/H100 进行多阶段微调；企业部署可借助 1,000 张级别集群进行多模态对齐。
- **基础设施**：
  - 混合精度 + LoRA/QLoRA 微调，结合多模态对比学习
  - 多模态数据自动标注、质量控制与隐私保护流水线

### 4.4 文心一言多模态、阿里通义千问VL
- **架构**：多模态编码器 + 文本 Decoder；使用检索增强和工具调用，并逐步引入端到端多模态骨干。
- **模型尺寸**：主干在千亿级；多模态适配器在数亿级；2025 年通义千问VL 2.5 提供 32B/110B 版本。
- **训练 GPU**：A100/H800 及自研 910B 芯片混合；推理部署在云端集群与政企私有化环境。
- **基础设施**：
  - 专用多模态数据中心（图像、视频、语音）
  - 多阶段蒸馏与知识蒸馏策略，结合安全合规审核

---

## 5. 轻量化与蒸馏方向

### 5.1 Phi-3 系列（Microsoft）
- **架构**：Transformer Decoder，结合高质量小语料与推理链（CoT）数据增强；2025 年推出 Phi-3.5 集成工具调用。
- **模型尺寸**：Phi-3-mini（3.8B）、Phi-3-small（7B）、Phi-3-medium（14B），以及 2025 年新增 Phi-3.5-medium（16B）。
- **训练 GPU**：NVIDIA A100/H100；针对 mini 级模型训练卡数约 256-512 张，中型版本约 1,000 张。
- **基础设施**：
  - 采用“教材式”数据合成策略与自动化推理数据生成
  - DeepSpeed ZeRO Stage-3 + FlashAttention-2 + ONNX Runtime 推理优化

### 5.2 Qwen2 / Qwen1.5 系列（阿里巴巴）
- **架构**：Transformer Decoder + GQA；针对语音/视觉扩展增加模态适配器，2025 年 Qwen2.5 引入自适应分层注意力与 Agent 工具链。
- **模型尺寸**：0.5B 到 72B 覆盖；Qwen1.5-72B 公开权重；Qwen2.5 新增 14B/32B 推理强化版本与 110B 企业版。
- **训练 GPU**：Qwen1.5-72B 使用 2,048 张 A100；Qwen2 系列部分使用 H800；Qwen2.5 在上海张江算力中心部署 3,000 张 H800。
- **基础设施**：
  - Colossal-AI、Megatron 结合的张量+流水线并行
  - 大规模中英文混合数据清洗、指令微调框架与企业数据治理

### 5.3 MiniCPM、Yi 系列
- **架构**：优化的 Transformer Decoder，主打轻量化部署；MiniCPM 采用蒸馏自研大模型，Yi 系列强化中文推理能力。
- **模型尺寸**：MiniCPM 2.4B/8B、MiniCPM-MoE 8x3B；Yi-34B/6B/9B，并新增 Yi-1.5 12B/34B 推理强化版。
- **训练 GPU**：MiniCPM 2.4B 使用数十至百张 A100；Yi-34B 使用约 512 张 A100；MiniCPM-MoE 训练约 600 张 H100。
- **基础设施**：
  - 侧重端侧部署，结合量化（INT4/INT8）、蒸馏与分层缓存
  - LoRA、QLoRA、大规模推理评估框架与端侧评测体系

---

## 6. 基础设施（Infra）建设要点

### 6.1 训练阶段
- **算力规划**：
  - 千亿级全参数模型：需要 5,000-20,000 张 A100/H100/B100 级 GPU，或等效的 TPU v5p/v6e 集群；2025 年出现 30,000 张规模的超大集群案例。
  - MoE 模型：活跃参数较少，可用 2,000-10,000 张 GPU 达到万亿级总参数；DeepSeek 等团队展示 2,048 张 H800 支撑 1T+ 参数训练。
- **网络互联**：
  - 必须具备 400-800Gbps InfiniBand 或 NVLink Switch；跨机柜需 Dragonfly+/Fat Tree 拓扑。
  - 延迟优化：RDMA、SHARP、NCCL 2.18+，以及定制通信库（DeepEP、XCCL 等）。
- **存储与数据**：
  - 训练数据量级 5-20T tokens；需求 PB 级对象存储 + NVMe 缓存集群；长上下文模型需高吞吐样本流。
  - 元数据管理需配合数据版本控制、质量评估；Lakehouse/Delta Lake 成为 2025 年主流治理方案。
- **软件栈**：
  - 主流选型：Megatron-LM、DeepSpeed、PyTorch FSDP、JAX + Pathways、Colossal-AI、MaxText、SGLang。
  - 自动化调度：Kubernetes + Slurm、Ray、Run:AI、NVIDIA Base Command。

### 6.2 推理与服务
- **模型压缩与量化**：INT8/INT4、FP8、KV Cache 量化、剪枝，推理侧引入自适应精度与推理蒸馏（Speculative Decoding）。
- **调度与弹性**：
  - 多租户场景需配合 Triton Inference Server、vLLM、SGLang、TensorRT-LLM 等。
  - 动态批处理、连续批处理（continuous batching）与推理级 KV Cache 交换提升吞吐。
- **安全与监控**：
  - 全链路日志、提示词审计、敏感输出过滤、自动化红队测试。
  - 实时指标：GPU 利用率、延迟、吞吐、内存碎片，并纳入算力/成本可视化看板。

### 6.3 成本与能效
- **算力资本支出（CapEx）**：
  - 2025 年主流 GPU 报价区间：H100 SXM 单价约 2.0-2.5 万美元，B100 约 2.8-3.2 万美元，H800 在 1.5-1.8 万美元；万卡集群一次性投入 2-6 亿美元，另需 15-25% 预算用于高速交换机、存储与机柜。
  - 若采用云上租赁，H100 on-demand 价格约 2.5-3.5 美元/GPU·小时，长期保留合约（1-3 年）可降至 1.5-2.2 美元/GPU·小时，但需提前锁定容量。
- **运营支出（OpEx）**：
  - 电力与散热：万卡集群耗电量可达 20-50MW，按 0.08-0.15 美元/kWh 计，全年电费约 1,400-6,000 万美元；需定制化液冷或浸没式冷却以控制 PUE（目标 1.1-1.2）。
  - 机房与网络：Tier-3 级别数据中心托管费用约 800-1,500 美元/机柜·月，万卡部署需 200-400 个机柜；跨地域专线与骨干网络每年额外支出 200-500 万美元。
  - 软件许可与支持：Run:AI、Slurm 企业版、Observability 平台等，每年预算约 200-500 万美元。
- **能效优化**：
  - 使用最新 H100/B100 或 TPU v5p/v6e，配合 FP8/混合精度降低功耗，并探索可再生能源供电。
  - 算法层面引入 MoE、蒸馏、缓存复用与自适应推理，降低推理能耗。

### 6.4 研发团队配置与人力成本
- **核心角色**：
  - 研究科学家 / 首席科学家（PhD/资深）：负责模型路线、训练策略；北美/中国一线城市总包 40-80 万美元/年，核心团队通常 3-8 人。
  - 大模型算法工程师：负责预训练、对齐与评估；年薪区间 20-40 万美元，旗舰项目需 15-30 人。
  - 数据工程与数据治理：建设数据管线、标注与质量体系；年薪 15-30 万美元，需 10-20 人，并配 200-500 人规模的外包标注队伍（约 3-6 美元/小时）。
  - 分布式系统 / HPC 工程师：维护集群、调度与性能优化；年薪 18-35 万美元，需 8-15 人。
  - MLOps / DevOps 与安全合规：负责 CI/CD、模型发布与红队；年薪 15-28 万美元，需 6-10 人。
- **综合预算**：一支支持千亿级模型研发与上线的核心团队（不含标注外包）年人力成本约 1,500-3,500 万美元；若加上数据标注、众包与顾问服务，整体人力相关支出可达 2,000-4,500 万美元/年。

---

## 7. 调研结论与建议
1. **架构选择**：
   - 若追求极致性能，仍以 Transformer Decoder（全参数或稀疏 MoE）为主流；MoE + 推理强化成为 2025 年热点。
   - 多模态需求强烈的团队应从统一 Transformer 骨干入手，减少模态间切换开销，并结合工具调用能力。
2. **模型尺寸规划**：
   - 中型企业可从 30B-90B 级模型切入，通过蒸馏/量化覆盖端侧与云端场景。
   - 超大规模模型（>400B）需结合 MoE 或分阶段蒸馏，否则训练成本与对齐难度巨大；推理强化模型可考虑多模型协同。
3. **硬件选型**：
   - 若能获取 H100/B100，应优先以其为主；否则 A100 仍是可行方案，但需更多节点与更长训练时间；国产 H800/910B 在部分场景具备成本优势。
   - TPU v5p/v6e 在 Google 生态下具备较高的性价比，但可获得性有限。
4. **基础设施建设**：
   - 数据治理与对齐流水线是成败关键；需要投入专门团队构建数据标注、评估、红队体系，并持续更新安全策略。
   - 推理服务应提前规划持续批处理、KV Cache 管理、多租户调度和成本监控，避免上线后大规模改造。

> 若需进一步针对特定垂直领域（如医疗、金融）开展深度调研，可在此基础上追加针对性数据与监管要求分析。
