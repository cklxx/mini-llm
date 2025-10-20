# 🤝 RLHF 流程概览

`src/rl` 模块展示了一个端到端的 RLHF（Reinforcement Learning from Human Feedback）原型，包括监督微调（SFT）、奖励模型训练与 PPO 策略优化。本文档按阶段拆解配置项、执行流程以及需要关注的风险点。

## 配置 (`RLHFConfig`)

`RLHFConfig` 汇总了三个阶段的核心超参数，并允许通过 `create_rlhf_pipeline` 从字典或 JSON 文件构造实例。【F:src/rl/rlhf_pipeline.py†L37-L440】

| 分组 | 字段 | 说明 |
| ---- | ---- | ---- |
| 基础 | `model_name` | 用于记录当前实验使用的基座模型名称，配合日志输出 | 
|      | `tokenizer_path` | 序列化分词器路径，`load_tokenizer` 会直接反序列化并在后续阶段复用。【F:src/rl/rlhf_pipeline.py†L137-L144】 |
|      | `device='auto'` | 自动检测 CUDA/MPS/CPU 并记录日志，确保三阶段在同一设备上运行。【F:src/rl/rlhf_pipeline.py†L116-L129】 |
| SFT  | `sft_data_path`/`sft_epochs`/`sft_batch_size`/`sft_lr` | 控制监督微调的数据源、轮次与学习率；`run_sft` 会读取这些参数构造 DataLoader 与训练器。【F:src/rl/rlhf_pipeline.py†L168-L203】 |
| 奖励 | `reward_data_path`/`reward_epochs`/`reward_lr`/`reward_batch_size`/`freeze_reward_backbone` | 定义偏好数据与奖励头训练策略；`freeze_reward_backbone=True` 时会冻结语言模型主干，仅更新奖励头。【F:src/rl/rlhf_pipeline.py†L205-L253】 |
| PPO | `ppo_data_path`/`ppo_iterations`/`ppo_lr_policy`/`ppo_lr_value`/`ppo_batch_size`/`ppo_mini_batch_size`/`ppo_epochs` | 控制策略优化的迭代次数、学习率与 mini-batch 划分，确保采样与更新步数平衡。【F:src/rl/rlhf_pipeline.py†L255-L313】 |
| 通用 | `max_length`/`save_dir`/`save_interval`/`log_level` | 限制序列最大长度、保存阶段产物并配置日志等级；初始化时会将配置写入 `save_dir/config.json` 便于复现。【F:src/rl/rlhf_pipeline.py†L87-L136】 |

## 管道执行顺序

1. **初始化**：`RLHFPipeline.__init__` 会配置日志、自动选择设备、创建输出目录并保存完整配置快照，为后续阶段提供统一的运行上下文。【F:src/rl/rlhf_pipeline.py†L75-L136】
2. **加载资源**：`run_full_pipeline` 会依次加载分词器、基础模型，然后串行执行三个训练阶段；也可以按需调用 `run_sft`、`run_reward_training`、`run_ppo_training` 组合自定义流程。【F:src/rl/rlhf_pipeline.py†L315-L341】
3. **监督微调（SFT）**：
   - 使用 `create_trainer('sft', ...)` 组装 `SFTTrainer`，并用 `ConversationDataset` + `DataLoader` 读取对话样本。【F:src/rl/rlhf_pipeline.py†L168-L200】
   - `train()` 会根据配置轮次保存最佳模型到 `save_dir/sft/best_model.pt`，供奖励和 PPO 阶段复用。【F:src/rl/rlhf_pipeline.py†L193-L203】
4. **奖励模型训练**：
   - `create_reward_model` 在 SFT backbone 上挂载奖励头，并可根据 `freeze_reward_backbone` 决定是否更新主干参数。【F:src/rl/rlhf_pipeline.py†L217-L227】
   - 偏好数据通过 `create_preference_dataloader` 转换为成对样本，内部会调用排序损失与正则项以提升训练稳定性。【F:src/rl/rlhf_pipeline.py†L228-L253】【F:src/rl/reward_model/reward_trainer.py†L33-L200】
   - 训练完成后保存 `reward_model/best_model.pt`，并在日志中输出路径。
5. **PPO 策略优化**：
   - 载入前两阶段权重，构造策略模型、价值模型与 PPO 训练器，其中 `create_value_model` 复用 SFT 权重以保持初始化一致性。【F:src/rl/rlhf_pipeline.py†L255-L295】
   - `_load_ppo_prompts` 会读取 JSONL 中的 `prompt` 字段或裸字符串作为采样起点。【F:src/rl/rlhf_pipeline.py†L298-L362】
   - 训练循环会按 `ppo_iterations` 运行策略更新，并按照 `save_interval` 落盘中间结果，最后输出 `ppo/final_model.pt`。

### 输出目录结构

执行 `run_full_pipeline` 后，`save_dir` 会包含如下子目录，方便阶段化调试：

| 子目录 | 内容 | 来源 |
| ------ | ---- | ---- |
| `sft/` | 最佳 SFT 模型权重与配置快照 | `run_sft` 保存 `best_model.pt` 与指标日志。【F:src/rl/rlhf_pipeline.py†L168-L203】 |
| `reward_model/` | 奖励模型 checkpoint 及偏好训练日志 | `run_reward_training` 保存 `best_model.pt` 与损失曲线。【F:src/rl/rlhf_pipeline.py†L205-L253】 |
| `ppo/` | PPO 策略/价值模型及中间检查点 | `run_ppo_training` 在每次 `save_interval` 写入 `iteration_xxx.pt`，最终输出 `final_model.pt`。【F:src/rl/rlhf_pipeline.py†L255-L341】 |
| `logs/` | 统一日志文件和配置快照 | 初始化阶段调用 `_init_logging` 与 `_save_config` 生成。【F:src/rl/rlhf_pipeline.py†L75-L136】 |

建议在每个阶段完成后手动验证输出目录中的模型是否可加载，避免下一阶段因路径错误导致重复训练。

## 数据要求与校验

- **SFT 阶段**：输入需兼容 `ConversationDataset`，至少包含 user→assistant 一轮；如数据未预先插入角色标记，可交由数据集类自动补齐。【F:src/training/datasets/conversation.py†L12-L178】
- **奖励模型**：偏好样本必须提供 `(prompt, chosen, rejected)` 字段，`create_preference_dataloader` 会在内部构造 positive/negative pair 并生成注意力掩码，缺失字段会被跳过。【F:src/rl/reward_model/preference_data.py†L9-L144】
- **PPO 阶段**：提示语可以是 JSONL 中的对象或纯文本；若为对象，需包含 `prompt` 字段，否则视为普通字符串处理。【F:src/rl/rlhf_pipeline.py†L352-L362】

在三阶段之间共享分词器与模型权重，因此确保 `tokenizer_path` 对应的特殊 token ID 与训练配置一致，避免奖励模型与 PPO 阶段解码出错。【F:src/rl/rlhf_pipeline.py†L137-L165】

## 常见失败模式与应对

- **奖励模型过拟合**：若发现奖励损失迅速下降但 PPO 阶段无法收敛，可开启 `freeze_reward_backbone` 或降低 `reward_lr`，只训练头部参数。【F:src/rl/rlhf_pipeline.py†L205-L253】
- **PPO 发散**：当 KL 奖励项持续增大时，调小 `ppo_lr_policy` 或增加 `ppo_mini_batch_size`，使每次更新更稳定；必要时可减少 `ppo_iterations` 缩短训练周期。【F:src/rl/rlhf_pipeline.py†L255-L313】
- **提示无效**：`_load_ppo_prompts` 若读取到空行会直接跳过，导致有效样本过少；请先在数据准备阶段清理空行并确认 JSONL 中 `prompt` 字段存在。【F:src/rl/rlhf_pipeline.py†L298-L362】
- **显存占用过高**：奖励/策略/价值模型默认同时驻留在 GPU，可通过 `device='cpu'` 强制在 CPU 上训练奖励模型，再将结果迁回 GPU。【F:src/rl/rlhf_pipeline.py†L116-L253】

## 常见扩展方向

- **替换模型规模**：可在 `load_base_model` 中调用自定义的 `create_model` 或加载外部 checkpoint，只要返回的模型接口兼容 `create_trainer`、`create_reward_model` 即可。【F:src/rl/rlhf_pipeline.py†L145-L167】
- **定制奖励目标**：`RewardTrainer` 默认组合排序损失与正则项，可在 `create_preference_loss` 中加入 KL 惩罚或其他偏好学习目标，再在 `reward_trainer` 中注入新的损失权重。【F:src/rl/reward_model/ranking_loss.py†L11-L196】【F:src/rl/reward_model/reward_trainer.py†L152-L284】
- **扩展 PPO 策略**：`ppo` 子模块内的 `PPOTrainer` 使用策略梯度模板，可复制该结构实现 DPO、KTO 等算法，并在 `RLHFPipeline` 中新增入口选择不同优化器。【F:src/rl/ppo/ppo_trainer.py†L11-L274】
- **监控与评估**：`evaluate_model` 预留了困惑度、BLEU/ROUGE 指标位置，可在训练完成后调用自定义评估脚本形成闭环；同时建议在 `save_dir` 下保存训练日志与样例，便于审查。【F:src/rl/rlhf_pipeline.py†L364-L418】

由于当前实现主要聚焦流程串联，若在生产环境中部署，还需补充分布式训练、策略评估与安全过滤等模块，确保输出质量与合规性。
