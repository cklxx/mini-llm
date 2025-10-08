# 🤝 RLHF 流程概览

`src/rl` 模块展示了一个端到端的 RLHF（Reinforcement Learning from Human Feedback）原型，包括监督微调、奖励模型训练与 PPO 策略优化。

## 配置 (`RLHFConfig`)
关键字段：
- `tokenizer_path`：序列化分词器路径
- `sft_data_path` / `reward_data_path` / `ppo_data_path`：三个阶段对应的数据文件
- `sft_epochs`、`reward_epochs`、`ppo_iterations` 等训练轮次
- `ppo_lr_policy` / `ppo_lr_value` / `ppo_batch_size` 等 PPO 参数
- `device='auto'`：根据硬件自动选择 CUDA/MPS/CPU

## 管道结构 (`RLHFPipeline`)
1. **初始化**：配置日志、创建设备与输出目录，并保存配置文件
2. **SFT 阶段**：
   - 通过 `create_trainer('sft', ...)` 构建 `SFTTrainer`
   - 使用 `ConversationDataset` 构造 DataLoader
   - 训练完成后保存模型权重
3. **奖励模型阶段**：
   - `create_reward_model` / `create_reward_trainer` 构建奖励模型
   - `create_preference_dataloader` 读取偏好数据
   - 训练完成后生成奖励模型 checkpoint
4. **PPO 阶段**：
   - `create_value_model` 与 `create_ppo_trainer` 生成策略/价值网络
   - 在采样数据上执行多轮策略优化，按 `save_interval` 保存中间结果

## 数据格式
- **SFT**：与 `ConversationDataset` 要求一致（包含 user/assistant 对）
- **奖励模型**：偏好数据需要包含 `chosen` 与 `rejected`
- **PPO**：prompt 列表或其他采样数据，用于生成探索序列

## 自定义建议
- **替换模型规模**：在 `load_base_model` 中调整 `create_model` 调用或加载已有 checkpoint
- **集成自有数据**：重写 `_load_sft_data`、`_load_reward_data`、`_load_ppo_data` 以适配企业内部格式
- **监控日志**：`save_dir` 下会生成阶段性配置与模型，可结合 TensorBoard/Weights & Biases 记录训练曲线
- **扩展算法**：若需支持更复杂的 RL 算法，可在 `ppo` 子模块中新增实现，并在 `RLHFPipeline` 中调用

当前实现侧重展示流程结构，具体的评估指标与高性能分布式训练可在此基础上进一步扩展。
