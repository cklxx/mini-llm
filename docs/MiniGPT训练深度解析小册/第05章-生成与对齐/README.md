# 第 05 章 · 生成与对齐实践

完成基础训练后，mini-llm 提供了推理、对话封装与 RLHF 管道，方便将模型落地到真实场景。本章梳理这条后训练链路。

## 5.1 文本生成接口
- `TextGenerator` 封装在 `src/inference/generator.py` 中，可直接传入训练好的 `MiniGPT` 与 tokenizer。
- 支持的解码策略：`greedy_search`、`sample_generate`（含 top-k / top-p）、`beam_search`，都由 `GenerationConfig` 控制。
- `apply_repetition_penalty`、`top_k_filtering`、`top_p_filtering` 组合可有效缓解重复问题，适合长文本生成。

## 5.2 推理流程小贴士
- 在推理前调用 `model.eval()` 与 `torch.no_grad()`，`TextGenerator` 构造函数会自动完成这些步骤。
- 若想复用训练时的配置，可从 `MiniGPTConfig` 中读取 `max_generate_length`、`temperature` 等默认值，并传入 `GenerationConfig`。
- 对话系统可在 `sample_generate` 后将生成的 token 转为字符串，并在遇到 `tokenizer.eos_id` 时提前结束输出。

## 5.3 奖励模型与偏好数据
- 奖励模型模块位于 `src/rl/reward_model`，`create_reward_model` 和 `RewardTrainer` 负责加载 backbone、计算 pairwise ranking loss (`ranking_loss.py`)。
- `preference_data.py` 定义了偏好数据的整理流程，包含 JSONL 到 PyTorch Dataset 的转换。
- 训练奖励模型时，可通过 `RewardTrainer` 的配置控制冻结 backbone、选择优化器等细节。

## 5.4 PPO 策略优化
- `src/rl/ppo` 提供 `PPOTrainer`、`policy_gradient.py` 与 `value_model.py`，分别负责策略更新、损失拆分和价值模型构建。
- `PPOTrainer.step()` 会在每个 iteration 内部执行 rollout、优势估计、策略/价值更新，支持 mini-batch 迭代。
- 常见调参：`ppo_epochs` 控制每批样本重复优化次数，`ppo_lr_policy`/`ppo_lr_value` 分别作用于策略和价值网络。

## 深度拆解章节
- [5.1 RLHF 理论基石](01-RLHF理论与数学基础/README.md)
- [5.2 奖励建模与偏好学习](02-奖励建模与偏好学习/README.md)
- [5.3 PPO 微调语言模型](03-PPO算法语言模型微调/README.md)
- [5.4 生成策略与推理控制](04-生成策略与推理控制/README.md)

## 5.5 RLHF 全流程
- `RLHFPipeline` 将 SFT → 奖励模型 → PPO 串联，`RLHFConfig` 统一管理各阶段路径和超参。
- `run_sft()`、`run_reward_training()`、`run_ppo_training()` 分别封装了训练逻辑，最终由 `run_full_pipeline()` 按顺序执行并保存中间 checkpoint。
- 若只需某一阶段，可单独调用对应方法，或直接使用 `create_trainer`/`create_reward_trainer`/`create_ppo_trainer` 获得底层组件。

> 实践提示：RLHF 计算成本较高，建议先用 `TextGenerator.sample_generate` 检查 SFT 模型质量，再进入奖励模型与 PPO 阶段，避免资源浪费。
