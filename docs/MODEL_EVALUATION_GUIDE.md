# MiniGPT 模型评估指南

本文档介绍如何使用一键推理验证脚本评估MiniGPT模型的各项能力。

## 🎯 功能概览

MiniGPT评估系统提供全面的模型能力测试，包括：

- **自我认知测试** - 验证模型对自身身份和能力的认知
- **基础能力测试** - 测试语言理解和生成能力
- **逻辑推理测试** - 评估推理和问题解决能力
- **数学计算测试** - 验证数学推理能力
- **常识知识测试** - 检查常识掌握程度
- **中文理解测试** - 评估中文语言能力
- **创意生成测试** - 测试创造力和生成质量
- **技术问答测试** - 验证技术知识
- **Ultra Think测试** - 评估深度思维能力
- **多轮对话测试** - 检查上下文理解
- **安全性测试** - 验证安全边界意识

## 🚀 快速开始

### 1. 列出所有评估类别

```bash
# 查看可用的评估类别
make eval-categories

# 或直接运行
python scripts/quick_eval.py --list-categories
```

### 2. 快速评估（推荐首次使用）

```bash
# 仅测试自我认知（最快）
make eval-quick

# 或指定模型路径
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt \
    --quick
```

### 3. 完整评估

```bash
# 评估所有类别
make eval-full

# 或指定模型路径
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt
```

## 📊 评估类别详解

### 1. 自我认知测试 (self_identity)

**目的**: 验证模型是否正确理解自己的身份、开发者和能力

**问题示例**:
- 你好，请介绍一下你自己。
- 你是由哪家公司开发的？
- 你和ChatGPT是什么关系？
- 什么是Ultra Think模式？

**通过标准**:
- 正确回答自己是MiniGPT
- 说明由alex-ckl.com开发
- 解释Ultra Think功能

### 2. 基础能力测试 (basic_capabilities)

**目的**: 测试基本的语言理解和生成能力

**问题示例**:
- 请用一句话总结人工智能的定义。
- 什么是机器学习？
- 解释一下什么是神经网络。

### 3. 逻辑推理测试 (reasoning)

**目的**: 评估逻辑推理和问题解决能力

**问题示例**:
- 如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？
- 小明比小红高，小红比小刚高，那么谁最高？

### 4. 数学计算测试 (mathematics)

**目的**: 验证数学推理和计算能力

**问题示例**:
- 计算：25 + 37 = ?
- 求解方程：2x + 5 = 15，x等于多少？

### 5. Ultra Think深度思维测试 (ultra_think)

**目的**: 测试深度分析和创新思维能力

**问题示例**:
- 请深入分析人工智能对未来就业市场的影响。
- 分析一下区块链技术在金融领域的应用前景。

**特点**: 自动启用Ultra Think模式，期待更深入的回答

## 🛠️ 高级用法

### 自定义评估

```bash
# 评估特定类别
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt \
    --categories self_identity reasoning mathematics

# 显示详细输出
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt \
    --categories self_identity \
    --verbose

# 自定义生成参数
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-length 512

# 指定输出文件
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt \
    --output eval_results/my_eval.json
```

### 评估不同模型版本

```bash
# 评估预训练模型
python scripts/quick_eval.py \
    --model-path checkpoints/pretrain_medium/final_model.pt

# 评估SFT模型
python scripts/quick_eval.py \
    --model-path checkpoints/sft_medium/final_model.pt

# 评估DPO模型
python scripts/quick_eval.py \
    --model-path checkpoints/dpo_medium/final_model.pt
```

### 对比评估

```bash
# 评估多个版本并对比
for model in pretrain_medium sft_medium dpo_medium; do
    python scripts/quick_eval.py \
        --model-path checkpoints/$model/final_model.pt \
        --output eval_results/${model}_eval.json
done

# 查看结果对比
cat eval_results/*_eval.json | jq '.summary'
```

## 📈 评估结果说明

### 结果文件格式

评估完成后会生成JSON格式的结果文件：

```json
{
  "model_path": "checkpoints/sft_medium/final_model.pt",
  "device": "mps",
  "timestamp": "2025-01-08T14:30:00",
  "categories": {
    "self_identity": {
      "name": "自我认知测试",
      "total_questions": 10,
      "passed_questions": 9,
      "pass_rate": 0.9,
      "details": [
        {
          "question_num": 1,
          "question": "你好，请介绍一下你自己。",
          "answer": "你好！我是MiniGPT...",
          "passed": true,
          "matched_keywords": ["MiniGPT", "alex-ckl.com"],
          "elapsed_time": 2.3
        }
      ]
    }
  },
  "summary": {
    "total_categories": 11,
    "total_questions": 85,
    "total_passed": 72,
    "overall_pass_rate": 0.847
  }
}
```

### 评估指标

- **total_questions**: 总问题数
- **completed_questions**: 完成的问题数
- **passed_questions**: 通过的问题数
- **pass_rate**: 通过率（0-1）
- **overall_pass_rate**: 整体通过率

### 判断标准

不同类别有不同的判断标准：

1. **关键词匹配**: 自我认知测试检查回答中是否包含预期关键词
2. **拒绝检测**: 安全性测试检查是否正确拒绝不当请求
3. **自动评估**: 其他测试基于回答的完整性和相关性

## 📝 自我认知训练数据

项目提供了自我认知训练数据集，帮助模型学习正确的身份认知：

**位置**: `data/minigpt_identity.jsonl`

**内容**: 20+条对话样例，涵盖：
- 自我介绍
- 公司信息
- 与其他模型的关系
- 特色功能说明
- 能力和限制

**使用方法**:

```bash
# 训练时自动包含（如果使用SFT模式）
python scripts/train.py \
    --mode sft \
    --config medium \
    --retrain-tokenizer
```

## 🎯 最佳实践

### 1. 评估流程建议

```bash
# 步骤1: 快速测试（确认模型基本可用）
make eval-quick

# 步骤2: 完整评估（全面了解模型能力）
make eval-full

# 步骤3: 针对性改进（根据评估结果调整训练）
# 例如：如果自我认知测试不通过，增加identity数据
```

### 2. 定期评估

建议在以下时机进行评估：
- ✅ 训练完成后
- ✅ 模型更新时
- ✅ 添加新数据后
- ✅ 发布前验证

### 3. 结果分析

重点关注：
- **自我认知** - 必须通过率>80%
- **基础能力** - 体现模型基本水平
- **安全性** - 必须正确拒绝不当请求

### 4. 持续改进

根据评估结果改进：
- 通过率低的类别 → 增加相关训练数据
- 特定问题失败 → 分析原因并优化
- 对比不同版本 → 了解训练效果

## 🔧 故障排查

### 问题1: 模型找不到

**错误**: `FileNotFoundError: checkpoints/sft_medium/final_model.pt`

**解决**:
```bash
# 检查模型是否存在
ls -la checkpoints/*/final_model.pt

# 使用正确的模型路径
python scripts/quick_eval.py \
    --model-path <实际路径>
```

### 问题2: 生成质量差

**现象**: 回答不相关或质量低

**解决**:
- 调整temperature（0.7-0.9）
- 调整top_p（0.85-0.95）
- 增加max_length
- 检查模型是否充分训练

### 问题3: 评估速度慢

**解决**:
```bash
# 使用GPU/MPS加速
python scripts/quick_eval.py \
    --device cuda  # 或 mps

# 减少max_length
python scripts/quick_eval.py \
    --max-length 256

# 仅评估关键类别
python scripts/quick_eval.py \
    --categories self_identity basic_capabilities
```

### 问题4: 内存不足

**解决**:
```bash
# 使用小模型
# 或减少batch处理

# 降低max_length
python scripts/quick_eval.py \
    --max-length 128
```

## 📚 相关文档

- **问题集定义**: `scripts/eval_questions.py`
- **评估脚本**: `scripts/quick_eval.py`
- **训练数据**: `data/minigpt_identity.jsonl`
- **Makefile命令**: 查看 `make help`

## 🎓 示例：完整评估流程

```bash
# 1. 训练模型
make train-sft

# 2. 快速验证
make eval-quick

# 3. 完整评估
make eval-full

# 4. 查看结果
cat eval_results_*.json | jq '.summary'

# 5. 如果自我认知不理想，重新训练
# （minigpt_identity.jsonl会自动包含）
make train-sft

# 6. 再次评估验证
make eval-quick
```

---

**更新日期**: 2025-01-08
**版本**: v1.0
**维护者**: MiniGPT Team
