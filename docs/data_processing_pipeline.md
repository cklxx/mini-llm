# 数据处理流程总结

## 概览

本文档详细说明了 MiniGPT 项目的数据处理流程，从原始数据到高质量训练数据的完整转换过程。

---

## 数据处理流程

### 1. 原始数据准备

**输入数据**:
- `data/sft/sft_mini_512.jsonl` - 原始 SFT 训练数据
  - 来源：预处理的对话数据
  - 格式：JSONL 格式，每行一个 JSON 对象
  - 规模：约 1,149,770 条对话记录

### 2. 初步清理（sft_mini_512.cleaned.jsonl）

**脚本**: `scripts/clean_sft_data.py` (如果存在)

**处理步骤**:
- 基本格式验证
- 初步数据清洗
- 移除明显的格式错误

**输出**:
- `data/final/sft_mini_512.cleaned.jsonl`
- 记录数：1,149,770 条

### 3. 数据质量分析

**脚本**: `analyze_cleaned_data.py`

**分析维度**:
- 用户消息长度统计
  - 平均：21.7 字符
  - 中位数：18.0 字符
  - 范围：0-461 字符

- 助手回复长度统计
  - 平均：137.7 字符
  - 中位数：108.0 字符
  - 范围：0-508 字符

- 对话轮次分布
  - 2 轮：31.94%
  - 4 轮：42.82%
  - 6 轮：14.50%
  - 其他：10.74%

**发现的问题**:
1. 重复数据：612 条 (0.05%)
2. 空内容：57 条
   - 空用户消息：11 条
   - 空助手回复：46 条
3. 极短内容：13,042 条 (1.13%)
   - 极短用户消息 (<5字符)：8,617 条
   - 极短助手回复 (<10字符)：4,425 条
4. 格式错误：0 条

### 4. 高质量数据生成

**脚本**: `scripts/create_high_quality_dataset.py`

**清理规则**:

1. **去重**
   - 使用 MD5 哈希检测完全重复的对话
   - 移除：3,673 条 (0.32%)

2. **移除空内容**
   - 过滤掉用户消息或助手回复为空的记录
   - 移除：59 条 (0.01%)

3. **过滤极短内容**
   - 用户消息 < 5 字符
   - 助手回复 < 10 字符
   - 移除：13,244 条 (1.15%)

4. **格式验证**
   - 确保 JSON 格式正确
   - 确保 conversations 字段存在且为列表
   - 移除：0 条

**输出**:
- `data/final/sft_high_quality.jsonl`
- **记录数**：1,132,794 条
- **保留率**：98.52%
- **移除率**：1.48% (16,976 条)

---

## 数据质量对比

| 指标 | 清理前 (cleaned) | 清理后 (high_quality) | 改进 |
|-----|-----------------|---------------------|------|
| 总记录数 | 1,149,770 | 1,132,794 | -1.48% |
| 重复记录 | 3,673 | 0 | ✓ 完全去重 |
| 空内容 | 57 | 0 | ✓ 完全移除 |
| 极短内容 | 13,042 | 0 | ✓ 完全移除 |
| 数据质量 | 中等 | 高 | ✓ 显著提升 |

---

## 训练配置更新

### run.sh 脚本修改

**修改位置**: `scripts/run.sh:253`

**修改内容**:
```bash
# 修改前
SFT_JSON=${SFT_JSON:-"$PRETRAIN_DEFAULT_ROOT/sft_mini_512.jsonl"}

# 修改后
# Use high-quality SFT dataset (cleaned, deduplicated, filtered)
SFT_JSON=${SFT_JSON:-"data/final/sft_high_quality.jsonl"}
```

**效果**:
- SFT 训练现在默认使用高质量数据集
- 可通过环境变量 `SFT_JSON` 覆盖此设置

---

## 使用方法

### 运行数据分析
```bash
python3 analyze_cleaned_data.py
```

### 生成高质量数据集
```bash
python3 scripts/create_high_quality_dataset.py
```

### 使用高质量数据训练

#### 本地环境
```bash
# 直接运行（自动使用高质量数据）
scripts/run.sh

# 或者显式指定
SFT_JSON=data/final/sft_high_quality.jsonl scripts/run.sh

# 烟雾测试
scripts/run.sh --smoke-test
```

#### 云端环境（OpenBayes）

**自动处理模式**（推荐）:
```bash
# 脚本会自动检测并处理 /openbayes/input/input0 中的数据
scripts/run.sh
```

云端运行时，脚本会自动执行以下操作：
1. 检测 `/openbayes/input/input0/sft_mini_512.jsonl` 或 `/openbayes/input/input0/final/sft_mini_512.cleaned.jsonl`
2. 自动生成高质量数据集到 `data/final/sft_high_quality.jsonl`
3. 使用高质量数据集进行训练

**手动指定数据路径**:
```bash
# 使用 input0 中的原始数据
SFT_JSON=/openbayes/input/input0/sft_mini_512.jsonl scripts/run.sh

# 使用自定义数据
SFT_JSON=/path/to/your/data.jsonl scripts/run.sh
```

### 使用自定义数据
```bash
# 指定其他 SFT 数据集
SFT_JSON=/path/to/your/data.jsonl scripts/run.sh
```

---

## 云端自动处理

### OpenBayes 环境集成

当在 OpenBayes 云端环境运行时，`scripts/run.sh` 会自动执行以下智能数据处理流程：

#### 1. 环境检测
脚本自动检测是否在云端环境（通过检查 `/openbayes/home` 目录）

#### 2. 数据源检测
按优先级顺序查找 SFT 数据：
1. `/openbayes/input/input0/final/sft_mini_512.cleaned.jsonl` - 已清理的数据
2. `/openbayes/input/input0/sft_mini_512.jsonl` - 原始数据

#### 3. 自动数据处理
如果找到数据源但高质量数据集不存在，自动执行：
- **去重**: 基于内容 MD5 哈希去重
- **过滤空内容**: 移除空的用户消息或助手回复
- **过滤短内容**: 移除极短的对话（用户 <5 字符，助手 <10 字符）
- **格式验证**: 确保 JSON 格式正确

#### 4. 输出位置
生成的高质量数据集保存到：
- `data/final/sft_high_quality.jsonl`

#### 5. 自动使用
训练自动使用生成的高质量数据集

### 云端数据处理示例

**场景 1: 首次运行，有原始数据**
```bash
# 云端 input0 结构
/openbayes/input/input0/
├── sft_mini_512.jsonl          # 原始数据（1.15M 条）
├── pretrain_hq.jsonl
└── dpo_pairs.jsonl

# 运行脚本
$ scripts/run.sh

# 输出
[cloud] Cloud environment detected, checking for input data...
[cloud] Found raw SFT data at /openbayes/input/input0/sft_mini_512.jsonl
[cloud] Auto-generating high-quality SFT dataset...
[cloud] Processing /openbayes/input/input0/sft_mini_512.jsonl -> data/final/sft_high_quality.jsonl
[cloud] Processed 100,000 records, kept 98,073
[cloud] Processed 200,000 records, kept 196,577
...
[cloud] Processing complete: kept 1,132,794, removed 16,976
[cloud] High-quality SFT dataset created successfully
[stage] Starting pretrain...
```

**场景 2: 高质量数据已存在**
```bash
# 高质量数据已生成
$ scripts/run.sh

# 输出
[cloud] Cloud environment detected, checking for input data...
[cloud] Found raw SFT data at /openbayes/input/input0/sft_mini_512.jsonl
[cloud] Using existing high-quality SFT dataset
[stage] Starting pretrain...
```

**场景 3: 使用已清理的数据**
```bash
# 云端 input0 结构
/openbayes/input/input0/
└── final/
    └── sft_mini_512.cleaned.jsonl  # 已清理的数据

# 运行脚本
$ scripts/run.sh

# 输出
[cloud] Cloud environment detected, checking for input data...
[cloud] Found cleaned SFT data at /openbayes/input/input0/final/sft_mini_512.cleaned.jsonl
[cloud] Auto-generating high-quality SFT dataset...
...
```

### 性能优化

云端自动处理针对大数据集进行了优化：
- **内存效率**: 流式处理，逐行读取和写入
- **进度显示**: 每 10 万条记录显示进度
- **错误处理**: 自动跳过格式错误的记录
- **失败回退**: 如果处理失败，自动回退到使用原始数据

### 手动控制

如果需要跳过自动处理，可以显式指定数据路径：

```bash
# 强制使用原始数据（跳过质量过滤）
SFT_JSON=/openbayes/input/input0/sft_mini_512.jsonl scripts/run.sh

# 使用自己预处理的数据
SFT_JSON=/openbayes/input/input0/my_custom_data.jsonl scripts/run.sh
```

---

## 数据文件结构

### 目录组织
```
data/
├── sft/
│   └── sft_mini_512.jsonl          # 原始数据
├── final/
│   ├── sft_mini_512.cleaned.jsonl  # 初步清理数据
│   └── sft_high_quality.jsonl      # 高质量数据（推荐）
└── processed/                      # 其他处理数据
```

### 数据格式
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "用户问题文本"
    },
    {
      "role": "assistant",
      "content": "助手回复文本"
    }
  ]
}
```

---

## 质量保证

### 数据清理标准

1. **最小长度要求**
   - 用户消息：≥ 5 字符
   - 助手回复：≥ 10 字符

2. **内容完整性**
   - 必须包含用户消息和助手回复
   - 内容不能为空或仅包含空白字符

3. **唯一性**
   - 基于对话内容的 MD5 哈希去重
   - 确保每条对话都是独特的

4. **格式正确性**
   - 有效的 JSON 格式
   - 包含必需的 conversations 字段
   - conversations 为合法的列表结构

### 预期训练效果改进

使用高质量数据集训练的优势：

1. **更高的训练效率**
   - 移除低质量样本减少训练噪声
   - 模型更快收敛

2. **更好的模型质量**
   - 避免学习极短或无意义的回复
   - 减少重复数据带来的过拟合

3. **更稳定的训练过程**
   - 所有样本都符合最小质量标准
   - 减少异常数据导致的训练不稳定

---

## 维护建议

### 定期检查

1. **新数据接入时**
   - 运行 `analyze_cleaned_data.py` 分析数据质量
   - 根据分析结果调整清理规则

2. **训练效果异常时**
   - 检查数据分布是否合理
   - 验证数据质量是否符合预期

### 扩展建议

1. **增加更多过滤规则**
   - 检测并移除有害内容
   - 过滤低质量或不相关的对话
   - 平衡数据分布

2. **数据增强**
   - 考虑同义改写增强数据多样性
   - 添加更多高质量对话样本

3. **持续优化**
   - 根据模型表现调整数据清理阈值
   - 定期更新和改进数据处理流程

---

## 相关脚本

- `analyze_cleaned_data.py` - 数据质量分析工具
- `scripts/create_high_quality_dataset.py` - 高质量数据生成脚本
- `scripts/run.sh` - 主训练脚本
- `scripts/build_chinese_mix.py` - 中文数据混合处理

---

## 总结

通过系统化的数据处理流程，我们将原始的 1,149,770 条对话数据清理为 1,132,794 条高质量训练数据，移除了 1.48% 的低质量样本，包括重复数据、空内容和极短对话。这个高质量数据集现在是默认的 SFT 训练数据源，预期将带来更好的训练效果和模型质量。
