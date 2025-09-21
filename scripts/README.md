# MiniGPT Scripts Directory

本目录包含了MiniGPT项目的所有脚本，按功能分类组织。

## 📁 目录结构

```
scripts/
├── README.md                    # 本文档
├── test_runner.py              # 主测试运行器
├── tests/                      # 测试脚本
│   ├── test_architecture.py    # 架构组件测试
│   └── test_training_inference.py # 训练推理测试
├── training/                   # 训练脚本
│   └── train_optimized.py     # 优化版训练脚本
├── inference/                  # 推理脚本
│   └── inference_optimized.py # 优化版推理脚本
├── data_processing/            # 数据处理脚本
│   └── prepare_datasets.py    # 数据集预处理脚本
└── evaluation/                 # 评估脚本
    └── evaluate_model.py       # 模型评估脚本
```

## 🚀 快速开始

### 1. 运行完整测试套件

```bash
# 运行所有架构和功能测试
python scripts/test_runner.py
```

### 2. 数据预处理

```bash
# 准备训练数据集
python scripts/data_processing/prepare_datasets.py \
    --input-dir data/dataset/minimind_dataset \
    --output-dir data/processed \
    --target-size 10000
```

### 3. 模型训练

```bash
# 使用优化架构训练模型
python scripts/training/train_optimized.py \
    --config small \
    --epochs 3 \
    --batch-size 8 \
    --data-paths data/processed/train.jsonl
```

### 4. 模型推理

```bash
# 交互式推理
python scripts/inference/inference_optimized.py \
    --model-path checkpoints/best_model.pt \
    --mode interactive

# 单次推理
python scripts/inference/inference_optimized.py \
    --model-path checkpoints/best_model.pt \
    --mode single \
    --prompt "你好，今天天气怎么样？"

# 工具调用测试
python scripts/inference/inference_optimized.py \
    --model-path checkpoints/best_model.pt \
    --mode tool \
    --prompt "帮我搜索人工智能的最新发展"
```

### 5. 模型评估

```bash
# 全面评估模型性能
python scripts/evaluation/evaluate_model.py \
    --model-path checkpoints/best_model.pt \
    --output-dir evaluation_results
```

## 📋 脚本详细说明

### 🧪 Tests (测试脚本)

#### `test_architecture.py`
- **功能**: 测试所有架构升级组件
- **包含**: RoPE、GQA、深度优化、权重共享测试
- **用途**: 验证架构改进是否正确实现

#### `test_training_inference.py`
- **功能**: 测试训练和推理流程
- **包含**: 数据加载、模型训练、生成测试
- **用途**: 确保端到端功能正常

### 🏋️ Training (训练脚本)

#### `train_optimized.py`
- **功能**: 优化版模型训练
- **特性**:
  - 支持所有架构升级（RoPE、GQA等）
  - 混合精度训练
  - 学习率调度
  - 检查点保存
  - 多种数据格式支持

**参数说明**:
- `--config`: 模型配置 (tiny/small/medium/large)
- `--data-paths`: 训练数据路径（支持多个）
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--max-length`: 最大序列长度
- `--use-fp16`: 启用混合精度

### 🔮 Inference (推理脚本)

#### `inference_optimized.py`
- **功能**: 高效模型推理
- **模式**:
  - `interactive`: 交互式对话
  - `single`: 单次推理
  - `tool`: 工具调用测试
  - `think`: Ultra Think深度分析
  - `benchmark`: 性能基准测试

**特性**:
- 支持工具调用检测
- Ultra Think深度推理
- 性能基准测试
- 多种采样策略

### 📊 Data Processing (数据处理脚本)

#### `prepare_datasets.py`
- **功能**: 数据集预处理和优化
- **特性**:
  - 多格式数据支持
  - 数据清洗和去重
  - 工具调用数据增强
  - 数据集分割
  - 统计分析报告

**处理流程**:
1. 加载多种数据源
2. 格式验证和清洗
3. 长度过滤和去重
4. 工具调用数据增强
5. 混合数据集创建
6. 训练/验证/测试分割

### 📈 Evaluation (评估脚本)

#### `evaluate_model.py`
- **功能**: 全面模型评估
- **评估维度**:
  - **困惑度**: 语言建模质量
  - **生成质量**: 多样性、连贯性
  - **工具调用**: 检测准确率
  - **Ultra Think**: 思维深度评估
  - **性能基准**: 速度、内存使用

**评估指标**:
- Perplexity (困惑度)
- Tokens per second (生成速度)
- Tool detection accuracy (工具检测准确率)
- Thinking depth score (思维深度评分)
- Memory usage (内存使用)

## 🎯 最佳实践工作流程

### 开发流程
```bash
1. 运行架构测试
   python scripts/test_runner.py

2. 准备数据
   python scripts/data_processing/prepare_datasets.py

3. 训练模型
   python scripts/training/train_optimized.py --config small

4. 评估模型
   python scripts/evaluation/evaluate_model.py --model-path checkpoints/best_model.pt

5. 测试推理
   python scripts/inference/inference_optimized.py --model-path checkpoints/best_model.pt
```

### 实验对比
```bash
# 对比不同配置
python scripts/training/train_optimized.py --config tiny
python scripts/training/train_optimized.py --config small
python scripts/training/train_optimized.py --config medium

# 评估对比
python scripts/evaluation/evaluate_model.py --model-path checkpoints/tiny_model.pt
python scripts/evaluation/evaluate_model.py --model-path checkpoints/small_model.pt
```

## 🔧 配置和自定义

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- 其他依赖见 `requirements.txt`

### 设备支持
- CUDA GPU (推荐)
- Apple Silicon MPS
- CPU (备选)

### 自定义扩展
所有脚本都支持参数配置，可以根据需求调整：
- 模型架构参数
- 训练超参数
- 数据处理选项
- 评估指标

## 📝 输出文件

### 训练输出
- `checkpoints/`: 模型检查点
- `logs/`: 训练日志
- `tensorboard/`: TensorBoard日志 (如果启用)

### 评估输出
- `evaluation_results/`: 评估报告
- `benchmark_results/`: 性能基准

### 数据处理输出
- `data/processed/`: 处理后的数据集
- `dataset_report.json`: 数据统计报告

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python scripts/training/train_optimized.py --batch-size 4

   # 启用梯度累积
   python scripts/training/train_optimized.py --accumulate-grad-steps 2
   ```

2. **数据格式错误**
   ```bash
   # 检查数据格式
   python scripts/data_processing/prepare_datasets.py --validate-only
   ```

3. **模型加载失败**
   ```bash
   # 检查模型文件
   python -c "import torch; print(torch.load('model.pt').keys())"
   ```

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=.
python -u scripts/training/train_optimized.py --verbose
```

## 📄 许可证

请参考项目根目录的 LICENSE 文件。

## 🤝 贡献

欢迎提交 Pull Request 或 Issue 来改进这些脚本！