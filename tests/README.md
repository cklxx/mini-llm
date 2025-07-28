# 测试脚本目录

本目录包含用于测试训练好的模型的脚本。

## 📁 文件说明

### `test_correct_small.py`
- **用途**: 测试Small模型(checkpoints/mac_small/)
- **功能**: 支持生成多个token的完整句子，包含交互式测试
- **配置**: 自动识别模型配置(6层，512维，5000词汇表)
- **使用**: `uv run python tests/test_correct_small.py`

### `inspect_model.py`
- **用途**: 检查模型checkpoint的结构和配置
- **功能**: 分析模型参数、推断配置、显示训练信息
- **支持**: 同时检查tiny和small模型
- **使用**: `uv run python tests/inspect_model.py`

## 🚀 快速开始

### 测试Small模型
```bash
# 使用UV环境
uv run python tests/test_correct_small.py

# 支持交互式生成测试
# 输入 'quit' 退出
```

### 检查模型结构
```bash
# 查看所有模型的详细信息
uv run python tests/inspect_model.py
```

## 📊 模型对比

| 模型 | 参数量 | 层数 | 维度 | 词汇表 | 路径 |
|------|--------|------|------|--------|------|
| Tiny | ~130万 | 4层 | 128 | 2000 | checkpoints/mac_tiny/ |
| Small | ~2400万 | 6层 | 512 | 5000 | checkpoints/mac_small/ |

## 💡 使用建议

1. **首次测试**: 建议先运行`inspect_model.py`了解模型配置
2. **生成测试**: 使用`test_correct_small.py`进行实际生成测试
3. **参数调整**: 可以修改temperature(0.1-2.0)控制生成的随机性
4. **长度控制**: 调整max_length参数控制生成文本长度

## 🔧 故障排除

- 确保使用UV环境: `uv run python ...`
- 检查模型文件是否存在
- 如果生成效果差，可能是训练不充分，建议增加训练步数 