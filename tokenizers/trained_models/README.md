# 训练好的分词器模型
## Trained Tokenizer Models

此目录包含MiniGPT项目中训练好的分词器模型，按照ISTJ系统化管理原则进行组织。

### 🎯 分词器模型列表

| 模型名称 | 文件名 | 词汇表大小 | 训练配置 | 适用场景 | 文件大小 |
|---------|---------|-----------|----------|-----------|----------|
| **Tiny分词器** | `mac_tiny_tokenizer.pkl` | ~5,000 | tiny配置 | 快速测试、原型开发 | ~56KB |
| **Small分词器** | `mac_small_tokenizer.pkl` | ~10,000 | small配置 | 开发验证、小规模应用 | ~56KB |
| **Medium分词器** | `mac_medium_tokenizer.pkl` | ~20,000 | medium配置 | 生产部署、完整应用 | ~294KB |

### 📊 分词器特性对比

#### Tiny分词器 (mac_tiny_tokenizer.pkl)
- **训练时间**: 2024年7月
- **词汇表规模**: ~5,000 tokens
- **优点**:
  - 文件小巧，加载快速
  - 适合资源受限环境
  - 快速原型验证
- **适用场景**:
  - 概念验证
  - 快速测试
  - 教学演示

#### Small分词器 (mac_small_tokenizer.pkl)
- **训练时间**: 2024年7月
- **词汇表规模**: ~10,000 tokens
- **优点**:
  - 平衡的性能和大小
  - 适合中小规模应用
  - 良好的中英文支持
- **适用场景**:
  - 开发环境测试
  - 小规模生产应用
  - 性能基准测试

#### Medium分词器 (mac_medium_tokenizer.pkl)
- **训练时间**: 2024年8月
- **词汇表规模**: ~20,000 tokens
- **优点**:
  - 完整的语言覆盖
  - 最佳的分词质量
  - 支持复杂文本处理
- **适用场景**:
  - 生产环境部署
  - 高质量文本生成
  - 专业应用开发

### 🔧 使用方法

#### Python代码示例
```python
import pickle
from src.tokenizer.tokenizer_manager import TokenizerManager

# 加载分词器
with open('tokenizers/trained_models/mac_medium_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 使用分词器
text = "你好，这是一个测试文本。"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

print(f"原文: {text}")
print(f"分词结果: {tokens}")
print(f"重建文本: {decoded}")
```

#### 命令行使用
```bash
# 评估分词器性能
python3 scripts/evaluation/tokenizer/comprehensive_tokenizer_evaluation.py \
    --tokenizer tokenizers/trained_models/mac_medium_tokenizer.pkl

# 对比多个分词器
python3 scripts/evaluation/tokenizer/comprehensive_tokenizer_evaluation.py \
    --directory tokenizers/trained_models/ \
    --compare
```

### 📈 性能建议

#### 选择指南
1. **开发阶段**: 使用 `mac_small_tokenizer.pkl`
   - 加载速度快
   - 足够的词汇覆盖
   - 便于调试和测试

2. **生产部署**: 使用 `mac_medium_tokenizer.pkl`
   - 最佳的分词质量
   - 完整的语言支持
   - 适合高要求应用

3. **资源受限**: 使用 `mac_tiny_tokenizer.pkl`
   - 最小的内存占用
   - 快速的处理速度
   - 适合嵌入式设备

### 🔍 质量评估

所有分词器都通过了以下测试:
- ✅ 基础功能验证
- ✅ 中英文混合文本处理
- ✅ 特殊字符和符号处理
- ✅ 编码/解码一致性
- ✅ 性能基准测试

### 📝 更新记录

- **2024年8月**: 添加medium分词器，提升词汇覆盖
- **2024年7月**: 创建tiny和small分词器，建立基础版本
- **2024年**: 初始版本，基础BPE实现

### 🛠️ 技术细节

#### 分词器架构
- **算法**: Byte Pair Encoding (BPE)
- **编码格式**: UTF-8
- **特殊标记**: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- **兼容性**: Python 3.8+

#### 训练数据
- 中文语料: 包含新闻、文学、技术文档
- 英文语料: 包含通用文本、技术资料
- 混合语料: 中英文混合场景
- 特殊内容: 代码、标点、数字处理

### 📞 技术支持

如需技术支持或遇到问题:
1. 查看评估脚本输出的详细信息
2. 使用 `comprehensive_tokenizer_evaluation.py` 进行诊断
3. 参考项目文档中的分词器部分

---
*创建时间: 2024年*
*维护团队: alex-ckl.com AI研发团队*
*遵循标准: ISTJ系统化管理原则*