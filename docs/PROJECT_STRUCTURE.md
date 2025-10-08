# 📁 项目结构说明

经过清理和整理后的项目文件夹结构如下：

## 🗂️ 根目录文件

```
minigpt-training/
├── README.md                    # 项目主说明文档
├── README_MAC_OPTIMIZED.md      # Mac优化训练指南
├── pyproject.toml              # 项目配置文件
├── uv.lock                     # UV依赖锁定文件
├── .gitignore                  # Git忽略文件配置
├── .python-version             # Python版本配置
├── main.py                     # 项目入口文件
├── setup_uv.sh                 # UV环境设置脚本
├── quick_start.py              # 快速开始脚本
└── quick_start_uv.py           # UV版快速开始脚本
```

## 📚 核心目录

### `/src/` - 源代码目录
```
src/
├── __init__.py
├── data/                       # 数据处理模块
│   ├── __init__.py
│   └── dataset_loader.py
├── model/                      # 模型定义
│   ├── __init__.py
│   └── transformer.py
├── tokenizer/                  # 分词器
│   ├── __init__.py
│   └── bpe_tokenizer.py
├── training/                   # 训练模块
│   ├── __init__.py
│   └── trainer.py
├── inference/                  # 推理模块
│   ├── __init__.py
│   └── generator.py
├── rl/                        # 强化学习模块
│   ├── __init__.py
│   ├── rlhf_pipeline.py
│   ├── ppo/
│   └── reward_model/
└── utils/
    └── __init__.py
```

### `/scripts/` - 脚本目录
```
scripts/
├── train.py                   # 基础训练脚本
├── train_optimized.py         # Mac优化训练脚本 ⭐
└── generate.py                # 文本生成脚本
```

### `/tests/` - 测试目录 🆕
```
tests/
├── README.md                  # 测试说明文档
├── test_correct_small.py      # Small模型测试脚本 ⭐
└── inspect_model.py           # 模型检查工具 ⭐
```

### `/config/` - 配置目录
```
config/
├── training_config.py         # 训练配置
└── mac_optimized_config.py    # Mac优化配置 ⭐
```

### `/data/` - 数据目录
```
data/
└── dataset/
    └── minimind_dataset/       # 训练数据集
        ├── pretrain_200.jsonl  # 200条高质量数据 ⭐
        ├── pretrain_test.jsonl # 测试数据
        └── ...
```

### `/checkpoints/` - 模型检查点
```
checkpoints/
├── mac_tiny/                  # Tiny模型 (130万参数)
│   ├── final_model.pt         # 最终模型 ⭐
│   ├── tokenizer.pkl          # 分词器
│   └── checkpoint_*.pt        # 训练检查点
└── mac_small/                 # Small模型 (2400万参数)
    ├── final_model.pt         # 最终模型 ⭐
    ├── tokenizer.pkl          # 分词器
    └── checkpoint_*.pt        # 训练检查点
```

### `/docs/` - 文档目录 🆕
```
docs/
├── TRAINING_SUMMARY.md        # 训练总结
├── DEVELOPMENT_ROADMAP.md     # 开发路线图
├── CLAUDE.md                  # Claude使用说明
├── 实践项目与验证方案.md      # 实践指南
├── 后训练验证指标与工具.md    # 验证工具
└── 大模型后训练学习路径与验证指南.md
```

### `/logs/` - 日志目录
```
logs/
└── (训练日志文件)
```

## 🗑️ 已清理的文件

### 删除的测试脚本 (重复/过时)
- `test_tiny_model.py` → 功能已整合到tests目录
- `enhanced_test.py` → 功能已优化整合
- `test_small_model.py` → 失败的测试脚本
- `test_pretrain.py` → 功能重复
- `test_pretrain_large.py` → 功能重复
- `test_vocab_match.py` → 功能重复
- `test_progress.py` → 功能重复
- `debug_pretrain.py` → 调试脚本
- `debug_tokenizer.py` → 调试脚本
- `simple_test.py` → 功能重复

### 删除的系统文件
- `._*` (macOS资源分叉文件)
- `.DS_Store` (macOS文件夹设置)
- `__pycache__/` (Python缓存目录)
- `*.pyc` (Python字节码文件)

## 🚀 快速使用指南

### 1. 模型测试
```bash
# 测试Small模型
uv run python tests/test_correct_small.py

# 检查模型信息
uv run python tests/inspect_model.py
```

### 2. 模型训练
```bash
# Mac优化训练
uv run python scripts/train_optimized.py --config tiny

# 快速开始
uv run python quick_start_uv.py
```

### 3. 文本生成
```bash
# 使用训练好的模型生成文本
uv run python scripts/generate.py --model-path checkpoints/mac_small/final_model.pt --tokenizer-path checkpoints/mac_small/tokenizer.pkl --mode chat
```

## 💡 项目特点

- ✅ **清晰的模块化结构**: 代码按功能分类组织
- ✅ **完整的测试套件**: 专门的tests目录
- ✅ **详细的文档**: 分类整理的docs目录
- ✅ **Mac优化**: 专门的Mac平台优化配置
- ✅ **多模型支持**: 支持Tiny和Small两种模型规模
- ✅ **UV集成**: 现代Python包管理和环境管理

## 📝 注意事项

- 所有命令都建议使用 `uv run` 前缀
- 模型文件较大，已在.gitignore中排除
- 测试脚本都在tests目录中，有详细的README说明
- 文档都整理在docs目录中，方便查阅 