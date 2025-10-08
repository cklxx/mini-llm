#!/bin/bash
# UV环境设置脚本 - MiniGPT Mac优化版本

set -e

echo "🚀 设置MiniGPT训练环境 (UV版本)"
echo "=================================="

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ UV未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ UV版本: $(uv --version)"

# 创建虚拟环境
echo "📦 创建虚拟环境..."
if [ -d ".venv" ]; then
    echo "⚠️  虚拟环境已存在，正在重新创建..."
    rm -rf .venv
fi

uv venv --python 3.11

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source .venv/bin/activate

# 安装依赖
echo "📥 安装项目依赖..."
uv pip install -e .

# 安装开发依赖（可选）
echo "🛠️  安装开发依赖..."
uv pip install -e .[dev]

# Mac Apple Silicon特定配置
if [[ $(uname -m) == "arm64" ]]; then
    echo "🍎 检测到Apple Silicon，安装MPS支持..."
    uv pip install -e .[mps]
fi

# 验证安装
echo "🧪 验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import psutil; print(f'psutil版本: {psutil.__version__}')"

# 检查Apple MPS支持
if [[ $(uname -m) == "arm64" ]]; then
    python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available()}')" 2>/dev/null || echo "MPS检查失败"
fi

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "📋 使用方法:"
echo "1. 激活环境: source .venv/bin/activate"
echo "2. 快速开始: python quick_start.py"
echo "3. 直接训练: python scripts/train_optimized.py --config tiny"
echo ""
echo "🔍 环境信息:"
echo "Python路径: $(which python)"
echo "虚拟环境: $VIRTUAL_ENV"
echo ""
echo "🎯 下一步:"
echo "- 运行 'python quick_start.py' 开始训练"
echo "- 查看 'README_MAC_OPTIMIZED.md' 了解详细使用方法" 