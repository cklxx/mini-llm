#!/bin/bash

# CUDA优化训练脚本 - 专门为英伟达GPU设备优化
# 作者: Claude Assistant
# 创建时间: $(date)

echo "🚀 启动CUDA优化训练..."
echo "🎮 检测英伟达GPU/CUDA设备优化配置"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "🔧 激活虚拟环境..."
    source .venv/bin/activate
fi

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ 无法检测CUDA，请确保已安装支持CUDA的PyTorch"
    exit 1
fi

# 运行CUDA优化训练
python scripts/train_optimized.py \
    --config medium \
    --use-full-data \
    --retrain-tokenizer \
    --tokenizer-vocab-size 25000 \
    --tokenizer-samples 150000 \
    --learning-rate 1e-4 \
    --max-steps 10000 \
    --batch-size 8 \
    --warmup-steps 1000 \
    --output-dir "checkpoints/mac_medium_cuda" \
    --save-steps 500 \
    --plot-loss \
    --max-cpu 90 \
    --max-memory 90 \
    --num-threads 8 \
    --dataloader-workers 8 \
    --enable-compile \
    --auto-resume

echo "✅ CUDA优化训练启动完成"