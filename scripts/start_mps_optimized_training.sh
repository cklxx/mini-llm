#!/bin/bash

# MPS优化训练脚本 - 专门为Mac Apple Silicon设备优化
# 作者: Claude Assistant
# 创建时间: $(date)

echo "🚀 启动MPS优化训练..."
echo "📱 检测Apple Silicon/MPS设备优化配置"

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "🔧 激活虚拟环境..."
    source .venv/bin/activate
fi

# 运行MPS优化训练
python scripts/train_optimized.py \
    --config medium \
    --use-full-data \
    --retrain-tokenizer \
    --tokenizer-vocab-size 20000 \
    --tokenizer-samples 100000 \
    --learning-rate 3e-5 \
    --max-steps 8000 \
    --batch-size 2 \
    --warmup-steps 800 \
    --output-dir "checkpoints/mac_medium_mps" \
    --save-steps 400 \
    --plot-loss \
    --max-cpu 80 \
    --max-memory 80 \
    --num-threads 4 \
    --dataloader-workers 1 \
    --enable-compile \
    --auto-resume

echo "✅ MPS优化训练启动完成"