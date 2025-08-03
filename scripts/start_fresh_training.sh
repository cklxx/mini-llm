#!/bin/bash
# 全新训练启动脚本示例

echo "🚀 开始全新的MiniGPT训练"
echo "================================"

# 确保在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 示例1: 使用全量数据训练medium模型，重新训练tokenizer，启用多线程优化
echo "📊 启动全量数据训练 (多线程优化版)..."
python scripts/train_optimized.py \
    --config medium \
    --use-full-data \
    --retrain-tokenizer \
    --tokenizer-vocab-size 20000 \
    --learning-rate 3e-5 \
    --max-steps 8000 \
    --batch-size 2 \
    --warmup-steps 800 \
    --output-dir "checkpoints/mac_medium_v2" \
    --save-steps 400 \
    --plot-loss \
    --max-cpu 85 \
    --max-memory 85 \
    --num-threads 6 \
    --dataloader-workers 2 \
    --enable-compile

echo "✅ 训练完成！"