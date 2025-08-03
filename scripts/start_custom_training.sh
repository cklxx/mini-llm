#!/bin/bash
# 自定义训练启动脚本示例

echo "🎯 开始自定义MiniGPT训练"
echo "================================"

# 确保在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 示例2: 使用指定数据集的medium模型训练，启用多线程优化
echo "📊 启动自定义数据训练 (多线程优化版)..."
python scripts/train_optimized.py \
    --config medium \
    --data-files "pretrain_hq.jsonl" "sft_mini_512.jsonl" \
    --max-data-size 300000 \
    --retrain-tokenizer \
    --tokenizer-vocab-size 15000 \
    --tokenizer-samples 50000 \
    --learning-rate 5e-5 \
    --max-steps 6000 \
    --batch-size 4 \
    --warmup-steps 600 \
    --output-dir "checkpoints/mac_medium_custom" \
    --save-steps 300 \
    --plot-loss \
    --max-cpu 80 \
    --max-memory 80 \
    --num-threads 4 \
    --dataloader-workers 2 \
    --enable-compile

echo "✅ 训练完成！"