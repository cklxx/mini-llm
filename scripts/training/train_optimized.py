#!/usr/bin/env python3
"""
优化版训练脚本
支持所有架构升级：RoPE、GQA、深度优化、工具调用等
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import time

from src.model.config import get_config, MiniGPTConfig
from src.model.transformer import MiniGPT


def setup_logging(log_dir: str = "logs"):
    """设置日志记录"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/train_optimized_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


class OptimizedDataset(Dataset):
    """优化的数据集，支持多种训练数据格式"""

    def __init__(self, data_paths: list, tokenizer=None, max_length: int = 512, mode: str = "sft"):
        self.data = []
        self.max_length = max_length
        self.mode = mode

        # 简单tokenizer（实际使用中应该使用BPE tokenizer）
        if tokenizer is None:
            self.tokenizer = self._build_simple_tokenizer(data_paths)
        else:
            self.tokenizer = tokenizer

        # 加载数据
        for data_path in data_paths:
            self._load_data(data_path)

        logging.info(f"Loaded {len(self.data)} samples in {mode} mode")

    def _build_simple_tokenizer(self, data_paths: list):
        """构建简单的字符级tokenizer"""
        chars = set()

        for data_path in data_paths:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                text = self._extract_text(data)
                                chars.update(text)
                            except:
                                continue

        # 添加特殊token
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>', '<|tool_call|>', '<|ultra_think|>'] + sorted(list(chars))

        return {
            'vocab': vocab,
            'char_to_id': {char: i for i, char in enumerate(vocab)},
            'id_to_char': {i: char for char, i in enumerate(vocab)}
        }

    def _extract_text(self, data: dict) -> str:
        """从数据中提取文本"""
        if 'conversations' in data:
            # 对话格式
            text = ""
            for conv in data['conversations']:
                role = conv.get('role', '')
                content = conv.get('content', '')

                # 检查是否包含工具调用
                if 'tool_calls' in conv:
                    text += f"{role}: {content} <|tool_call|>\n"
                    # 添加工具调用信息
                    for tool_call in conv['tool_calls']:
                        text += f"Tool: {tool_call.get('function', {}).get('name', '')}\n"
                else:
                    text += f"{role}: {content}\n"

            return text.strip()
        elif 'text' in data:
            # 纯文本格式
            return data['text']
        else:
            return str(data)

    def _load_data(self, data_path: str):
        """加载训练数据"""
        if not os.path.exists(data_path):
            logging.warning(f"Data file not found: {data_path}")
            return

        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = self._extract_text(data)
                        if text:
                            tokens = self._tokenize(text)
                            if len(tokens) > 1:
                                self.data.append(tokens)
                    except Exception as e:
                        logging.warning(f"Error processing line {line_num} in {data_path}: {e}")

    def _tokenize(self, text: str) -> list:
        """分词"""
        tokens = [self.tokenizer['char_to_id'].get('<bos>', 2)]

        for char in text[:self.max_length - 2]:
            tokens.append(self.tokenizer['char_to_id'].get(char, 1))  # UNK=1

        tokens.append(self.tokenizer['char_to_id'].get('<eos>', 3))
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]

        # 处理序列长度
        if len(tokens) <= self.max_length:
            padded = tokens + [0] * (self.max_length - len(tokens))
            input_ids = torch.tensor(padded[:-1], dtype=torch.long)
            labels = torch.tensor(padded[1:], dtype=torch.long)
        else:
            input_ids = torch.tensor(tokens[:self.max_length-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:self.max_length], dtype=torch.long)

        return input_ids, labels


def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01):
    """创建优化器"""
    # 分离需要weight decay的参数
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    return torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.95))


def create_scheduler(optimizer, num_training_steps: int, warmup_steps: int = None):
    """创建学习率调度器"""
    if warmup_steps is None:
        warmup_steps = min(1000, num_training_steps // 10)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        logits = model(input_ids)

        # 计算损失
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 记录进度
        if batch_idx % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                       f"Loss: {loss.item():.4f}, LR: {lr:.6f}")

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

    return avg_loss


def evaluate_model(model, dataloader, criterion, device, logger):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logger.info(f"Evaluation Loss: {avg_loss:.4f}")

    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': model.config.to_dict()
    }

    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='优化版MiniGPT训练脚本')
    parser.add_argument('--config', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='模型配置')
    parser.add_argument('--data-paths', nargs='+',
                       default=['data/dataset/minimind_dataset/sft_mini_512.jsonl',
                               'data/dataset/minimind_dataset/tool_calling_basic.jsonl',
                               'data/dataset/minimind_dataset/agent_ultra_think.jsonl'],
                       help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--max-length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--eval-steps', type=int, default=500, help='评估间隔')
    parser.add_argument('--use-fp16', action='store_true', help='使用混合精度训练')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()
    logger.info(f"Starting optimized training with config: {args.config}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 创建保存目录
    Path(args.save_dir).mkdir(exist_ok=True)

    # 加载配置和模型
    config = get_config(args.config)
    logger.info(f"Model config: {config.to_dict()}")

    model = MiniGPT(config)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # 准备数据
    logger.info("Loading training data...")
    train_dataset = OptimizedDataset(
        args.data_paths,
        max_length=args.max_length,
        mode="sft"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 创建优化器和调度器
    optimizer = create_optimizer(model, lr=args.lr)
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = create_scheduler(optimizer, num_training_steps)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 and device.type == 'cuda' else None

    logger.info("Starting training...")
    logger.info(f"Total training steps: {num_training_steps}")

    best_loss = float('inf')

    # 训练循环
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            criterion, device, epoch + 1, logger
        )

        # 保存检查点
        checkpoint_path = f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = f"{args.save_dir}/best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, best_path)
            logger.info(f"Best model saved: {best_path}")

    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")

    # 保存最终模型
    final_path = f"{args.save_dir}/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'vocab': train_dataset.tokenizer
    }, final_path)
    logger.info(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()