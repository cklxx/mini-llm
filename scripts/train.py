#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniGPT 训练脚本
支持完整的训练流水线：pretrain → sft → dpo → rlhf
"""
import os
import sys
import argparse
import time
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import create_model
from tokenizer.bpe_tokenizer import BPETokenizer
from training.trainer import create_trainer, LanguageModelingDataset, ConversationDataset
from config.training_config import get_config


class MiniGPTTrainer:
    """MiniGPT训练器，支持多种训练模式"""

    def __init__(self, config, mode="pretrain"):
        self.config = config
        self.mode = mode
        self.device = self._setup_device()
        self.output_dir = os.path.join(config.checkpoint_dir, f"{mode}_{config.model_size}")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"=== MiniGPT {mode.upper()} 训练 ===")
        print(f"模型配置: {config.model_size}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

    def _setup_device(self):
        """设置训练设备"""
        if torch.backends.mps.is_available():
            device = "mps"
            print("🔧 使用Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"🔧 使用CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("🔧 使用CPU")
        return device

    def setup_tokenizer(self, retrain=False):
        """设置分词器"""
        print("🔤 设置分词器...")

        tokenizer_path = os.path.join(self.output_dir, "tokenizer.pkl")

        if os.path.exists(tokenizer_path) and not retrain:
            print(f"加载现有分词器: {tokenizer_path}")
            tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
            tokenizer.load(tokenizer_path)
        else:
            print("训练新的分词器...")
            # 收集训练文本
            texts = self._collect_texts_for_tokenizer()

            tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
            tokenizer.train(texts)
            tokenizer.save(tokenizer_path)
            print(f"分词器已保存: {tokenizer_path}")

        print(f"词汇表大小: {tokenizer.vocab_size}")
        return tokenizer

    def _collect_texts_for_tokenizer(self):
        """收集用于训练分词器的文本"""
        texts = []
        data_paths = self._get_data_paths()

        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"⚠️  数据文件不存在，跳过: {data_path}")
                continue

            with open(data_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        text = self._extract_text_from_data(data)
                        if text:
                            texts.append(text)

                        # 限制分词器训练样本数量
                        if len(texts) >= 50000:
                            break
                    except json.JSONDecodeError:
                        continue

            if len(texts) >= 50000:
                break

        print(f"收集了 {len(texts)} 条文本用于训练分词器")
        return texts

    def _get_data_paths(self):
        """根据训练模式获取数据路径"""
        base_dir = self.config.data_dir

        if self.mode == "pretrain":
            return [
                os.path.join(base_dir, "pretrain_hq.jsonl"),
                os.path.join(base_dir, "sft_mini_512.jsonl")  # 补充数据
            ]
        elif self.mode == "sft":
            return [
                os.path.join(base_dir, "sft_mini_512.jsonl"),
                os.path.join(base_dir, "alex_identity.jsonl"),
                os.path.join(base_dir, "ultra_think.jsonl")
            ]
        elif self.mode == "dpo":
            return [
                os.path.join(base_dir, "dpo.jsonl")
            ]
        elif self.mode == "rlhf":
            return [
                os.path.join(base_dir, "alex_identity.jsonl"),
                os.path.join(base_dir, "ultra_think.jsonl")
            ]
        else:
            raise ValueError(f"不支持的训练模式: {self.mode}")

    def _extract_text_from_data(self, data):
        """从数据中提取文本"""
        if 'text' in data:
            return data['text']
        elif 'conversations' in data:
            # 对话格式
            text = ""
            for turn in data['conversations']:
                if 'content' in turn:
                    text += turn['content'] + " "
            return text.strip()
        elif 'input' in data and 'output' in data:
            return f"{data['input']} {data['output']}"
        elif 'chosen' in data and 'rejected' in data:
            # DPO格式
            return data['chosen']
        return None

    def setup_data_loader(self, tokenizer):
        """设置数据加载器"""
        print(f"📚 设置{self.mode}数据加载器...")

        # 加载数据
        all_data = []
        data_paths = self._get_data_paths()

        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"⚠️  数据文件不存在，跳过: {data_path}")
                continue

            file_data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        file_data.append(data)
                    except json.JSONDecodeError:
                        continue

            all_data.extend(file_data)
            print(f"从 {os.path.basename(data_path)} 加载了 {len(file_data)} 条数据")

        print(f"总共加载 {len(all_data)} 条{self.mode}训练数据")

        # 根据训练模式创建数据集
        if self.mode == "pretrain":
            dataset = self._create_pretrain_dataset(all_data, tokenizer)
        elif self.mode == "sft":
            dataset = self._create_sft_dataset(all_data, tokenizer)
        elif self.mode == "dpo":
            dataset = self._create_dpo_dataset(all_data, tokenizer)
        elif self.mode == "rlhf":
            dataset = self._create_rlhf_dataset(all_data, tokenizer)
        else:
            raise ValueError(f"不支持的训练模式: {self.mode}")

        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # 简化为单进程
            drop_last=True
        )

        print(f"数据批次数: {len(data_loader)}")
        return data_loader

    def _create_pretrain_dataset(self, data, tokenizer):
        """创建预训练数据集"""
        texts = []
        for item in data:
            text = self._extract_text_from_data(item)
            if text and len(text.strip()) > 10:
                texts.append(text)

        return LanguageModelingDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_len
        )

    def _create_sft_dataset(self, data, tokenizer):
        """创建SFT数据集"""
        conversations = []
        for item in data:
            if 'conversations' in item:
                conversations.append(item['conversations'])
            elif 'input' in item and 'output' in item:
                # 转换为对话格式
                conv = [
                    {"role": "user", "content": item['input']},
                    {"role": "assistant", "content": item['output']}
                ]
                conversations.append(conv)

        return ConversationDataset(
            conversations=conversations,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_len
        )

    def _create_dpo_dataset(self, data, tokenizer):
        """创建DPO数据集"""
        # 简化实现，将chosen作为正例训练
        texts = []
        for item in data:
            if 'chosen' in item:
                texts.append(item['chosen'])

        return LanguageModelingDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_len
        )

    def _create_rlhf_dataset(self, data, tokenizer):
        """创建RLHF数据集"""
        # 使用对话格式进行强化学习微调
        return self._create_sft_dataset(data, tokenizer)

    def setup_model(self, tokenizer, resume_from=None):
        """设置模型"""
        print("🧠 创建模型...")

        model = create_model(vocab_size=tokenizer.vocab_size, model_size=self.config.model_size)
        model = model.to(self.device)

        # 如果有预训练模型，加载权重
        if resume_from:
            print(f"🔄 从检查点加载模型: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        return model

    def train(self, resume_from=None, retrain_tokenizer=False):
        """执行训练"""
        print(f"🚀 开始{self.mode}训练...")
        start_time = time.time()

        # 设置分词器
        tokenizer = self.setup_tokenizer(retrain=retrain_tokenizer)

        # 设置数据加载器
        data_loader = self.setup_data_loader(tokenizer)

        # 设置模型
        model = self.setup_model(tokenizer, resume_from=resume_from)

        # 设置优化器和损失函数
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        # 训练循环
        model.train()
        step = 0
        best_loss = float('inf')

        print(f"开始训练，最大步数: {self.config.max_steps}")

        for epoch in range(1000):  # 最大epoch数
            epoch_loss = 0
            epoch_steps = 0

            for batch in data_loader:
                if step >= self.config.max_steps:
                    break

                # 数据移到设备
                batch = batch.to(self.device)

                # 验证batch尺寸
                if batch.size(1) < 2:
                    continue

                # 准备输入和目标
                input_ids = batch[:, :-1]
                target_ids = batch[:, 1:]

                # 前向传播
                optimizer.zero_grad()
                outputs = model(input_ids)

                # 计算损失
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 更新统计
                epoch_loss += loss.item()
                epoch_steps += 1
                step += 1

                # 记录日志
                if step % 100 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    elapsed = time.time() - start_time
                    print(f"Step {step:5d} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | "
                          f"Time: {elapsed/60:.1f}min")

                # 保存检查点
                if step % self.config.save_steps == 0:
                    self._save_checkpoint(model, tokenizer, optimizer, step, loss.item())

                # 评估模型
                if step % self.config.eval_steps == 0:
                    self._evaluate_model(model, tokenizer)

                if step >= self.config.max_steps:
                    break

            if step >= self.config.max_steps:
                break

        # 保存最终模型
        final_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_vocab_size': tokenizer.vocab_size,
            'config': self.config,
            'mode': self.mode,
            'step': step
        }, final_path)

        print(f"🎉 {self.mode}训练完成！")
        print(f"总步数: {step}")
        print(f"训练时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"最终模型已保存: {final_path}")

        return final_path

    def _save_checkpoint(self, model, tokenizer, optimizer, step, loss):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'mode': self.mode
        }, checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")

    def _evaluate_model(self, model, tokenizer):
        """简单评估模型"""
        model.eval()
        try:
            test_prompt = "你好，我是"
            input_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=self.device)

            with torch.no_grad():
                for _ in range(10):
                    outputs = model(input_tensor)
                    next_token_logits = outputs[0, -1, :]
                    next_token = torch.argmax(next_token_logits).item()
                    input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

            generated_text = tokenizer.decode(input_tensor[0].cpu().tolist())
            print(f"🧪 生成测试: '{generated_text}'")
        except Exception as e:
            print(f"生成测试失败: {e}")
        finally:
            model.train()


def main():
    parser = argparse.ArgumentParser(description='MiniGPT训练脚本')

    # 训练模式
    parser.add_argument('--mode', choices=['pretrain', 'sft', 'dpo', 'rlhf'], default='sft',
                        help='训练模式 (pretrain: 预训练, sft: 监督微调, dpo: 直接偏好优化, rlhf: 强化学习)')

    # 模型配置
    parser.add_argument('--config', choices=['medium', 'large'], default='medium',
                        help='模型配置大小 (medium: ~200M参数, large: ~500M参数)')

    # 数据相关
    parser.add_argument('--retrain-tokenizer', action='store_true',
                        help='重新训练分词器')

    # 继续训练
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点继续训练')

    # 训练参数覆盖
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='学习率')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大训练步数')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小')

    args = parser.parse_args()

    # 获取配置
    config = get_config(args.config)

    # 应用命令行参数覆盖
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # 根据训练模式调整配置
    if args.mode == "pretrain":
        config.max_steps = config.max_steps or 50000
        config.learning_rate = config.learning_rate or 1e-4
        print("📚 预训练模式：建立基础语言理解能力")
    elif args.mode == "sft":
        config.max_steps = config.max_steps or 10000
        config.learning_rate = config.learning_rate or 5e-5
        print("🎯 监督微调模式：训练对话和特定任务能力")
    elif args.mode == "dpo":
        config.max_steps = config.max_steps or 5000
        config.learning_rate = config.learning_rate or 1e-5
        print("⚖️  直接偏好优化模式：根据人类偏好调整响应")
    elif args.mode == "rlhf":
        config.max_steps = config.max_steps or 3000
        config.learning_rate = config.learning_rate or 1e-5
        print("🔄 强化学习微调模式：通过奖励模型优化")

    # 创建训练器
    trainer = MiniGPTTrainer(config, mode=args.mode)

    # 开始训练
    final_model_path = trainer.train(
        resume_from=args.resume,
        retrain_tokenizer=args.retrain_tokenizer
    )

    print(f"\n✅ 训练完成！模型保存在: {final_model_path}")

    # 提示下一步训练建议
    if args.mode == "pretrain":
        print("\n💡 建议下一步运行SFT训练:")
        print(f"uv run python scripts/train.py --mode sft --config {args.config} --resume {final_model_path}")
    elif args.mode == "sft":
        print("\n💡 建议下一步运行DPO训练:")
        print(f"uv run python scripts/train.py --mode dpo --config {args.config} --resume {final_model_path}")


if __name__ == "__main__":
    main()