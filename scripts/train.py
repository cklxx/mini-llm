#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniGPT 训练脚本
支持完整的训练流水线：pretrain → sft → dpo → rlhf
支持从checkpoint恢复训练
"""
import os
import sys
import argparse
import time
import json
import glob
import math
import signal
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
from training.training_monitor import TrainingMonitor
from config.training_config import get_config


class MiniGPTTrainer:
    """MiniGPT训练器，支持多种训练模式"""

    def __init__(self, config, mode="pretrain"):
        self.config = config
        self.mode = mode
        self.device = self._setup_device()
        self.output_dir = os.path.join(config.checkpoint_dir, f"{mode}_{config.model_size}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 优雅中断标志
        self.interrupted = False
        self.save_on_interrupt = True  # 中断时保存模型

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
        """设置分词器 - 支持跨阶段复用"""
        print("🔤 设置分词器...")

        # 当前模式的分词器路径
        tokenizer_path = os.path.join(self.output_dir, "tokenizer.pkl")
        
        # 如果是 SFT/DPO/RLHF，优先从 pretrain checkpoint 加载分词器
        pretrain_tokenizer_path = None
        if self.mode in ["sft", "dpo", "rlhf"]:
            # 尝试找到 pretrain 的分词器
            pretrain_dir = os.path.join(self.config.checkpoint_dir, f"pretrain_{self.config.model_size}")
            pretrain_tokenizer_path = os.path.join(pretrain_dir, "tokenizer.pkl")
            
            if os.path.exists(pretrain_tokenizer_path):
                print(f"✅ 从 pretrain checkpoint 加载分词器: {pretrain_tokenizer_path}")
                tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
                tokenizer.load(pretrain_tokenizer_path)
                
                # 复制到当前目录以便推理时使用
                import shutil
                shutil.copy2(pretrain_tokenizer_path, tokenizer_path)
                print(f"📋 分词器已复制到: {tokenizer_path}")
                print(f"词汇表大小: {tokenizer.vocab_size}")
                return tokenizer
            else:
                print(f"⚠️  未找到 pretrain 分词器: {pretrain_tokenizer_path}")
                print(f"   将训练新的分词器（建议先运行 pretrain 模式）")

        # 检查当前目录是否已有分词器
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

        # 创建数据加载器 - 多进程优化
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=getattr(self.config, 'prefetch_factor', 2),
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
        """创建SFT数据集 - 支持多种数据格式"""
        conversations = []
        for item in data:
            if 'conversations' in item:
                # 格式1: {'conversations': [{'role': 'user', 'content': ...}, ...]}
                conversations.append(item['conversations'])
            elif 'input' in item and 'output' in item:
                # 格式2: {'input': ..., 'output': ...} - 直接使用字典格式
                conversations.append({
                    'input': item['input'],
                    'output': item['output']
                })
            else:
                # 跳过无法识别的格式
                continue

        print(f"📊 SFT数据集包含 {len(conversations)} 个对话")
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

    def _signal_handler(self, signum, frame):
        """处理中断信号 (Ctrl+C)"""
        print(f"\n\n⚠️  收到中断信号 (Ctrl+C)")
        if not self.interrupted:
            self.interrupted = True
            print("🔄 正在优雅地停止训练...")
            print("💾 将保存当前模型状态...")
            print("   (再次按 Ctrl+C 可强制退出)")
        else:
            print("⚡ 强制退出！")
            sys.exit(1)

    def train(self, resume_from=None, auto_resume=False, retrain_tokenizer=False):
        """执行训练
        
        参数:
            resume_from: 指定checkpoint文件路径
            auto_resume: 自动从最新checkpoint恢复
            retrain_tokenizer: 是否重新训练分词器
        """
        print(f"🚀 开始{self.mode}训练...")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        print("💡 按 Ctrl+C 可优雅地停止训练并保存模型")
        
        start_time = time.time()

        # 设置分词器
        tokenizer = self.setup_tokenizer(retrain=retrain_tokenizer)

        # 设置数据加载器
        data_loader = self.setup_data_loader(tokenizer)

        # 设置模型
        model = self.setup_model(tokenizer, resume_from=None)  # 稍后会加载完整checkpoint

        # 设置优化器和损失函数
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps
        )

        # 学习率调度器: Warmup + Cosine Decay
        def get_lr_scheduler(optimizer, warmup_steps, max_steps, last_epoch=-1):
            """
            创建带Warmup的Cosine退火学习率调度器

            - 0 ~ warmup_steps: 线性增长从0到peak_lr
            - warmup_steps ~ max_steps: Cosine退火到min_lr
            
            Args:
                last_epoch: 上次训练的epoch/step数，用于恢复训练 (default: -1表示从头开始)
            """
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Warmup阶段: 线性增长
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Cosine退火阶段
                    progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # 最低降到10%

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

        # Note: scheduler will be created after checkpoint loading to use correct last_epoch
        scheduler = None
        warmup_ratio = self.config.warmup_steps / self.config.max_steps * 100
        print(f"✅ 学习率调度器: Warmup({self.config.warmup_steps}步, {warmup_ratio:.1f}%) + Cosine Decay")
        print(f"   初始LR: 0 -> 峰值LR: {self.config.learning_rate:.2e} -> 最低LR: {self.config.learning_rate * 0.1:.2e}")

        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        # 设置混合精度训练
        scaler = None
        if self.config.mixed_precision and self.device == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            print("✅ 启用混合精度训练 (FP16)")

        # 启用梯度检查点
        if self.config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✅ 启用梯度检查点")

        # 处理checkpoint恢复
        start_step = 0
        checkpoint_loaded = False
        
        if auto_resume:
            # 自动查找最新checkpoint
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                print(f"🔍 找到checkpoint: {latest_checkpoint}")
                start_step = self._load_checkpoint(latest_checkpoint, model, optimizer)
                checkpoint_loaded = True
            else:
                print("ℹ️  未找到当前模式的checkpoint")
        elif resume_from:
            # 从指定checkpoint恢复
            if os.path.exists(resume_from):
                start_step = self._load_checkpoint(resume_from, model, optimizer)
                checkpoint_loaded = True
            else:
                print(f"⚠️  Checkpoint文件不存在: {resume_from}")
        
        # 如果是 SFT/DPO/RLHF 模式且没有加载到checkpoint，尝试从 pretrain 加载初始权重
        if not checkpoint_loaded and self.mode in ["sft", "dpo", "rlhf"]:
            pretrain_dir = os.path.join(self.config.checkpoint_dir, f"pretrain_{self.config.model_size}")
            pretrain_model_path = os.path.join(pretrain_dir, "final_model.pt")
            
            if os.path.exists(pretrain_model_path):
                print(f"\n🎯 {self.mode.upper()} 模式：从 pretrain checkpoint 加载初始权重")
                print(f"   加载路径: {pretrain_model_path}")
                try:
                    checkpoint = torch.load(pretrain_model_path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"✅ 成功加载 pretrain 模型权重")
                    print(f"   💡 将在预训练基础上进行 {self.mode} 训练")
                    checkpoint_loaded = True
                except Exception as e:
                    print(f"⚠️  加载 pretrain 权重失败: {e}")
                    print(f"   将使用随机初始化的模型")
            else:
                print(f"\n⚠️  未找到 pretrain 模型: {pretrain_model_path}")
                print(f"   建议先运行 pretrain 模式训练基础模型：")
                print(f"   uv run python scripts/train.py --mode pretrain --config {self.config.model_size}")
                print(f"   现在将使用随机初始化的模型进行 {self.mode} 训练")
        
        if not checkpoint_loaded and self.mode == "pretrain":
            print("\n📚 Pretrain 模式：从随机初始化开始训练")
        
        # 创建学习率调度器，使用正确的last_epoch参数
        # 当恢复checkpoint时，last_epoch应该是start_step - 1
        scheduler = get_lr_scheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            last_epoch=start_step - 1 if start_step > 0 else -1
        )
        
        # 显示当前学习率状态
        if start_step > 0:
            print(f"📊 学习率调度器已恢复到第 {start_step} 步")
            current_lr = optimizer.param_groups[0]['lr']
            if start_step >= self.config.warmup_steps:
                phase = "Cosine Decay"
                progress = (start_step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps) * 100
                print(f"   当前阶段: {phase} (已完成{progress:.1f}%)")
            else:
                phase = "Warmup"
                progress = start_step / self.config.warmup_steps * 100
                print(f"   当前阶段: {phase} (已完成{progress:.1f}%)")
            print(f"   当前学习率: {current_lr:.2e}")

        # 初始化训练监控器（轻量级模式）
        # TensorBoard日志统一存储在 runs/{mode}_{size}_{timestamp}/
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = os.path.join(
            self.config.tensorboard_dir,
            f"{self.mode}_{self.config.model_size}_{timestamp}"
        )

        monitor = TrainingMonitor(
            model=model,
            log_dir=tensorboard_dir,
            enable_tensorboard=self.config.enable_tensorboard,
            enable_real_time_plots=False,  # 禁用实时绘图以节省性能
            lightweight_mode=True,         # 启用轻量级模式
            log_interval=10                # 每10步记录一次完整指标
        )

        # 清理GPU缓存
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"🧹 GPU缓存已清理")

        # 训练循环
        model.train()
        step = start_step  # 从checkpoint的步数继续
        best_loss = float('inf')
        accumulation_steps = self.config.gradient_accumulation_steps

        print(f"开始训练，最大步数: {self.config.max_steps}")
        print(f"Batch size: {self.config.batch_size}, 梯度累积: {accumulation_steps}, 有效batch: {self.config.batch_size * accumulation_steps}")

        # 显示内存使用情况
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 初始GPU内存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")

        for epoch in range(1000):  # 最大epoch数
            epoch_loss = 0
            epoch_steps = 0
            optimizer.zero_grad()  # 在epoch开始时清空梯度

            for batch_idx, batch in enumerate(data_loader):
                # 检查中断标志
                if self.interrupted:
                    print(f"\n⚠️  训练被用户中断（步骤 {step}）")
                    break
                
                if step >= self.config.max_steps:
                    break

                try:
                    # 数据移到设备 - 处理字典和tensor两种格式
                    if isinstance(batch, dict):
                        # 字典格式 (ConversationDataset)
                        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                        if 'labels' in batch:
                            target_ids = batch['labels'].to(self.device, non_blocking=True)
                        else:
                            # 如果没有labels，从input_ids生成
                            target_ids = torch.cat([
                                input_ids[:, 1:],
                                torch.full((input_ids.size(0), 1), tokenizer.pad_id, 
                                         dtype=torch.long, device=self.device)
                            ], dim=1)
                    else:
                        # Tensor格式 (LanguageModelingDataset)
                        batch = batch.to(self.device, non_blocking=True)
                        
                        # 验证batch尺寸
                        if batch.size(1) < 2:
                            continue
                        
                        # 准备输入和目标
                        input_ids = batch[:, :-1]
                        target_ids = batch[:, 1:]

                    # 混合精度前向传播
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(input_ids)
                            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
                            # 梯度累积：损失除以累积步数
                            loss = loss / accumulation_steps
                    else:
                        outputs = model(input_ids)
                        loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
                        loss = loss / accumulation_steps

                    # 反向传播
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"\n❌ CUDA OOM错误在步骤 {step}!")
                    # 获取batch信息 - 处理字典和tensor格式
                    if isinstance(batch, dict):
                        batch_size = batch['input_ids'].size(0)
                        seq_length = batch['input_ids'].size(1)
                    else:
                        batch_size = batch.size(0)
                        seq_length = batch.size(1)
                    print(f"   当前批次大小: {batch_size}")
                    print(f"   序列长度: {seq_length}")
                    if self.device == "cuda":
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"   GPU内存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")

                    # 清理显存
                    optimizer.zero_grad()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                    print(f"\n💡 建议解决方案:")
                    print(f"   1. 降低batch_size: --batch-size {self.config.batch_size // 2}")
                    print(f"   2. 增加梯度累积: 当前={accumulation_steps}, 建议={accumulation_steps * 2}")
                    print(f"   3. 减小序列长度: 当前max_seq_len={self.config.max_seq_len}")
                    print(f"   4. 启用梯度检查点 (gradient checkpointing)")
                    print(f"   5. 设置环境变量: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

                    raise  # 重新抛出异常以终止训练

                # 梯度累积：只在累积步数达到时更新参数
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    # 更新学习率调度器
                    scheduler.step()

                    optimizer.zero_grad()
                    step += 1

                    # 更新统计（使用实际损失值）
                    actual_loss = loss.item() * accumulation_steps
                    epoch_loss += actual_loss
                    epoch_steps += 1

                    # 使用监控器记录指标
                    # 获取batch size - 处理字典和tensor格式
                    if isinstance(batch, dict):
                        current_batch_size = batch['input_ids'].size(0)
                    else:
                        current_batch_size = batch.size(0)
                    
                    monitor.log_step(
                        step=step,
                        epoch=epoch,
                        loss=actual_loss,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        batch_size=current_batch_size * accumulation_steps
                    )

                    # 记录日志 - 每步都显示学习率和loss
                    avg_loss = epoch_loss / epoch_steps
                    elapsed = time.time() - start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # 显示学习率阶段
                    lr_phase = "Warmup" if step < self.config.warmup_steps else "Decay"
                    lr_progress = f"{step}/{self.config.warmup_steps}" if step < self.config.warmup_steps else f"{step}/{self.config.max_steps}"
                    
                    print(f"Step {step:5d} | Loss: {actual_loss:.4f} | Avg: {avg_loss:.4f} | LR: {current_lr:.2e} ({lr_phase} {lr_progress}) | Time: {elapsed/60:.1f}min")

                    # 保存检查点 - 每100步自动保存
                    if step % 100 == 0:
                        self._save_checkpoint(model, tokenizer, optimizer, step, actual_loss)

                    # 评估模型
                    if step % self.config.eval_steps == 0:
                        self._evaluate_model(model, tokenizer)

                    if step >= self.config.max_steps:
                        break

            # 检查是否需要退出epoch循环
            if step >= self.config.max_steps or self.interrupted:
                break

        # 关闭监控器并保存总结
        monitor.close()

        # 如果是中断，先保存checkpoint以便恢复
        if self.interrupted:
            print(f"\n💾 正在保存中断checkpoint...")
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / max(epoch_steps, 1),
                'config': self.config,
                'mode': self.mode
            }, checkpoint_path)
            print(f"✅ Checkpoint已保存: {checkpoint_path}")
            print(f"💡 可使用 --auto-resume 从此处恢复训练")

        # 保存最终模型
        final_path = os.path.join(self.output_dir, "final_model.pt")
        print(f"\n💾 正在保存最终模型...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_vocab_size': tokenizer.vocab_size,
            'config': self.config,
            'mode': self.mode,
            'step': step
        }, final_path)

        # 根据是否中断显示不同的消息
        if self.interrupted:
            print(f"\n⚠️  训练被用户中断")
            print(f"✅ 已成功保存中断时的模型状态")
        else:
            print(f"\n🎉 {self.mode}训练完成！")
        
        print(f"总步数: {step}")
        print(f"训练时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"最终模型已保存: {final_path}")
        print(f"📊 TensorBoard日志: {tensorboard_dir}")
        print(f"💡 查看训练过程: tensorboard --logdir={tensorboard_dir}")

        return final_path

    def _save_checkpoint(self, model, tokenizer, optimizer, step, loss):
        """保存检查点（只保留最新的一个）"""
        # 删除旧的checkpoint文件
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        old_checkpoints = glob.glob(checkpoint_pattern)
        for old_ckpt in old_checkpoints:
            try:
                os.remove(old_ckpt)
                print(f"🗑️  删除旧checkpoint: {os.path.basename(old_ckpt)}")
            except Exception as e:
                print(f"⚠️  删除旧checkpoint失败: {e}")
        
        # 保存新的checkpoint
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

    def _find_latest_checkpoint(self, search_pretrain_if_not_found=True):
        """查找最新的checkpoint文件
        
        参数:
            search_pretrain_if_not_found: 如果是SFT/DPO/RLHF模式且找不到checkpoint，
                                         是否查找pretrain的checkpoint
        """
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            # 尝试查找 final_model.pt
            final_model = os.path.join(self.output_dir, "final_model.pt")
            if os.path.exists(final_model):
                return final_model
            
            # 如果是 SFT/DPO/RLHF 模式，尝试从 pretrain 查找
            if search_pretrain_if_not_found and self.mode in ["sft", "dpo", "rlhf"]:
                print(f"   当前模式({self.mode})未找到checkpoint，尝试查找 pretrain checkpoint...")
                pretrain_dir = os.path.join(self.config.checkpoint_dir, f"pretrain_{self.config.model_size}")
                
                # 查找 pretrain 的中间 checkpoint
                pretrain_pattern = os.path.join(pretrain_dir, "checkpoint_step_*.pt")
                pretrain_checkpoints = glob.glob(pretrain_pattern)
                
                if pretrain_checkpoints:
                    # 按步数排序，返回最新的
                    def get_step_num(filename):
                        try:
                            basename = os.path.basename(filename)
                            step_str = basename.replace("checkpoint_step_", "").replace(".pt", "")
                            return int(step_str)
                        except:
                            return 0
                    
                    pretrain_checkpoints.sort(key=get_step_num, reverse=True)
                    print(f"   ✅ 找到 pretrain checkpoint: {pretrain_checkpoints[0]}")
                    return pretrain_checkpoints[0]
                
                # 查找 pretrain 的 final_model.pt
                pretrain_final = os.path.join(pretrain_dir, "final_model.pt")
                if os.path.exists(pretrain_final):
                    print(f"   ✅ 找到 pretrain final_model: {pretrain_final}")
                    return pretrain_final
            
            return None
        
        # 按步数排序，返回最新的
        def get_step_num(filename):
            try:
                # 从文件名中提取步数: checkpoint_step_5000.pt -> 5000
                basename = os.path.basename(filename)
                step_str = basename.replace("checkpoint_step_", "").replace(".pt", "")
                return int(step_str)
            except:
                return 0
        
        checkpoint_files.sort(key=get_step_num, reverse=True)
        return checkpoint_files[0]
    
    def _load_checkpoint(self, checkpoint_path, model, optimizer):
        """加载checkpoint并恢复训练状态
        
        返回:
            start_step: 从哪一步开始继续训练
        """
        print(f"🔄 正在加载checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 模型权重已加载")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 模型权重已加载（旧格式）")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ 优化器状态已加载")
                except Exception as e:
                    print(f"⚠️  优化器状态加载失败: {e}")
                    print("   将使用新的优化器状态")
            
            # 获取训练步数
            start_step = checkpoint.get('step', 0)
            if start_step > 0:
                print(f"✅ 将从第 {start_step} 步继续训练")
            
            # 显示checkpoint信息
            if 'loss' in checkpoint:
                print(f"📊 Checkpoint损失: {checkpoint['loss']:.4f}")
            if 'mode' in checkpoint:
                print(f"📝 训练模式: {checkpoint['mode']}")
            
            return start_step
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}")
            print("   将从头开始训练")
            return 0


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

    # Checkpoint恢复
    parser.add_argument('--resume', '--resume-from-checkpoint', type=str, default=None,
                        dest='resume_from_checkpoint',
                        help='从指定checkpoint文件继续训练（例如: checkpoints/sft_medium/checkpoint_step_5000.pt）')
    parser.add_argument('--auto-resume', action='store_true',
                        help='自动从最新的checkpoint恢复训练。注意：SFT/DPO/RLHF模式会自动加载pretrain权重作为初始化')

    # 训练参数覆盖
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='学习率')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大训练步数')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='学习率warmup步数（如果不指定，将根据训练模式自动设置）')

    args = parser.parse_args()

    # 获取配置
    config = get_config(args.config)

    # 应用命令行参数覆盖（优先级最低）
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # 根据训练模式调整配置（会自动设置warmup_steps）
    if args.mode == "pretrain":
        config.max_steps = config.max_steps or 50000
        if args.learning_rate is None:
            config.learning_rate = 1e-4  # 预训练基础学习率
        # 预训练：从零开始，需要较长warmup稳定训练
        config.warmup_steps = min(500, int(config.max_steps * 0.05))  # 5% 或最多500步
        print("📚 预训练模式：建立基础语言理解能力")
        print(f"   学习率: {config.learning_rate:.2e}")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
    elif args.mode == "sft":
        config.max_steps = config.max_steps or 10000
        # SFT微调：使用更小的学习率避免破坏预训练知识
        if args.learning_rate is None:  # 只在用户未指定时设置默认值
            config.learning_rate = 5e-5  # 比预训练低一个数量级
        # SFT：已有预训练基础，使用较短warmup快速适应
        config.warmup_steps = min(200, int(config.max_steps * 0.02))  # 2% 或最多200步
        print("🎯 监督微调模式：训练对话和特定任务能力")
        print(f"   学习率: {config.learning_rate:.2e} (比预训练低，保护已学知识)")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
        print(f"   💡 模型已有预训练基础，使用短warmup快速进入衰减阶段")
    elif args.mode == "dpo":
        config.max_steps = config.max_steps or 5000
        if args.learning_rate is None:
            config.learning_rate = 1e-5  # DPO使用更小学习率
        # DPO：在SFT基础上微调，使用极短warmup
        config.warmup_steps = min(100, int(config.max_steps * 0.02))  # 2% 或最多100步
        print("⚖️  直接偏好优化模式：根据人类偏好调整响应")
        print(f"   学习率: {config.learning_rate:.2e}")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
        print(f"   💡 在SFT基础上优化，使用极短warmup")
    elif args.mode == "rlhf":
        config.max_steps = config.max_steps or 3000
        if args.learning_rate is None:
            config.learning_rate = 1e-5  # RLHF使用小学习率
        # RLHF：在已训练模型上强化学习，使用极短warmup
        config.warmup_steps = min(100, int(config.max_steps * 0.02))  # 2% 或最多100步
        print("🔄 强化学习微调模式：通过奖励模型优化")
        print(f"   学习率: {config.learning_rate:.2e}")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
        print(f"   💡 在已训练模型上强化学习，使用极短warmup")

    # 命令行参数覆盖warmup_steps（优先级最高）
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
        print(f"⚙️  使用自定义warmup步数: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")

    # 创建训练器
    trainer = MiniGPTTrainer(config, mode=args.mode)

    # 显示恢复信息
    if args.auto_resume:
        print("🔄 启用自动恢复模式")
    elif args.resume_from_checkpoint:
        print(f"🔄 将从checkpoint恢复: {args.resume_from_checkpoint}")

    # 开始训练
    final_model_path = trainer.train(
        resume_from=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
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