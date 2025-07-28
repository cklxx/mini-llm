#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mac优化训练脚本
集成资源监控，防止系统卡死，使用最小数据集快速验证智能效果
"""
import os
import sys
import argparse
import time
import signal
import threading
from datetime import datetime
import torch
from torch.utils.data import DataLoader


def print_progress_bar(current, total, prefix='', suffix='', length=40, fill='█', empty='░'):
    """打印动态进度条"""
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = fill * filled_length + empty * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import create_model
from tokenizer.bpe_tokenizer import BPETokenizer, train_tokenizer_from_data
from data.dataset_loader import create_data_loader, DatasetConfig
from training.trainer import create_trainer, LanguageModelingDataset, ConversationDataset
from config.mac_optimized_config import (
    get_mac_tiny_config, get_mac_small_config, get_mac_medium_config, MacResourceConfig, 
    MacResourceMonitor, estimate_model_size, validate_config_for_mac,
    get_system_info
)


class OptimizedTrainer:
    """优化的训练器，带资源监控"""
    
    def __init__(self, config, resource_config: MacResourceConfig):
        self.config = config
        self.resource_config = resource_config
        self.resource_monitor = MacResourceMonitor(resource_config)
        self.should_stop = False
        self.pause_training = False
        self.training_thread = None
        
        # 注册信号处理器，优雅退出
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在安全退出...")
        self.should_stop = True
        
    def _monitor_resources(self):
        """资源监控线程"""
        while not self.should_stop:
            try:
                resources = self.resource_monitor.check_resources()
                
                # 检查是否需要暂停训练
                if self.resource_monitor.should_pause_training():
                    if not self.pause_training:
                        print(f"\n⚠️  资源使用过高，暂停训练:")
                        print(f"   CPU: {resources['cpu_percent']:.1f}% (限制: {self.resource_config.max_cpu_percent}%)")
                        print(f"   内存: {resources['memory_percent']:.1f}% (限制: {self.resource_config.max_memory_percent}%)")
                        self.pause_training = True
                else:
                    if self.pause_training:
                        print(f"\n✅ 资源使用恢复正常，继续训练")
                        self.pause_training = False
                
                # 只在资源异常时记录
                if hasattr(self, 'step_count') and (
                    resources['cpu_percent'] > 80 or resources['memory_percent'] > 80
                ) and self.step_count % 50 == 0:
                    print(f"\n⚠️  高资源使用 - CPU: {resources['cpu_percent']:.1f}%, 内存: {resources['memory_percent']:.1f}%")
                
                time.sleep(self.resource_config.monitoring_interval)
                
            except Exception as e:
                print(f"资源监控错误: {e}")
                time.sleep(5)
    
    def train(self):
        """执行训练"""
        try:
            # 显示系统信息
            self._print_system_info()
            
            # 验证配置
            warnings = validate_config_for_mac(self.config)
            if warnings:
                print("⚠️  配置警告:")
                for warning in warnings:
                    print(f"   - {warning}")
                input("按回车键继续，或Ctrl+C退出...")
            
            # 启动资源监控
            if self.resource_config.enable_monitoring:
                print("🔍 启动资源监控...")
                monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
                monitor_thread.start()
            
            # 加载或训练分词器
            tokenizer = self._setup_tokenizer()
            
            # 创建数据加载器
            data_loader = self._setup_data_loader(tokenizer)
            
            # 创建模型
            model = self._setup_model(tokenizer)
            
            # 执行训练
            self._run_training(model, data_loader, tokenizer)
            
        except KeyboardInterrupt:
            print("\n用户中断训练")
        except Exception as e:
            print(f"\n训练错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.should_stop = True
            print("训练结束")
    
    def _print_system_info(self):
        """打印系统信息"""
        print("=" * 60)
        print("🚀 Mac优化训练 - 智能效果验证")
        print("=" * 60)
        
        sys_info = get_system_info()
        print(f"系统平台: {sys_info['platform']}")
        print(f"CPU核心数: {sys_info['cpu_count']}")
        print(f"总内存: {sys_info['memory_gb']:.1f}GB")
        print(f"可用内存: {sys_info['available_memory_gb']:.1f}GB")
        
        model_info = estimate_model_size(self.config)
        print(f"\n📊 模型信息:")
        print(f"预估参数量: {model_info['total_params']:,}")
        print(f"模型内存: {model_info['model_memory_mb']:.1f}MB")
        print(f"训练内存: {model_info['training_memory_mb']:.1f}MB")
        
        print(f"\n⚙️  训练配置:")
        print(f"数据文件: {self.config.data.train_files}")
        print(f"批次大小: {self.config.data.batch_size}")
        print(f"最大步数: {self.config.pretrain.max_steps}")
        print(f"学习率: {self.config.pretrain.learning_rate}")
        print(f"设备: {self.config.device}")
        print("-" * 60)
    
    def _setup_tokenizer(self):
        """设置分词器"""
        print("🔧 设置分词器...")
        
        tokenizer_path = os.path.join(self.config.output_dir, "tokenizer.pkl")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if os.path.exists(tokenizer_path):
            print(f"加载现有分词器: {tokenizer_path}")
            tokenizer = BPETokenizer(vocab_size=self.config.tokenizer.vocab_size)
            tokenizer.load(tokenizer_path)
        else:
            print("训练新的分词器...")
            # 从训练数据构建分词器
            data_path = os.path.join(self.config.data.data_dir, self.config.data.train_files[0])
            
            # 读取文本数据
            texts = []
            import json
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    texts.append(data['text'])
            
            print(f"从 {len(texts)} 条文本训练分词器...")
            tokenizer = BPETokenizer(vocab_size=self.config.tokenizer.vocab_size)
            tokenizer.train(texts)
            
            tokenizer.save(tokenizer_path)
            print(f"分词器已保存: {tokenizer_path}")
        
        print(f"词汇表大小: {tokenizer.vocab_size}")
        return tokenizer
    
    def _setup_data_loader(self, tokenizer):
        """设置数据加载器"""
        print("📚 设置数据加载器...")
        
        data_path = os.path.join(self.config.data.data_dir, self.config.data.train_files[0])
        
        # 读取训练数据
        texts = []
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                texts.append(data['text'])
        
        print(f"加载 {len(texts)} 条训练文本")
        
        # 创建数据集
        dataset = LanguageModelingDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config.data.max_seq_len
        )
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=self.config.data.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=False  # Mac上避免内存问题
        )
        
        print(f"数据批次数: {len(data_loader)}")
        return data_loader
    
    def _setup_model(self, tokenizer):
        """设置模型"""
        print("🧠 创建模型...")
        
        # 设置设备
        device = torch.device(self.config.device)
        print(f"使用设备: {device}")
        
        # 创建模型
        model = create_model(tokenizer.vocab_size, self.config.model.model_size)
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return model
    
    def _run_training(self, model, data_loader, tokenizer):
        """执行训练循环"""
        print("🏃 开始训练...")
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.pretrain.learning_rate,
            weight_decay=self.config.pretrain.weight_decay
        )
        
        # 设置损失函数
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        
        # 训练统计
        self.step_count = 0
        best_loss = float('inf')
        start_time = time.time()
        
        model.train()
        
        for epoch in range(1000):  # 最大epoch数
            if self.should_stop:
                break
                
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(data_loader):
                if self.should_stop:
                    break
                
                # 检查是否需要暂停
                while self.pause_training and not self.should_stop:
                    time.sleep(1)
                
                if self.should_stop:
                    break
                
                try:
                    # 数据移到设备
                    device = next(model.parameters()).device
                    batch = batch.to(device)
                    
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
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.pretrain.max_grad_norm)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 更新统计
                    epoch_loss += loss.item()
                    epoch_steps += 1
                    self.step_count += 1
                    
                    # 记录日志 - 使用覆盖式显示
                    if self.step_count % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        elapsed = time.time() - start_time
                        steps_per_sec = self.step_count / elapsed
                        progress = (self.step_count / self.config.pretrain.max_steps) * 100
                        
                        # 使用进度条显示
                        suffix = f"损失 {loss.item():.4f} | 平均 {avg_loss:.4f} | {steps_per_sec:.1f} steps/s"
                        print_progress_bar(self.step_count, self.config.pretrain.max_steps, 
                                         prefix=f'🏃 训练中 (轮次 {epoch})', suffix=suffix)
                    
                    # 保存检查点
                    if self.step_count % self.config.pretrain.save_steps == 0:
                        print(f"\n💾 步骤 {self.step_count} - 保存检查点...")
                        self._save_checkpoint(model, tokenizer, optimizer, self.step_count, loss.item())
                        print()  # 空行后继续显示进度条
                    
                    # 简单验证（生成一段文本）
                    if self.step_count % self.config.pretrain.eval_steps == 0:
                        print(f"\n🧪 步骤 {self.step_count} - 模型评估...")
                        self._evaluate_model(model, tokenizer)
                        print()  # 空行后继续显示进度条
                    
                    # 检查是否达到最大步数
                    if self.step_count >= self.config.pretrain.max_steps:
                        print(f"达到最大训练步数: {self.config.pretrain.max_steps}")
                        self.should_stop = True
                        break
                        
                except Exception as e:
                    print(f"训练步骤错误: {e}")
                    continue
            
            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                print(f"\n📈 轮次 {epoch} 完成 - 平均损失: {avg_epoch_loss:.4f}")
                
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    print(f"🎉 新的最佳损失: {best_loss:.4f}")
        
        print(f"\n🏁 训练完成!")
        print(f"总步数: {self.step_count}")
        print(f"最佳损失: {best_loss:.4f}")
        print(f"训练时间: {(time.time() - start_time) / 60:.1f}分钟")
        
        # 保存最终模型
        self._save_checkpoint(model, tokenizer, optimizer, self.step_count, best_loss, final=True)
    
    def _save_checkpoint(self, model, tokenizer, optimizer, step, loss, final=False):
        """保存检查点"""
        checkpoint_dir = self.config.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if final:
            checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
        }, checkpoint_path)
        
        print(f"💾 检查点已保存: {checkpoint_path}")
    
    def _evaluate_model(self, model, tokenizer):
        """简单评估模型（生成测试）"""
        model.eval()
        
        try:
            # 简单的生成测试
            test_prompt = "你好，"
            print(f"\n🧪 生成测试 - 输入: '{test_prompt}'")
            
            # 编码输入
            input_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
            device = next(model.parameters()).device  # 获取模型设备
            input_tensor = torch.tensor([input_ids], device=device)
            
            # 生成
            with torch.no_grad():
                for _ in range(10):  # 生成10个token
                    outputs = model(input_tensor)
                    next_token_logits = outputs[0, -1, :]
                    next_token = torch.argmax(next_token_logits).item()
                    input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            
            # 解码结果
            generated_ids = input_tensor[0].cpu().tolist()
            generated_text = tokenizer.decode(generated_ids)
            print(f"🤖 生成结果: '{generated_text}'")
            
        except Exception as e:
            print(f"生成测试失败: {e}")
        finally:
            model.train()


def main():
    parser = argparse.ArgumentParser(description='Mac优化训练脚本')
    parser.add_argument('--config', choices=['tiny', 'small', 'medium'], default='tiny',
                        help='选择配置 (tiny: 超小模型, small: 小模型, medium: 中模型)')
    parser.add_argument('--max-cpu', type=float, default=85.0,
                        help='最大CPU使用率 (%)')
    parser.add_argument('--max-memory', type=float, default=85.0,
                        help='最大内存使用率 (%)')
    parser.add_argument('--disable-monitoring', action='store_true',
                        help='禁用资源监控')
    
    args = parser.parse_args()
    
    # 获取配置
    if args.config == 'tiny':
        config = get_mac_tiny_config()
        print("使用超小模型配置 (最快验证)")
    elif args.config == 'small':
        config = get_mac_small_config()
        print("使用小模型配置 (平衡性能)")
    else: # medium
        config = get_mac_medium_config()
        print("使用中模型配置 (性能与资源平衡)")
    
    # 资源监控配置
    resource_config = MacResourceConfig(
        max_cpu_percent=args.max_cpu,
        max_memory_percent=args.max_memory,
        enable_monitoring=not args.disable_monitoring
    )
    
    # 创建训练器
    trainer = OptimizedTrainer(config, resource_config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main() 