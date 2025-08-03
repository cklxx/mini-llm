#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从检查点继续训练脚本
支持从现有checkpoint恢复训练，使用完整的预训练数据集
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import create_model
from tokenizer.bpe_tokenizer import BPETokenizer
from training.trainer import LanguageModelingDataset
from config.mac_optimized_config import (
    get_mac_medium_config, MacResourceConfig, MacResourceMonitor,
    estimate_model_size, validate_config_for_mac, get_system_info
)


def print_progress_bar(current, total, prefix='', suffix='', length=40, fill='█', empty='░'):
    """打印动态进度条"""
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = fill * filled_length + empty * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="", flush=True)


class ContinueTrainer:
    """从检查点继续训练的训练器"""
    
    def __init__(self, checkpoint_path, config, resource_config: MacResourceConfig):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.resource_config = resource_config
        self.resource_monitor = MacResourceMonitor(resource_config)
        self.should_stop = False
        self.pause_training = False
        
        # 损失历史记录
        self.loss_history = []
        self.step_history = []
        
        # 注册信号处理器
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
                
                time.sleep(self.resource_config.monitoring_interval)
                
            except Exception as e:
                print(f"资源监控错误: {e}")
                time.sleep(5)
    
    def load_checkpoint(self):
        """加载检查点"""
        print(f"📂 加载检查点: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"✅ 检查点信息:")
        print(f"   步数: {checkpoint.get('step', 'unknown')}")
        print(f"   损失: {checkpoint.get('loss', 'unknown'):.4f}")
        
        # 恢复损失历史
        if 'loss_history' in checkpoint and 'step_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
            self.step_history = checkpoint['step_history']
            print(f"   已恢复 {len(self.loss_history)} 个历史损失记录")
            
            # 绘制恢复的损失曲线
            if len(self.loss_history) > 0:
                current_step = checkpoint.get('step', 0)
                print(f"📊 绘制恢复的损失曲线...")
                self._plot_and_save_loss_curve(current_step, recovered=True)
        else:
            print("   未找到历史损失记录，从当前步开始记录")
        
        return checkpoint
    
    def setup_for_continue_training(self):
        """设置继续训练所需的组件"""
        print("🔧 设置继续训练环境...")
        
        # 加载检查点
        checkpoint = self.load_checkpoint()
        
        # 加载分词器
        tokenizer_path = os.path.join(os.path.dirname(self.checkpoint_path), "tokenizer.pkl")
        if not os.path.exists(tokenizer_path):
            # 尝试从checkpoints目录加载
            tokenizer_path = "checkpoints/tokenizer.pkl"
        
        print(f"🔤 加载分词器: {tokenizer_path}")
        tokenizer = BPETokenizer(vocab_size=self.config.tokenizer.vocab_size)
        tokenizer.load(tokenizer_path)
        
        # 创建模型
        print("🧠 创建并加载模型...")
        device = torch.device(self.config.device)
        model = create_model(tokenizer.vocab_size, self.config.model.model_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 创建优化器并加载状态
        print("⚙️ 创建并加载优化器...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.pretrain.learning_rate,
            weight_decay=self.config.pretrain.weight_decay
        )
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 设置数据加载器（使用完整数据集）
        print("📚 设置数据加载器（完整预训练数据）...")
        data_loader = self._setup_data_loader(tokenizer)
        
        return model, tokenizer, optimizer, checkpoint, data_loader
    
    def _setup_data_loader(self, tokenizer):
        """设置数据加载器，使用完整的预训练数据集"""
        # 使用多个预训练文件
        train_files = [
            "pretrain_hq.jsonl",      # 高质量预训练数据
            "pretrain_large_sample.jsonl",  # 大样本数据
            "sft_1024.jsonl",         # 监督微调数据
            "sft_2048.jsonl",         # 更长序列数据
        ]
        
        all_texts = []
        
        for file_name in train_files:
            file_path = os.path.join(self.config.data.data_dir, file_name)
            if os.path.exists(file_path):
                print(f"📖 加载数据文件: {file_name}")
                
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_texts = []
                        for line_num, line in enumerate(f):
                            try:
                                data = json.loads(line.strip())
                                if 'text' in data:
                                    file_texts.append(data['text'])
                                elif 'conversation' in data:
                                    # 处理对话格式
                                    conv_text = ""
                                    for turn in data['conversation']:
                                        conv_text += f"{turn.get('human', '')} {turn.get('assistant', '')} "
                                    if conv_text.strip():
                                        file_texts.append(conv_text.strip())
                                elif 'input' in data and 'output' in data:
                                    # 处理输入输出格式
                                    file_texts.append(f"{data['input']} {data['output']}")
                            except json.JSONDecodeError as e:
                                if line_num < 10:  # 只显示前10个错误
                                    print(f"   跳过无效行 {line_num}: {e}")
                                continue
                        
                        all_texts.extend(file_texts)
                        print(f"   加载了 {len(file_texts)} 条文本")
                        
                except Exception as e:
                    print(f"   ⚠️  加载文件失败: {e}")
                    continue
            else:
                print(f"   ⚠️  文件不存在: {file_name}")
        
        print(f"📊 总共加载 {len(all_texts)} 条训练文本")
        
        if not all_texts:
            # 如果没有加载到数据，使用fallback
            print("⚠️  没有加载到数据，使用备用数据集...")
            fallback_path = os.path.join(self.config.data.data_dir, "pretrain_200.jsonl")
            with open(fallback_path, 'r', encoding='utf-8') as f:
                import json
                for line in f:
                    data = json.loads(line.strip())
                    all_texts.append(data['text'])
            print(f"使用备用数据集，共 {len(all_texts)} 条文本")
        
        # 创建数据集
        dataset = LanguageModelingDataset(
            texts=all_texts,
            tokenizer=tokenizer,
            max_length=self.config.data.max_seq_len
        )
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=self.config.data.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=False
        )
        
        print(f"📊 数据批次数: {len(data_loader)}")
        return data_loader
    
    def continue_training(self):
        """继续训练"""
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
            
            # 设置训练组件
            model, tokenizer, optimizer, checkpoint, data_loader = self.setup_for_continue_training()
            
            # 获取起始步数
            start_step = checkpoint.get('step', 0)
            
            # 执行继续训练
            self._run_continue_training(model, tokenizer, optimizer, data_loader, start_step)
            
        except KeyboardInterrupt:
            print("\n用户中断训练")
        except Exception as e:
            print(f"\n训练错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.should_stop = True
            print("继续训练结束")
    
    def _print_system_info(self):
        """打印系统信息"""
        print("=" * 60)
        print("🔄 从检查点继续训练 - Mac优化版")
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
        
        print(f"\n⚙️  继续训练配置:")
        print(f"检查点路径: {self.checkpoint_path}")
        print(f"批次大小: {self.config.data.batch_size}")
        print(f"目标步数: {self.config.pretrain.max_steps}")
        print(f"学习率: {self.config.pretrain.learning_rate}")
        print(f"设备: {self.config.device}")
        print("-" * 60)
    
    def _run_continue_training(self, model, tokenizer, optimizer, data_loader, start_step):
        """执行继续训练循环"""
        print(f"🏃 从步骤 {start_step} 继续训练...")
        
        # 设置损失函数
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        
        # 训练统计
        self.step_count = start_step
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
                
                # 检查是否达到最大步数
                if self.step_count >= self.config.pretrain.max_steps:
                    print(f"\n🎯 达到最大训练步数: {self.config.pretrain.max_steps}")
                    self.should_stop = True
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
                    
                    # 记录损失历史
                    self.loss_history.append(loss.item())
                    self.step_history.append(self.step_count)
                    
                    # 记录日志
                    if self.step_count % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        elapsed = time.time() - start_time
                        steps_per_sec = (self.step_count - start_step) / elapsed if elapsed > 0 else 0
                        progress = (self.step_count / self.config.pretrain.max_steps) * 100
                        
                        suffix = f"损失 {loss.item():.4f} | 平均 {avg_loss:.4f} | {steps_per_sec:.1f} steps/s"
                        print_progress_bar(self.step_count, self.config.pretrain.max_steps, 
                                         prefix=f'🔄 继续训练 (轮次 {epoch})', suffix=suffix)
                    
                    # 保存检查点
                    if self.step_count % self.config.pretrain.save_steps == 0:
                        print(f"\n💾 步骤 {self.step_count} - 保存检查点...")
                        self._save_checkpoint(model, tokenizer, optimizer, self.step_count, loss.item())
                        print()
                    
                    # 评估模型
                    if self.step_count % self.config.pretrain.eval_steps == 0:
                        print(f"\n🧪 步骤 {self.step_count} - 模型评估...")
                        self._evaluate_model(model, tokenizer)
                        print()
                        
                except Exception as e:
                    print(f"\n训练步骤错误: {e}")
                    continue
            
            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                print(f"\n📈 轮次 {epoch} 完成 - 平均损失: {avg_epoch_loss:.4f}")
                
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    print(f"🎉 新的最佳损失: {best_loss:.4f}")
        
        print(f"\n🏁 继续训练完成!")
        print(f"总步数: {self.step_count}")
        print(f"最佳损失: {best_loss:.4f}")
        print(f"训练时间: {(time.time() - start_time) / 60:.1f}分钟")
        
        # 保存最终模型
        self._save_checkpoint(model, tokenizer, optimizer, self.step_count, best_loss, final=True)
        
        # 保存最终损失曲线
        if len(self.loss_history) > 0:
            print("📊 保存最终训练损失曲线...")
            self._plot_and_save_loss_curve(self.step_count, final=True)
    
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
            'loss_history': self.loss_history,
            'step_history': self.step_history,
        }, checkpoint_path)
        
        print(f"💾 检查点已保存: {checkpoint_path}")
        
        # 绘制并保存损失曲线
        if len(self.loss_history) > 0:
            self._plot_and_save_loss_curve(step)
    
    def _plot_and_save_loss_curve(self, current_step, final=False, recovered=False):
        """绘制并保存损失曲线"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 主损失曲线
            plt.subplot(2, 1, 1)
            plt.plot(self.step_history, self.loss_history, 'b-', linewidth=1.5, alpha=0.8, label='训练损失')
            plt.title(f'训练损失曲线 (Step {current_step})', fontsize=14, fontweight='bold')
            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('损失值', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 显示统计信息
            if len(self.loss_history) > 1:
                current_loss = self.loss_history[-1]
                min_loss = min(self.loss_history)
                max_loss = max(self.loss_history)
                avg_loss = sum(self.loss_history) / len(self.loss_history)
                
                stats_text = f'当前: {current_loss:.4f} | 最小: {min_loss:.4f} | 最大: {max_loss:.4f} | 平均: {avg_loss:.4f}'
                plt.figtext(0.5, 0.46, stats_text, ha='center', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # 最近1000步损失曲线（如果有的话）
            plt.subplot(2, 1, 2)
            if len(self.loss_history) > 1000:
                recent_steps = self.step_history[-1000:]
                recent_losses = self.loss_history[-1000:]
                plt.plot(recent_steps, recent_losses, 'r-', linewidth=1.5, label='最近1000步')
                plt.title('最近1000步损失曲线', fontsize=12)
            else:
                plt.plot(self.step_history, self.loss_history, 'r-', linewidth=1.5, label='所有损失')
                plt.title('损失曲线（放大视图）', fontsize=12)
            
            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('损失值', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # 保存图片
            plot_dir = os.path.join(self.config.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # 保存当前损失曲线
            if final:
                current_plot_path = os.path.join(plot_dir, f"loss_curve_final_step_{current_step}.png")
            elif recovered:
                current_plot_path = os.path.join(plot_dir, f"loss_curve_recovered_step_{current_step}.png")
            else:
                current_plot_path = os.path.join(plot_dir, f"loss_curve_step_{current_step}.png")
            plt.savefig(current_plot_path, dpi=300, bbox_inches='tight')
            
            # 总是更新最新的损失曲线
            latest_plot_path = os.path.join(plot_dir, "loss_curve_latest.png")
            plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
            
            plt.close()  # 释放内存
            
            if final:
                print(f"🎯 最终损失曲线已保存: {current_plot_path}")
            elif recovered:
                print(f"🔄 恢复的损失曲线已保存: {current_plot_path}")
            else:
                print(f"📊 损失曲线已保存: {current_plot_path}")
            
        except Exception as e:
            print(f"⚠️  绘制损失曲线失败: {e}")
    
    def _evaluate_model(self, model, tokenizer):
        """简单评估模型"""
        model.eval()
        
        try:
            test_prompts = [
                "你好，",
                "人工智能",
                "今天天气",
                "学习编程"
            ]
            
            for prompt in test_prompts:
                print(f"\n🧪 测试输入: '{prompt}'")
                
                # 编码输入
                input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                device = next(model.parameters()).device
                input_tensor = torch.tensor([input_ids], device=device)
                
                # 生成
                with torch.no_grad():
                    for _ in range(15):  # 生成15个token
                        outputs = model(input_tensor)
                        next_token_logits = outputs[0, -1, :]
                        next_token = torch.argmax(next_token_logits).item()
                        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
                
                # 解码结果
                generated_ids = input_tensor[0].cpu().tolist()
                generated_text = tokenizer.decode(generated_ids)
                print(f"🤖 生成: '{generated_text}'")
                
                # 只显示前两个测试样例，避免输出过长
                if prompt == test_prompts[1]:
                    break
            
        except Exception as e:
            print(f"生成测试失败: {e}")
        finally:
            model.train()


def main():
    parser = argparse.ArgumentParser(description='从检查点继续训练脚本')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/mac_medium/final_model.pt',
                        help='检查点文件路径')
    parser.add_argument('--max-steps', type=int, default=8000,
                        help='继续训练的最大步数')
    parser.add_argument('--max-cpu', type=float, default=85.0,
                        help='最大CPU使用率 (%)')
    parser.add_argument('--max-memory', type=float, default=85.0,
                        help='最大内存使用率 (%)')
    parser.add_argument('--disable-monitoring', action='store_true',
                        help='禁用资源监控')
    
    args = parser.parse_args()
    
    # 获取medium配置
    config = get_mac_medium_config()
    
    # 更新最大步数
    config.pretrain.max_steps = args.max_steps
    
    print(f"使用medium模型配置")
    print(f"从检查点继续训练: {args.checkpoint}")
    print(f"目标最大步数: {args.max_steps}")
    
    # 资源监控配置
    resource_config = MacResourceConfig(
        max_cpu_percent=args.max_cpu,
        max_memory_percent=args.max_memory,
        enable_monitoring=not args.disable_monitoring
    )
    
    # 创建继续训练器
    trainer = ContinueTrainer(args.checkpoint, config, resource_config)
    
    # 开始继续训练
    trainer.continue_training()


if __name__ == "__main__":
    main()