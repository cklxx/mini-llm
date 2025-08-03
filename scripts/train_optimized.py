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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import multiprocessing
import psutil


def print_progress_bar(current, total, prefix='', suffix='', length=40, fill='█', empty='░'):
    """打印动态进度条"""
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = fill * filled_length + empty * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)


def optimize_pytorch_performance(num_threads=None, enable_optimizations=True):
    """优化PyTorch性能设置"""
    print("🚀 优化PyTorch性能设置...")
    
    # 获取系统信息
    cpu_count = multiprocessing.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    
    # 自动确定最佳线程数
    if num_threads is None:
        # 使用物理核心数，但不超过8个线程（避免过度并行）
        num_threads = min(physical_cores or cpu_count, 8)
    
    print(f"   系统CPU核心: {cpu_count} (物理: {physical_cores})")
    print(f"   设置线程数: {num_threads}")
    
    if enable_optimizations:
        # 设置PyTorch线程数
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        
        # 启用PyTorch优化
        torch.backends.cudnn.benchmark = True  # 如果有CUDA
        torch.backends.cudnn.deterministic = False  # 提高性能，牺牲一定的确定性
        
        # 启用JIT融合优化
        try:
            torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 2)])
            print("   ✅ JIT融合优化已启用")
        except:
            print("   ⚠️  JIT融合优化不可用")
        
            # CUDA优化（英伟达GPU）
    if torch.cuda.is_available():
        print("   ✅ CUDA 可用")
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"   🔧 CUDA 设备: {device_name} (设备 {current_device}/{device_count})")
        
        # CUDA特定优化
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
        torch.backends.cudnn.deterministic = False  # 允许非确定性提高性能
        
        # 设置CUDA内存分配策略
        try:
            torch.cuda.empty_cache()  # 清空缓存
            # 设置内存分配策略为扩展分配
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            print("   🔧 CUDA 内存优化已启用")
        except Exception as e:
            print(f"   ⚠️  CUDA 内存优化设置失败: {e}")
        
        print("   🔧 CUDA 优化配置已启用")
    
    # MPS优化（Mac GPU）
    elif torch.backends.mps.is_available():
        print("   ✅ MPS (Metal Performance Shaders) 可用")
        # MPS特定优化
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 减少内存使用
        # 启用MPS特定的优化
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 允许CPU回退
        # 设置MPS内存分配策略
        if hasattr(torch.mps, 'set_per_process_memory_fraction'):
            torch.mps.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
        print("   🔧 MPS 优化配置已启用")
        
        # Intel MKL优化
        try:
            torch.backends.mkldnn.enabled = True
            print("   ✅ Intel MKL-DNN 优化已启用")
        except:
            print("   ⚠️  Intel MKL-DNN 不可用")
        
        # 设置环境变量优化
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        
        print("   ✅ 环境变量已优化")
    
    return num_threads


def get_optimal_dataloader_workers(batch_size, device_type="cpu"):
    """获取最佳的DataLoader worker数量"""
    cpu_count = multiprocessing.cpu_count()
    
    # 对于batch_size=1的测试情况，禁用多进程避免worker崩溃
    if batch_size == 1:
        print(f"   🔧 测试模式 (batch_size=1): 禁用DataLoader多进程 (workers=0)")
        return 0
    
    if device_type == "cuda":
        # CUDA设备可以使用更多worker，因为GPU处理速度快
        optimal_workers = min(8, cpu_count)  # CUDA可以使用更多worker
        print(f"   🔧 CUDA设备优化: 使用 {optimal_workers} 个DataLoader worker")
    elif device_type == "mps":
        # MPS设备时，减少worker数量避免瓶颈
        optimal_workers = min(1, cpu_count // 4)  # MPS使用更少worker
        print(f"   🔧 MPS设备优化: 使用 {optimal_workers} 个DataLoader worker")
    elif batch_size <= 4:
        # 小批次时使用更少的worker
        optimal_workers = min(2, cpu_count // 2)
    else:
        # 大批次时可以使用更多worker
        optimal_workers = min(4, cpu_count // 2)
    
    print(f"   📊 推荐DataLoader workers: {optimal_workers} (CPU核心: {cpu_count}, 设备: {device_type}, batch_size: {batch_size})")
    return max(0, optimal_workers)


def compile_model_if_supported(model, device_type="cpu"):
    """如果支持，编译模型以提高性能"""
    try:
        # PyTorch 2.0+ 的torch.compile
        if hasattr(torch, 'compile'):
            print("🔥 编译模型以提高性能...")
            # 使用适合的编译模式
            if device_type == "cuda":
                # CUDA设备使用最激进的编译优化
                try:
                    compiled_model = torch.compile(model, mode="max-autotune", dynamic=True)
                    print(f"   ✅ CUDA模型编译成功，使用 max-autotune 模式")
                except Exception as e:
                    print(f"   ⚠️  CUDA模型编译失败，尝试 reduce-overhead 模式: {e}")
                    try:
                        compiled_model = torch.compile(model, mode="reduce-overhead")
                        print(f"   ✅ CUDA模型编译成功，使用 reduce-overhead 模式")
                    except Exception as e2:
                        print(f"   ⚠️  CUDA模型编译完全失败，使用原始模型: {e2}")
                        compiled_model = model
            elif device_type == "mps":
                # MPS设备使用专门的编译优化
                try:
                    # 尝试使用MPS特定的优化模式
                    compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
                    print(f"   ✅ MPS模型编译成功，使用 reduce-overhead 模式")
                except Exception as e:
                    print(f"   ⚠️  MPS模型编译失败，使用原始模型: {e}")
                    compiled_model = model
            else:
                # CPU设备使用轻量级编译
                try:
                    compiled_model = torch.compile(model, mode="reduce-overhead")
                    print(f"   ✅ CPU模型编译成功，使用 reduce-overhead 模式")
                except Exception as e:
                    print(f"   ⚠️  CPU模型编译失败，使用原始模型: {e}")
                    compiled_model = model
            
            print("   ✅ 模型编译成功")
            return compiled_model
        else:
            print("   ℹ️  模型编译不可用或不适用")
            return model
    except Exception as e:
        print(f"   ⚠️  模型编译失败: {e}")
        return model

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
        
        # 损失历史记录
        self.loss_history = []
        self.step_history = []
        
        # 恢复训练相关
        self.start_step = 0
        self.resume_checkpoint_path = None
        
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
    
    def train(self, resume_checkpoint_path=None, auto_resume=False):
        """执行训练"""
        try:
            # 显示系统信息
            self._print_system_info()
            
            # 处理checkpoint恢复
            checkpoint = None
            if resume_checkpoint_path:
                checkpoint = self.load_checkpoint(resume_checkpoint_path)
            elif auto_resume:
                latest_checkpoint = self.find_latest_checkpoint(self.config.output_dir)
                if latest_checkpoint:
                    print(f"🔄 发现已有检查点，自动恢复训练...")
                    checkpoint = self.load_checkpoint(latest_checkpoint)
                else:
                    print("📝 未找到已有检查点，开始全新训练")
            
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
            
            # 如果有checkpoint，恢复模型状态
            if checkpoint and 'model_state_dict' in checkpoint:
                print("🔧 恢复模型状态...")
                model.load_state_dict(checkpoint['model_state_dict'])
            
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
        
        # 检查是否需要重新训练tokenizer
        retrain_tokenizer = getattr(self.config, 'retrain_tokenizer', False)
        
        if os.path.exists(tokenizer_path) and not retrain_tokenizer:
            print(f"加载现有分词器: {tokenizer_path}")
            tokenizer = BPETokenizer(vocab_size=self.config.tokenizer.vocab_size)
            tokenizer.load(tokenizer_path)
        else:
            if retrain_tokenizer:
                print("🔄 重新训练分词器...")
            else:
                print("训练新的分词器...")
            
            # 从多个训练数据文件构建分词器
            texts = []
            import json
            
            # 获取训练样本数量限制
            tokenizer_samples = getattr(self.config, 'tokenizer_samples', 100000)
            
            for train_file in self.config.data.train_files:
                data_path = os.path.join(self.config.data.data_dir, train_file)
                print(f"   读取文件: {train_file}")
                
                if not os.path.exists(data_path):
                    print(f"   ⚠️  文件不存在，跳过: {data_path}")
                    continue
                
                file_texts = []
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        for line_no, line in enumerate(f):
                            try:
                                data = json.loads(line.strip())
                                if 'text' in data and len(data['text'].strip()) > 10:
                                    file_texts.append(data['text'])
                                
                                # 限制每个文件的样本数量
                                if len(file_texts) >= tokenizer_samples // len(self.config.data.train_files):
                                    break
                            except json.JSONDecodeError:
                                if line_no < 10:  # 只报告前10个错误
                                    print(f"   跳过无效行 {line_no}")
                                continue
                    
                    texts.extend(file_texts)
                    print(f"     加载了 {len(file_texts)} 条文本")
                    
                except Exception as e:
                    print(f"   ❌ 读取文件失败: {e}")
                    continue
            
            if not texts:
                raise RuntimeError("没有找到有效的训练文本，无法训练分词器")
            
            print(f"🔤 从 {len(texts)} 条文本训练分词器 (词汇表大小: {self.config.tokenizer.vocab_size})...")
            tokenizer = BPETokenizer(vocab_size=self.config.tokenizer.vocab_size)
            tokenizer.train(texts)
            
            tokenizer.save(tokenizer_path)
            print(f"✅ 分词器已保存: {tokenizer_path}")
        
        print(f"📊 词汇表大小: {tokenizer.vocab_size}")
        return tokenizer
    
    def _setup_data_loader(self, tokenizer):
        """设置数据加载器"""
        print("📚 设置数据加载器...")
        
        # 获取数据量限制
        max_data_size = getattr(self.config, 'max_data_size', 0)
        
        # 读取多个训练数据文件
        all_texts = []
        import json
        
        for train_file in self.config.data.train_files:
            data_path = os.path.join(self.config.data.data_dir, train_file)
            print(f"📖 加载数据文件: {train_file}")
            
            if not os.path.exists(data_path):
                print(f"   ⚠️  文件不存在，跳过: {data_path}")
                continue
            
            file_texts = []
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            text = None
                            
                            # 处理不同的数据格式
                            if 'text' in data:
                                text = data['text']
                            elif 'conversation' in data:
                                # 处理对话格式
                                conv_text = ""
                                for turn in data['conversation']:
                                    conv_text += f"{turn.get('human', '')} {turn.get('assistant', '')} "
                                text = conv_text.strip()
                            elif 'input' in data and 'output' in data:
                                # 处理输入输出格式
                                text = f"{data['input']} {data['output']}"
                            
                            if text and len(text.strip()) > 10:
                                file_texts.append(text.strip())
                            
                            # 如果有数据量限制，检查是否达到上限
                            if max_data_size > 0:
                                per_file_limit = max_data_size // len(self.config.data.train_files)
                                if len(file_texts) >= per_file_limit:
                                    break
                                    
                        except json.JSONDecodeError:
                            if line_no < 10:  # 只报告前10个错误
                                print(f"   跳过无效行 {line_no}")
                            continue
                
                all_texts.extend(file_texts)
                print(f"     加载了 {len(file_texts)} 条文本")
                
            except Exception as e:
                print(f"   ❌ 读取文件失败: {e}")
                continue
        
        if not all_texts:
            raise RuntimeError("没有找到有效的训练文本")
        
        # 如果有总数据量限制，截断数据
        if max_data_size > 0 and len(all_texts) > max_data_size:
            import random
            random.shuffle(all_texts)
            all_texts = all_texts[:max_data_size]
            print(f"📊 限制数据量为 {max_data_size} 条")
        
        print(f"📊 总共加载 {len(all_texts)} 条训练文本")
        
        # 创建数据集
        dataset = LanguageModelingDataset(
            texts=all_texts,
            tokenizer=tokenizer,
            max_length=self.config.data.max_seq_len
        )
        
        # 优化DataLoader worker数量
        dataloader_workers = getattr(self.config, 'dataloader_workers', None)
        if dataloader_workers is None:
            device_type = self.config.device if hasattr(self.config, 'device') else 'cpu'
            dataloader_workers = get_optimal_dataloader_workers(
                batch_size=self.config.data.batch_size,
                device_type=device_type
            )
        else:
            # 使用用户指定的worker数量
            pass
        
        print(f"📊 DataLoader workers: {dataloader_workers}")
        
        # 检测设备类型以优化DataLoader设置
        device_type = getattr(self.config, 'device', 'cpu')
        if device_type.startswith('cuda'):
            pin_memory = True  # CUDA设备启用pin_memory加速数据传输
            print("   🔧 CUDA设备: 启用pin_memory加速数据传输")
        else:
            pin_memory = False  # CPU/MPS设备避免内存问题
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=self.config.data.shuffle,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,  # CUDA设备启用，其他设备禁用
            persistent_workers=dataloader_workers > 0,  # 持久化worker提高效率
            prefetch_factor=2 if dataloader_workers > 0 else None,  # 预取因子
            drop_last=True  # 丢弃最后一个不完整的batch，避免尺寸问题
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
        
        # 应用模型编译优化（如果启用）
        enable_compile = getattr(self.config, 'enable_compile', False)
        if enable_compile:
            device_type = str(device).split(':')[0]  # 获取设备类型 (cpu, cuda, mps)
            model = compile_model_if_supported(model, device_type)
        
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
        
        # 如果有checkpoint，恢复优化器状态
        if hasattr(self, 'resume_checkpoint_path') and self.resume_checkpoint_path:
            checkpoint = torch.load(self.resume_checkpoint_path, map_location='cpu', weights_only=False)
            if 'optimizer_state_dict' in checkpoint:
                print("🔧 恢复优化器状态...")
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 设置损失函数
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        
        # 训练统计
        # 初始化步数计数器，考虑checkpoint恢复
        self.step_count = self.start_step
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
                    
                    # 验证batch尺寸
                    if batch.size(1) < 2:
                        print(f"⚠️  跳过太短的batch: {batch.size()}")
                        continue
                    
                    # 准备输入和目标
                    input_ids = batch[:, :-1]
                    target_ids = batch[:, 1:]
                    
                    # 验证输入尺寸
                    if input_ids.size(1) == 0 or target_ids.size(1) == 0:
                        print(f"⚠️  跳过空的输入或目标: input={input_ids.size()}, target={target_ids.size()}")
                        continue
                    
                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(input_ids)
                    
                    # 验证输出尺寸
                    if outputs.size(0) == 0 or outputs.size(-1) == 0:
                        print(f"⚠️  跳过无效的模型输出: {outputs.size()}")
                        continue
                    
                    # 计算损失
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
                    
                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  跳过无效损失: {loss.item()}")
                        continue
                    
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
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"📂 加载检查点: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"✅ 检查点信息:")
        print(f"   步数: {checkpoint.get('step', 'unknown')}")
        
        loss_value = checkpoint.get('loss', 'unknown')
        if isinstance(loss_value, (int, float)):
            print(f"   损失: {loss_value:.4f}")
        else:
            print(f"   损失: {loss_value}")
        
        # 恢复损失历史
        if 'loss_history' in checkpoint and 'step_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
            self.step_history = checkpoint['step_history']
            print(f"   已恢复 {len(self.loss_history)} 个历史损失记录")
            
            # 绘制恢复的损失曲线
            if len(self.loss_history) > 0:
                current_step = checkpoint.get('step', 0)
                print(f"📊 绘制恢复的损失曲线...")
                plot_loss = getattr(self.config, 'plot_loss', False)
                if plot_loss:
                    self._plot_and_save_loss_curve(current_step, recovered=True)
        else:
            print("   未找到历史损失记录，从当前步开始记录")
        
        # 设置起始步数
        self.start_step = checkpoint.get('step', 0)
        self.resume_checkpoint_path = checkpoint_path
        
        return checkpoint
    
    def find_latest_checkpoint(self, output_dir):
        """找到最新的检查点文件"""
        if not os.path.exists(output_dir):
            return None
        
        # 查找所有检查点文件
        checkpoint_files = []
        for file in os.listdir(output_dir):
            if file.startswith('checkpoint_step_') and file.endswith('.pt'):
                try:
                    step = int(file.replace('checkpoint_step_', '').replace('.pt', ''))
                    checkpoint_files.append((step, os.path.join(output_dir, file)))
                except ValueError:
                    continue
        
        # 检查final_model.pt
        final_model_path = os.path.join(output_dir, 'final_model.pt')
        if os.path.exists(final_model_path):
            checkpoint_files.append((float('inf'), final_model_path))
        
        if checkpoint_files:
            # 返回最新的检查点
            latest = max(checkpoint_files, key=lambda x: x[0])
            return latest[1]
        
        return None

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
        
        # 绘制并保存损失曲线（如果启用）
        plot_loss = getattr(self.config, 'plot_loss', False)
        if plot_loss and len(self.loss_history) > 0:
            self._plot_and_save_loss_curve(step, final=final)
    
    def _plot_and_save_loss_curve(self, current_step, final=False):
        """绘制并保存损失曲线"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 主损失曲线
            plt.subplot(2, 1, 1)
            plt.plot(self.step_history, self.loss_history, 'b-', linewidth=1.5, alpha=0.8, label='Training Loss')
            plt.title(f'Training Loss Curve (Step {current_step})', fontsize=14, fontweight='bold')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Loss Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 显示统计信息
            if len(self.loss_history) > 1:
                current_loss = self.loss_history[-1]
                min_loss = min(self.loss_history)
                max_loss = max(self.loss_history)
                avg_loss = sum(self.loss_history) / len(self.loss_history)
                
                stats_text = f'Current: {current_loss:.4f} | Min: {min_loss:.4f} | Max: {max_loss:.4f} | Avg: {avg_loss:.4f}'
                plt.figtext(0.5, 0.46, stats_text, ha='center', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # 最近1000步损失曲线
            plt.subplot(2, 1, 2)
            if len(self.loss_history) > 1000:
                recent_steps = self.step_history[-1000:]
                recent_losses = self.loss_history[-1000:]
                plt.plot(recent_steps, recent_losses, 'r-', linewidth=1.5, label='Recent 1000 Steps')
                plt.title('Recent 1000 Steps Loss', fontsize=12)
            else:
                plt.plot(self.step_history, self.loss_history, 'r-', linewidth=1.5, label='All Steps')
                plt.title('Loss (Detailed View)', fontsize=12)
            
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Loss Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # 保存图片
            plot_dir = os.path.join(self.config.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # 保存当前损失曲线
            if final:
                current_plot_path = os.path.join(plot_dir, f"loss_curve_final_step_{current_step}.png")
            else:
                current_plot_path = os.path.join(plot_dir, f"loss_curve_step_{current_step}.png")
            plt.savefig(current_plot_path, dpi=300, bbox_inches='tight')
            
            # 总是更新最新的损失曲线
            latest_plot_path = os.path.join(plot_dir, "loss_curve_latest.png")
            plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
            
            plt.close()  # 释放内存
            
            if final:
                print(f"🎯 Final loss curve saved: {current_plot_path}")
            else:
                print(f"📊 Loss curve saved: {current_plot_path}")
                
        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")
    
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
    parser = argparse.ArgumentParser(description='Mac优化训练脚本 - 支持全量数据和自定义配置')
    parser.add_argument('--config', choices=['tiny', 'small', 'medium'], default='medium',
                        help='选择配置 (tiny: 超小模型, small: 小模型, medium: 中模型)')
    
    # 数据相关参数
    parser.add_argument('--use-full-data', action='store_true',
                        help='使用全量数据集进行训练')
    parser.add_argument('--data-files', nargs='+', 
                        default=['pretrain_hq.jsonl', 'sft_mini_512.jsonl'],
                        help='指定训练数据文件列表')
    parser.add_argument('--max-data-size', type=int, default=500000,
                        help='最大数据条数限制 (0表示不限制)')
    
    # Tokenizer相关参数
    parser.add_argument('--retrain-tokenizer', action='store_true',
                        help='重新训练tokenizer')
    parser.add_argument('--tokenizer-vocab-size', type=int, default=15000,
                        help='Tokenizer词汇表大小')
    parser.add_argument('--tokenizer-samples', type=int, default=100000,
                        help='训练tokenizer使用的样本数量')
    
    # 训练参数调整
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='学习率 (不指定则使用配置默认值)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大训练步数 (不指定则使用配置默认值)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小 (不指定则使用配置默认值)')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='预热步数 (不指定则使用配置默认值)')
    
    # 系统资源参数
    parser.add_argument('--max-cpu', type=float, default=85.0,
                        help='最大CPU使用率 (%%)')
    parser.add_argument('--max-memory', type=float, default=85.0,
                        help='最大内存使用率 (%%)')
    parser.add_argument('--disable-monitoring', action='store_true',
                        help='禁用资源监控')
    
    # 多线程和性能优化参数
    parser.add_argument('--num-threads', type=int, default=None,
                        help='PyTorch线程数 (不指定则自动优化)')
    parser.add_argument('--dataloader-workers', type=int, default=None,
                        help='DataLoader worker数量 (不指定则自动优化)')
    parser.add_argument('--enable-compile', action='store_true',
                        help='启用PyTorch模型编译优化 (需要PyTorch 2.0+)')
    parser.add_argument('--disable-optimizations', action='store_true',
                        help='禁用所有性能优化')
    
    # 输出和保存参数
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (不指定则使用配置默认值)')
    parser.add_argument('--save-steps', type=int, default=None,
                        help='保存检查点的步数间隔')
    parser.add_argument('--plot-loss', action='store_true',
                        help='实时绘制和保存损失曲线')
    
    # Checkpoint resume options
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='从指定的检查点文件恢复训练 (例如: checkpoints/mac_medium/checkpoint_step_4000.pt)')
    parser.add_argument('--auto-resume', action='store_true',
                        help='自动从最新的检查点恢复训练')
    
    args = parser.parse_args()
    
    # 获取基础配置
    if args.config == 'tiny':
        config = get_mac_tiny_config()
        print("使用超小模型配置 (最快验证)")
    elif args.config == 'small':
        config = get_mac_small_config()
        print("使用小模型配置 (平衡性能)")
    else: # medium
        config = get_mac_medium_config()
        print("使用中模型配置 (性能与资源平衡)")
    
    # 应用命令行参数覆盖配置
    print("\n📝 应用自定义配置...")
    
    # 数据配置
    if args.use_full_data:
        print("✅ 启用全量数据训练")
        config.data.train_files = [
            "pretrain_hq.jsonl",
            "sft_1024.jsonl", 
            "sft_512.jsonl",
            "r1_mix_1024.jsonl"
        ]
        args.max_data_size = 0  # 不限制数据量
    else:
        config.data.train_files = args.data_files
        print(f"使用指定数据文件: {args.data_files}")
    
    # Tokenizer配置
    if args.retrain_tokenizer:
        print("✅ 启用重新训练tokenizer")
        config.tokenizer.vocab_size = args.tokenizer_vocab_size
        print(f"Tokenizer词汇表大小: {args.tokenizer_vocab_size}")
    
    # 训练参数覆盖
    # 设备检测和特殊优化配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🔧 检测到CUDA设备，应用CUDA特定优化...")
        # CUDA设备优化配置
        if args.batch_size is None and hasattr(config, 'pretrain') and hasattr(config.pretrain, 'batch_size'):
            original_batch = config.pretrain.batch_size
            # CUDA设备可以使用更大的batch_size
            config.pretrain.batch_size = min(original_batch * 2, 16)
            print(f"   📊 CUDA优化: batch_size从 {original_batch} 调整为 {config.pretrain.batch_size}")
        
        # CUDA设备可以使用更高的学习率
        if args.learning_rate is None and hasattr(config, 'pretrain') and hasattr(config.pretrain, 'learning_rate'):
            original_lr = config.pretrain.learning_rate
            # CUDA设备可以使用稍高的学习率
            config.pretrain.learning_rate = max(original_lr, 1e-4)
            print(f"   📊 CUDA优化: learning_rate从 {original_lr} 调整为 {config.pretrain.learning_rate}")
            
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🔧 检测到MPS设备，应用MPS特定优化...")
        # MPS设备优化配置（保持原有逻辑）
        if args.batch_size is None and hasattr(config, 'pretrain') and hasattr(config.pretrain, 'batch_size'):
            original_batch = config.pretrain.batch_size
            # MPS设备推荐使用较小的batch_size以避免内存问题
            config.pretrain.batch_size = min(original_batch, 4)
            print(f"   📊 MPS优化: batch_size从 {original_batch} 调整为 {config.pretrain.batch_size}")
        
        # 为MPS设备调整学习率
        if args.learning_rate is None and hasattr(config, 'pretrain') and hasattr(config.pretrain, 'learning_rate'):
            # MPS设备可能需要稍低的学习率以保持稳定性
            original_lr = config.pretrain.learning_rate
            config.pretrain.learning_rate = min(original_lr, 5e-5)
            print(f"   📊 MPS优化: learning_rate从 {original_lr} 调整为 {config.pretrain.learning_rate}")
    else:
        device = torch.device("cpu")
        print("🔧 使用CPU设备，应用CPU特定优化...")
        # CPU设备优化配置
        if args.batch_size is None and hasattr(config, 'pretrain') and hasattr(config.pretrain, 'batch_size'):
            original_batch = config.pretrain.batch_size
            # CPU设备使用更小的batch_size以避免内存压力
            config.pretrain.batch_size = min(original_batch, 2)
            print(f"   📊 CPU优化: batch_size从 {original_batch} 调整为 {config.pretrain.batch_size}")

    if args.learning_rate is not None:
        config.pretrain.learning_rate = args.learning_rate
        print(f"学习率: {args.learning_rate}")
    
    if args.max_steps is not None:
        config.pretrain.max_steps = args.max_steps
        print(f"最大训练步数: {args.max_steps}")
    
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
        print(f"批次大小: {args.batch_size}")
    
    if args.warmup_steps is not None:
        config.pretrain.warmup_steps = args.warmup_steps
        print(f"预热步数: {args.warmup_steps}")
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        print(f"输出目录: {args.output_dir}")
    
    if args.save_steps is not None:
        config.pretrain.save_steps = args.save_steps
        print(f"保存间隔: {args.save_steps}")
    
    # 为配置添加额外属性
    config.retrain_tokenizer = args.retrain_tokenizer
    config.tokenizer_samples = args.tokenizer_samples
    config.max_data_size = args.max_data_size
    config.plot_loss = args.plot_loss
    config.enable_compile = args.enable_compile
    config.num_threads = args.num_threads
    config.dataloader_workers = args.dataloader_workers
    
    # 应用PyTorch性能优化
    print("\n🚀 应用性能优化...")
    enable_optimizations = not args.disable_optimizations
    optimized_threads = optimize_pytorch_performance(
        num_threads=args.num_threads,
        enable_optimizations=enable_optimizations
    )
    
    # 资源监控配置
    resource_config = MacResourceConfig(
        max_cpu_percent=args.max_cpu,
        max_memory_percent=args.max_memory,
        enable_monitoring=not args.disable_monitoring
    )
    
    print("\n" + "="*60)
    print("🚀 开始训练")
    print("="*60)
    
    # 创建训练器
    trainer = OptimizedTrainer(config, resource_config)
    
    # 开始训练
    trainer.train(
        resume_checkpoint_path=args.resume_from_checkpoint,
        auto_resume=args.auto_resume
    )


if __name__ == "__main__":
    main() 