"""
训练脚本
支持预训练、SFT、DPO等多种训练模式
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import create_model
from tokenizer.bpe_tokenizer import BPETokenizer, train_tokenizer_from_data
from tokenizer.tokenizer_manager import get_tokenizer
from data.dataset_loader import create_data_loader, DatasetConfig
from training.trainer import create_trainer, LanguageModelingDataset, ConversationDataset
from config.training_config import TrainingConfig, get_small_config, get_tiny_config


def setup_device():
    """设置训练设备"""
    if torch.backends.mps.is_available():
        device = "mps"
        print("使用Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"使用CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("使用CPU")
    
    return device


def train_or_load_tokenizer(config: TrainingConfig, force_retrain: bool = False):
    """使用智能tokenizer管理系统训练或加载分词器"""
    print("🔧 Using smart tokenizer management system...")

    # 选择训练数据
    data_path = os.path.join(config.data.data_dir, config.data.train_files[0])

    # 使用智能tokenizer管理器
    tokenizer = get_tokenizer(
        data_path=data_path,
        vocab_size=config.tokenizer.vocab_size,
        tokenizer_type="bpe",
        force_retrain=force_retrain,
        cache_dir=os.path.join(config.output_dir, "tokenizers")
    )

    # 为了向后兼容，同时保存到原来的位置
    tokenizer_path = os.path.join(config.output_dir, "tokenizer.pkl")
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"📁 Tokenizer also saved to: {tokenizer_path}")

    return tokenizer


def prepare_data(config: TrainingConfig, tokenizer, training_mode: str):
    """准备训练数据"""
    if training_mode == "pretrain":
        # 预训练数据
        data_config = DatasetConfig(
            data_path=os.path.join(config.data.data_dir, "pretrain_hq.jsonl"),
            max_length=config.data.max_seq_len
        )
        loader = create_data_loader("pretrain", data_config)
        texts = loader.load_pretrain_data()
        
        dataset = LanguageModelingDataset(texts, tokenizer, config.data.max_seq_len)
        
    elif training_mode == "sft":
        # SFT数据
        data_config = DatasetConfig(
            data_path=os.path.join(config.data.data_dir, config.data.train_files[0]),
            max_length=config.data.max_seq_len
        )
        loader = create_data_loader("sft", data_config)
        conversations = loader.load_conversations()
        
        dataset = ConversationDataset(conversations, tokenizer, config.data.max_seq_len)
        
    else:
        raise ValueError(f"不支持的训练模式: {training_mode}")
    
    # 划分训练集和验证集
    train_size = int((1 - config.data.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    ) if val_size > 0 else None
    
    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="MiniGPT训练脚本")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft", "dpo"], 
                       default="sft", help="训练模式")
    parser.add_argument("--config", type=str, choices=["tiny", "small", "medium"], 
                       default="small", help="模型配置")
    parser.add_argument("--data-dir", type=str, default="data/dataset/minimind_dataset",
                       help="数据目录")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                       help="输出目录")
    parser.add_argument("--resume", type=str, default=None,
                       help="从checkpoint恢复训练")
    parser.add_argument("--retrain-tokenizer", action="store_true",
                       help="重新训练分词器")
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device()
    
    # 加载配置
    if args.config == "tiny":
        config = get_tiny_config()
    elif args.config == "small":
        config = get_small_config()
    else:
        raise ValueError(f"不支持的配置: {args.config}")
    
    # 更新配置
    config.device = device
    config.data.data_dir = args.data_dir
    config.output_dir = args.output_dir
    
    print(f"训练模式: {args.mode}")
    print(f"模型配置: {args.config}")
    print(f"设备: {device}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # 训练或加载分词器
    tokenizer = train_or_load_tokenizer(config, args.retrain_tokenizer)
    print(f"分词器词汇表大小: {tokenizer.get_vocab_size()}")
    
    # 更新模型词汇表大小
    config.model.vocab_size = tokenizer.get_vocab_size()
    
    # 创建模型
    model = create_model(
        vocab_size=config.model.vocab_size,
        model_size=config.model.model_size
    )
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 准备数据
    train_dataloader, val_dataloader = prepare_data(config, tokenizer, args.mode)
    print(f"训练数据批次数: {len(train_dataloader)}")
    if val_dataloader:
        print(f"验证数据批次数: {len(val_dataloader)}")
    
    # 创建训练器
    trainer = create_trainer(args.mode, model, tokenizer, device)
    
    # 从checkpoint恢复（如果指定）
    if args.resume:
        print(f"从checkpoint恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    if args.mode == "pretrain":
        num_epochs = config.pretrain.max_steps // len(train_dataloader) + 1
        trainer.train(train_dataloader, val_dataloader, num_epochs, config.output_dir)
    elif args.mode == "sft":
        trainer.train(train_dataloader, val_dataloader, config.sft.max_epochs, config.output_dir)
    
    print("训练完成！")


if __name__ == "__main__":
    main()