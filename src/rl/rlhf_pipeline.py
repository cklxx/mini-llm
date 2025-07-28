"""
RLHF完整流程管道

实现从人类反馈的强化学习（RLHF）的完整训练流程。
RLHF包含三个主要阶段：

1. 监督微调（SFT）: 使用高质量数据微调预训练模型
2. 奖励模型训练（RM）: 学习人类偏好，训练奖励模型
3. 强化学习（RL）: 使用PPO优化策略，最大化奖励

这个管道提供了统一的接口来管理整个RLHF流程。
"""

import os
import sys
import json
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

# 添加项目根目录到路径（用于独立运行）
if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, 'src'))

# 导入各个组件
from ..training.trainer import SFTTrainer, create_trainer
from .reward_model.reward_trainer import RewardTrainer, create_reward_model, create_reward_trainer
from .reward_model.preference_data import create_preference_dataloader
from .ppo.ppo_trainer import PPOTrainer, create_ppo_trainer
from .ppo.value_model import create_value_model


@dataclass
class RLHFConfig:
    """RLHF配置"""
    
    # 模型配置
    model_name: str = "minigpt"
    tokenizer_path: str = "tokenizer.pkl"
    device: str = "auto"
    
    # SFT配置
    sft_data_path: str = "data/sft_data.jsonl"
    sft_epochs: int = 3
    sft_lr: float = 5e-5
    sft_batch_size: int = 32
    
    # 奖励模型配置
    reward_data_path: str = "data/preference_data.jsonl"
    reward_epochs: int = 5
    reward_lr: float = 5e-5
    reward_batch_size: int = 32
    freeze_reward_backbone: bool = False
    
    # PPO配置
    ppo_data_path: str = "data/ppo_prompts.jsonl"
    ppo_iterations: int = 1000
    ppo_lr_policy: float = 1e-5
    ppo_lr_value: float = 3e-4
    ppo_batch_size: int = 32
    ppo_mini_batch_size: int = 8
    ppo_epochs: int = 4
    
    # 通用配置
    max_length: int = 512
    save_dir: str = "rlhf_outputs"
    save_interval: int = 100
    log_level: str = "INFO"


class RLHFPipeline:
    """RLHF训练管道"""
    
    def __init__(self, config: RLHFConfig):
        """
        初始化RLHF管道
        
        Args:
            config: RLHF配置
        """
        self.config = config
        
        # 设置日志
        self._setup_logging()
        
        # 设置设备
        self.device = self._setup_device()
        
        # 创建输出目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 保存配置
        self._save_config()
        
        # 初始化组件
        self.base_model = None
        self.sft_model = None
        self.reward_model = None
        self.ppo_trainer = None
        self.tokenizer = None
        
        self.logger.info("RLHF管道初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RLHF")
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.config.device
        
        self.logger.info(f"使用设备: {device}")
        return device
    
    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.config.save_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
    
    def load_tokenizer(self):
        """加载分词器"""
        import pickle
        with open(self.config.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.logger.info("分词器加载完成")
    
    def load_base_model(self, model_path: str):
        """
        加载基础模型
        
        Args:
            model_path: 模型路径
        """
        # 这里需要根据实际的模型加载逻辑进行实现
        # 假设我们有一个create_model函数
        from ..model.transformer import create_model
        
        if os.path.exists(model_path):
            # 加载已保存的模型
            checkpoint = torch.load(model_path, map_location=self.device)
            self.base_model = create_model(vocab_size=len(self.tokenizer), model_size="small")
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 创建新模型
            self.base_model = create_model(vocab_size=len(self.tokenizer), model_size="small")
        
        self.base_model.to(self.device)
        self.logger.info("基础模型加载完成")
    
    def run_sft(self) -> str:
        """
        运行监督微调（SFT）
        
        Returns:
            SFT模型保存路径
        """
        self.logger.info("开始SFT训练...")
        
        # 创建SFT trainer
        sft_trainer = create_trainer(
            training_type='sft',
            model=self.base_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # 加载SFT数据
        # 这里需要根据实际数据格式实现数据加载
        sft_data = self._load_sft_data()
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        from ..training.trainer import ConversationDataset
        
        dataset = ConversationDataset(sft_data, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.sft_batch_size, shuffle=True)
        
        # 训练
        sft_save_dir = os.path.join(self.config.save_dir, "sft")
        sft_trainer.train(
            train_dataloader=dataloader,
            num_epochs=self.config.sft_epochs,
            save_dir=sft_save_dir
        )
        
        # 保存SFT模型
        sft_model_path = os.path.join(sft_save_dir, "best_model.pt")
        self.sft_model = sft_trainer.model
        
        self.logger.info(f"SFT训练完成，模型保存至: {sft_model_path}")
        return sft_model_path
    
    def run_reward_training(self, sft_model_path: str) -> str:
        """
        运行奖励模型训练
        
        Args:
            sft_model_path: SFT模型路径
            
        Returns:
            奖励模型保存路径
        """
        self.logger.info("开始奖励模型训练...")
        
        # 加载SFT模型作为backbone
        if self.sft_model is None:
            checkpoint = torch.load(sft_model_path, map_location=self.device)
            self.sft_model = self.base_model
            self.sft_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建奖励模型
        self.reward_model = create_reward_model(
            self.sft_model, 
            freeze_backbone=self.config.freeze_reward_backbone
        )
        
        # 创建奖励训练器
        reward_trainer = create_reward_trainer(
            self.reward_model,
            self.tokenizer,
            self.device,
            learning_rate=self.config.reward_lr
        )
        
        # 加载偏好数据
        train_dataloader = create_preference_dataloader(
            self.config.reward_data_path,
            self.tokenizer,
            batch_size=self.config.reward_batch_size,
            max_length=self.config.max_length
        )
        
        # 训练奖励模型
        reward_save_dir = os.path.join(self.config.save_dir, "reward_model")
        reward_trainer.train(
            train_dataloader=train_dataloader,
            num_epochs=self.config.reward_epochs,
            save_dir=reward_save_dir
        )
        
        # 保存奖励模型
        reward_model_path = os.path.join(reward_save_dir, "best_model.pt")
        
        self.logger.info(f"奖励模型训练完成，模型保存至: {reward_model_path}")
        return reward_model_path
    
    def run_ppo_training(self, sft_model_path: str, reward_model_path: str) -> str:
        """
        运行PPO训练
        
        Args:
            sft_model_path: SFT模型路径
            reward_model_path: 奖励模型路径
            
        Returns:
            PPO模型保存路径
        """
        self.logger.info("开始PPO训练...")
        
        # 加载策略模型（SFT模型）
        if self.sft_model is None:
            checkpoint = torch.load(sft_model_path, map_location=self.device)
            self.sft_model = self.base_model
            self.sft_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载奖励模型
        if self.reward_model is None:
            reward_checkpoint = torch.load(reward_model_path, map_location=self.device)
            self.reward_model = create_reward_model(self.sft_model)
            self.reward_model.load_state_dict(reward_checkpoint['model_state_dict'])
        
        # 创建价值模型
        value_model = create_value_model(self.sft_model)
        
        # 创建PPO训练器
        self.ppo_trainer = create_ppo_trainer(
            policy_model=self.sft_model,
            value_model=value_model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            device=self.device,
            lr_policy=self.config.ppo_lr_policy,
            lr_value=self.config.ppo_lr_value,
            batch_size=self.config.ppo_batch_size,
            mini_batch_size=self.config.ppo_mini_batch_size,
            ppo_epochs=self.config.ppo_epochs
        )
        
        # 加载PPO提示数据
        prompts = self._load_ppo_prompts()
        
        # PPO训练
        ppo_save_dir = os.path.join(self.config.save_dir, "ppo")
        self.ppo_trainer.train(
            prompts=prompts,
            num_iterations=self.config.ppo_iterations,
            save_interval=self.config.save_interval,
            save_dir=ppo_save_dir
        )
        
        # 保存最终模型
        final_model_path = os.path.join(ppo_save_dir, "final_model.pt")
        
        self.logger.info(f"PPO训练完成，模型保存至: {final_model_path}")
        return final_model_path
    
    def run_full_pipeline(self, base_model_path: str) -> str:
        """
        运行完整的RLHF流程
        
        Args:
            base_model_path: 基础模型路径
            
        Returns:
            最终模型保存路径
        """
        self.logger.info("开始完整RLHF流程...")
        
        # 加载分词器和基础模型
        self.load_tokenizer()
        self.load_base_model(base_model_path)
        
        # 第一阶段：SFT
        sft_model_path = self.run_sft()
        
        # 第二阶段：奖励模型训练
        reward_model_path = self.run_reward_training(sft_model_path)
        
        # 第三阶段：PPO训练
        final_model_path = self.run_ppo_training(sft_model_path, reward_model_path)
        
        self.logger.info("完整RLHF流程完成！")
        return final_model_path
    
    def _load_sft_data(self) -> List[Dict]:
        """加载SFT数据"""
        sft_data = []
        with open(self.config.sft_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                sft_data.append(data)
        return sft_data
    
    def _load_ppo_prompts(self) -> List[str]:
        """加载PPO提示数据"""
        prompts = []
        with open(self.config.ppo_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'prompt' in data:
                    prompts.append(data['prompt'])
                elif isinstance(data, str):
                    prompts.append(data)
        return prompts
    
    def evaluate_model(self, model_path: str, test_data_path: str) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model_path: 模型路径
            test_data_path: 测试数据路径
            
        Returns:
            评估结果
        """
        self.logger.info("开始模型评估...")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        model = self.base_model
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 实现评估逻辑
        # 这里可以添加各种评估指标：困惑度、BLEU、Rouge等
        
        eval_results = {
            'perplexity': 0.0,  # 困惑度
            'bleu_score': 0.0,  # BLEU分数
            'rouge_score': 0.0,  # ROUGE分数
        }
        
        self.logger.info("模型评估完成")
        return eval_results
    
    def generate_sample(self, prompt: str, model_path: Optional[str] = None) -> str:
        """
        生成样本文本
        
        Args:
            prompt: 输入提示
            model_path: 模型路径（可选）
            
        Returns:
            生成的文本
        """
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = self.base_model
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = self.sft_model or self.base_model
        
        model.eval()
        
        # 实现文本生成逻辑
        # 这里需要根据实际的生成器实现
        
        return "生成的样本文本"


def create_rlhf_pipeline(config: Union[RLHFConfig, Dict, str]) -> RLHFPipeline:
    """
    创建RLHF管道
    
    Args:
        config: 配置对象、字典或配置文件路径
        
    Returns:
        RLHF管道实例
    """
    if isinstance(config, str):
        # 从文件加载配置
        with open(config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = RLHFConfig(**config_dict)
    elif isinstance(config, dict):
        config = RLHFConfig(**config)
    
    return RLHFPipeline(config)


# 示例配置
def get_default_config() -> RLHFConfig:
    """获取默认配置"""
    return RLHFConfig(
        # 基础配置
        model_name="minigpt",
        tokenizer_path="checkpoints/tokenizer.pkl",
        device="auto",
        
        # 数据路径
        sft_data_path="data/dataset/minimind_dataset/sft_mini_512.jsonl",
        reward_data_path="data/dataset/minimind_dataset/dpo.jsonl",
        ppo_data_path="data/dataset/minimind_dataset/pretrain_hq.jsonl",
        
        # 训练参数
        sft_epochs=3,
        reward_epochs=5,
        ppo_iterations=1000,
        
        # 输出配置
        save_dir="rlhf_outputs",
        save_interval=100
    )


if __name__ == "__main__":
    # 示例使用
    print("RLHF管道实现完成")
    print("主要组件：")
    print("- RLHFConfig: 配置类")
    print("- RLHFPipeline: RLHF训练管道")
    print("- create_rlhf_pipeline: 工厂函数")
    print("- get_default_config: 默认配置")
    
    # 使用示例
    config = get_default_config()
    pipeline = create_rlhf_pipeline(config)
    print("\\n可以通过以下方式运行完整流程：")
    print("pipeline.run_full_pipeline('path/to/base_model.pt')")