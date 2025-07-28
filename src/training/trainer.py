"""
训练器模块
支持预训练、SFT、DPO等多种训练模式
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class LanguageModelingDataset(Dataset):
    """语言模型数据集"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📊 数据集初始化: {len(texts)} 条文本, 最大长度: {max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            
            # 编码文本
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # 截断或填充到固定长度
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids.extend([self.tokenizer.pad_id] * (self.max_length - len(token_ids)))
            
            result = torch.tensor(token_ids, dtype=torch.long)
            return result
            
        except Exception as e:
            print(f"❌ 处理数据项 {idx} 时发生错误: {e}")
            print(f"❌ 错误文本长度: {len(text)}")
            print(f"❌ 错误文本预览: {text[:200]}...")
            import traceback
            traceback.print_exc()
            raise e


class ConversationDataset(Dataset):
    """对话数据集"""
    
    def __init__(self, conversations: List[Dict], tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # 构造输入和标签
        input_text = conv['input']
        output_text = conv['output']
        
        # 编码输入和输出
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        
        # 构造完整序列：<BOS> input output <EOS>
        full_sequence = [self.tokenizer.bos_id] + input_ids + output_ids + [self.tokenizer.eos_id]
        
        # 截断到最大长度
        if len(full_sequence) > self.max_length:
            full_sequence = full_sequence[:self.max_length]
        
        # 创建标签（用于计算损失）
        labels = full_sequence[1:] + [self.tokenizer.pad_id]  # 向左移动一位
        
        # 填充
        while len(full_sequence) < self.max_length:
            full_sequence.append(self.tokenizer.pad_id)
            labels.append(self.tokenizer.pad_id)
        
        return {
            'input_ids': torch.tensor(full_sequence, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class PreTrainer:
    """预训练器"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算语言模型损失"""
        # 展平张量
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        # 忽略PAD token
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        loss = loss_fn(logits, labels)
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"🚀 开始训练epoch，数据批次: {len(dataloader)}")
        print(f"📊 设备: {self.device}")
        
        progress_bar = tqdm(dataloader, desc="预训练")
        print("📦 开始遍历数据批次...")
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                # 关闭调试模式，正常训练所有batch
                debug_mode = False
                
                if debug_mode:
                    print(f"\n--- 🔄 处理batch {batch_idx + 1} ---")
                
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    if debug_mode:
                        print(f"📦 字典格式 - input_ids: {input_ids.shape}")
                else:
                    input_ids = batch.to(self.device)
                    # 对于预训练，标签就是输入向右移动一位
                    labels = torch.cat([input_ids[:, 1:], 
                                      torch.full((input_ids.size(0), 1), 
                                               self.tokenizer.pad_id, device=self.device)], dim=1)
                    if debug_mode:
                        print(f"📦 Tensor格式 - input_ids: {input_ids.shape}")
                
                if debug_mode:
                    print("🧠 模型前向传播...")
                
                # 前向传播
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, labels)
                
                if debug_mode:
                    print(f"📉 损失: {loss.item():.4f}")
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if debug_mode:
                    print(f"✅ Batch {batch_idx + 1} 完成")
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
                
                # 调试模式下只处理前几个batch
                if debug_mode and batch_idx >= 2:
                    print("🔍 调试模式：处理3个batch后停止")
                    break
                    
        except Exception as e:
            print(f"❌ 训练错误: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"🎯 Epoch完成，平均损失: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="验证"):
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    input_ids = batch.to(self.device)
                    labels = torch.cat([input_ids[:, 1:], 
                                      torch.full((input_ids.size(0), 1), 
                                               self.tokenizer.pad_id, device=self.device)], dim=1)
                
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, 
              num_epochs: int = 10, save_dir: str = "checkpoints"):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                print(f"验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                    print("保存最佳模型")
            
            # 定期保存checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
        
        # 绘制训练曲线
        self.plot_training_curve(save_dir)
    
    def save_checkpoint(self, path: str):
        """保存模型checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载模型checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
    
    def plot_training_curve(self, save_dir: str):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='训练损失')
        
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-', label='验证损失')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练过程')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, 'training_curve.png'))
        plt.close()


class SFTTrainer(PreTrainer):
    """监督微调训练器"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        super().__init__(model, tokenizer, device)
        
        # SFT使用较小的学习率
        self.optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    input_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算SFT损失，只对输出部分计算损失"""
        # 展平张量
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        # 创建掩码，只对非PAD token计算损失
        mask = (labels != self.tokenizer.pad_id)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 计算损失
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits, labels)
        
        # 应用掩码
        masked_losses = losses * mask.float()
        
        return masked_losses.sum() / mask.sum()


class DPOTrainer:
    """DPO (Direct Preference Optimization) 训练器"""
    
    def __init__(self, model, reference_model, tokenizer, device='cpu', beta=0.1):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        
        self.model.to(device)
        self.reference_model.to(device)
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    def compute_dpo_loss(self, chosen_logits: torch.Tensor, rejected_logits: torch.Tensor,
                        chosen_labels: torch.Tensor, rejected_labels: torch.Tensor,
                        ref_chosen_logits: torch.Tensor, ref_rejected_logits: torch.Tensor) -> torch.Tensor:
        """计算DPO损失"""
        
        def get_log_probs(logits, labels):
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            # 只对非PAD token计算
            mask = (labels != self.tokenizer.pad_id)
            masked_log_probs = selected_log_probs * mask.float()
            
            return masked_log_probs.sum(dim=-1) / mask.sum(dim=-1)
        
        # 计算策略模型的log概率
        policy_chosen_log_probs = get_log_probs(chosen_logits, chosen_labels)
        policy_rejected_log_probs = get_log_probs(rejected_logits, rejected_labels)
        
        # 计算参考模型的log概率
        ref_chosen_log_probs = get_log_probs(ref_chosen_logits, chosen_labels)
        ref_rejected_log_probs = get_log_probs(ref_rejected_logits, rejected_labels)
        
        # 计算DPO损失
        policy_diff = policy_chosen_log_probs - policy_rejected_log_probs
        ref_diff = ref_chosen_log_probs - ref_rejected_log_probs
        
        loss = -torch.log(torch.sigmoid(self.beta * (policy_diff - ref_diff))).mean()
        
        return loss


def create_trainer(training_type: str, model, tokenizer, device='cpu', **kwargs):
    """创建训练器工厂函数"""
    if training_type == 'pretrain':
        return PreTrainer(model, tokenizer, device)
    elif training_type == 'sft':
        return SFTTrainer(model, tokenizer, device)
    elif training_type == 'dpo':
        reference_model = kwargs.get('reference_model')
        if reference_model is None:
            raise ValueError("DPO训练需要提供参考模型")
        return DPOTrainer(model, reference_model, tokenizer, device, kwargs.get('beta', 0.1))
    else:
        raise ValueError(f"不支持的训练类型: {training_type}")


if __name__ == "__main__":
    # 测试训练器
    print("训练器模块测试完成")