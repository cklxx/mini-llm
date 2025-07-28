"""
文本生成和推理模块
支持多种生成策略：贪心搜索、采样、beam search等
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """生成配置"""
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True


class TextGenerator:
    """文本生成器
    
    支持多种解码策略：
    1. 贪心搜索 (Greedy Search)
    2. 随机采样 (Random Sampling)
    3. Top-k采样
    4. Top-p采样 (Nucleus Sampling)
    5. Beam Search
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def apply_repetition_penalty(self, logits: torch.Tensor, 
                                input_ids: torch.Tensor, 
                                penalty: float = 1.1) -> torch.Tensor:
        """应用重复惩罚"""
        if penalty == 1.0:
            return logits
        
        # 获取已生成的token
        unique_tokens = input_ids.unique()
        
        # 对已出现的token应用惩罚
        for token in unique_tokens:
            if logits[0, token] > 0:
                logits[0, token] = logits[0, token] / penalty
            else:
                logits[0, token] = logits[0, token] * penalty
        
        return logits
    
    def top_k_filtering(self, logits: torch.Tensor, top_k: int = 50) -> torch.Tensor:
        """Top-k过滤"""
        if top_k <= 0:
            return logits
        
        # 获取top-k的值和索引
        top_k = min(top_k, logits.size(-1))
        top_k_scores, top_k_indices = torch.topk(logits, top_k)
        
        # 创建掩码
        mask = torch.full_like(logits, -float('inf'))
        mask.scatter_(1, top_k_indices, top_k_scores)
        
        return mask
    
    def top_p_filtering(self, logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        """Top-p (nucleus) 过滤"""
        if top_p >= 1.0:
            return logits
        
        # 按概率排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 保留第一个超过阈值的token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 创建掩码
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('inf'))
        
        return logits
    
    def greedy_search(self, input_ids: torch.Tensor, 
                     max_length: int = 100) -> torch.Tensor:
        """贪心搜索"""
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                outputs = self.model(input_ids)
                
                # 获取下一个token的logits
                next_token_logits = outputs[:, -1, :]
                
                # 选择概率最大的token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 拼接到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # 检查是否生成了结束符
                if next_token.item() == self.tokenizer.eos_id:
                    break
        
        return input_ids
    
    def sample_generate(self, input_ids: torch.Tensor, 
                       config: GenerationConfig) -> torch.Tensor:
        """采样生成"""
        with torch.no_grad():
            for step in range(config.max_length):
                # 前向传播
                outputs = self.model(input_ids)
                
                # 获取下一个token的logits
                next_token_logits = outputs[:, -1, :] / config.temperature
                
                # 应用重复惩罚
                if config.repetition_penalty != 1.0:
                    next_token_logits = self.apply_repetition_penalty(
                        next_token_logits, input_ids, config.repetition_penalty
                    )
                
                # 应用top-k过滤
                if config.top_k > 0:
                    next_token_logits = self.top_k_filtering(next_token_logits, config.top_k)
                
                # 应用top-p过滤
                if config.top_p < 1.0:
                    next_token_logits = self.top_p_filtering(next_token_logits, config.top_p)
                
                # 计算概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                if config.do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # 拼接到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # 检查是否生成了结束符
                if next_token.item() == self.tokenizer.eos_id:
                    break
        
        return input_ids
    
    def beam_search(self, input_ids: torch.Tensor, 
                   config: GenerationConfig) -> torch.Tensor:
        """束搜索"""
        batch_size = input_ids.size(0)
        vocab_size = self.model.vocab_size
        
        # 初始化beam
        beam_size = config.num_beams
        beams = []
        
        # 扩展输入以支持多个beam
        expanded_input_ids = input_ids.repeat(beam_size, 1)
        beam_scores = torch.zeros(beam_size, device=self.device)
        
        with torch.no_grad():
            for step in range(config.max_length):
                # 前向传播
                outputs = self.model(expanded_input_ids)
                next_token_logits = outputs[:, -1, :]
                
                # 计算得分
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                
                if step == 0:
                    # 第一步只使用第一个beam
                    next_token_scores = next_token_scores[0:1, :]
                    beam_scores = beam_scores[0:1]
                    expanded_input_ids = expanded_input_ids[0:1, :]
                
                # 计算总得分
                scores = beam_scores.unsqueeze(1) + next_token_scores
                
                # 展平得分
                scores = scores.reshape(-1)
                
                # 获取top-k得分和索引
                top_scores, top_indices = torch.topk(scores, beam_size)
                
                # 计算beam索引和token索引
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # 更新beam
                new_beam_scores = top_scores
                new_beam_input_ids = []
                
                for i, (beam_idx, token_idx) in enumerate(zip(beam_indices, token_indices)):
                    new_sequence = torch.cat([
                        expanded_input_ids[beam_idx],
                        token_idx.unsqueeze(0)
                    ])
                    new_beam_input_ids.append(new_sequence.unsqueeze(0))
                
                expanded_input_ids = torch.cat(new_beam_input_ids, dim=0)
                beam_scores = new_beam_scores
                
                # 检查是否所有beam都生成了结束符
                if config.early_stopping:
                    finished_beams = (expanded_input_ids[:, -1] == self.tokenizer.eos_id).all()
                    if finished_beams:
                        break
        
        # 返回得分最高的beam
        best_beam_idx = torch.argmax(beam_scores)
        return expanded_input_ids[best_beam_idx].unsqueeze(0)
    
    def generate(self, input_text: str, config: GenerationConfig) -> str:
        """生成文本的主接口"""
        # 编码输入文本
        input_ids = torch.tensor([self.tokenizer.encode(input_text, add_special_tokens=True)], 
                                device=self.device)
        
        # 根据配置选择生成策略
        if config.num_beams > 1:
            output_ids = self.beam_search(input_ids, config)
        elif config.do_sample:
            output_ids = self.sample_generate(input_ids, config)
        else:
            output_ids = self.greedy_search(input_ids, config.max_length)
        
        # 解码输出
        output_text = self.tokenizer.decode(output_ids[0].cpu().tolist())
        
        return output_text
    
    def chat(self, message: str, history: List[str] = None, 
             config: GenerationConfig = None) -> str:
        """对话接口"""
        if config is None:
            config = GenerationConfig()
        
        # 构建对话上下文
        if history:
            context = "\\n".join(history + [f"用户: {message}", "助手: "])
        else:
            context = f"用户: {message}\\n助手: "
        
        # 生成回复
        response = self.generate(context, config)
        
        # 提取助手的回复部分
        if "助手: " in response:
            assistant_response = response.split("助手: ")[-1].strip()
        else:
            assistant_response = response.strip()
        
        return assistant_response


class ChatBot:
    """聊天机器人类"""
    
    def __init__(self, model_path: str, tokenizer_path: str, device='cpu'):
        # 加载模型和分词器
        self.device = device
        self.load_model(model_path)
        self.load_tokenizer(tokenizer_path)
        
        # 创建生成器
        self.generator = TextGenerator(self.model, self.tokenizer, device)
        
        # 对话历史
        self.conversation_history = []
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 这里需要根据实际的模型保存格式调整
        # self.model = create_model(vocab_size=...)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        pass
    
    def load_tokenizer(self, tokenizer_path: str):
        """加载分词器"""
        # self.tokenizer = BPETokenizer()
        # self.tokenizer.load(tokenizer_path)
        pass
    
    def chat(self, message: str, use_history: bool = True) -> str:
        """对话"""
        config = GenerationConfig(
            max_length=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        history = self.conversation_history if use_history else None
        response = self.generator.chat(message, history, config)
        
        # 更新对话历史
        if use_history:
            self.conversation_history.append(f"用户: {message}")
            self.conversation_history.append(f"助手: {response}")
            
            # 限制历史长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def reset_history(self):
        """重置对话历史"""
        self.conversation_history = []


if __name__ == "__main__":
    # 测试生成配置
    config = GenerationConfig(
        max_length=50,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        do_sample=True
    )
    
    print(f"生成配置: {config}")
    print("推理模块测试完成")