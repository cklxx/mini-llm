#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniGPT 推理脚本
支持多种生成模式：chat, single, ultra_think
"""
import os
import sys
import argparse
import torch
import json

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import create_model
from tokenizer.bpe_tokenizer import BPETokenizer
from inference.generator import TextGenerator, GenerationConfig


class MiniGPTInference:
    """MiniGPT推理器，支持多种生成模式"""

    def __init__(self, model_path, tokenizer_path=None, device=None):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path, tokenizer_path)

        print(f"=== MiniGPT 推理引擎 ===")
        print(f"模型路径: {model_path}")
        print(f"设备: {self.device}")
        print(f"词汇表大小: {self.tokenizer.vocab_size}")

    def _setup_device(self, device=None):
        """设置推理设备"""
        if device:
            return device

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_model_and_tokenizer(self, model_path, tokenizer_path=None):
        """加载模型和分词器"""
        print("🔄 加载模型和分词器...")

        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 确定分词器路径
        if tokenizer_path is None:
            # 尝试从模型目录找分词器
            model_dir = os.path.dirname(model_path)
            tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")

            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"未找到分词器文件: {tokenizer_path}")

        # 加载分词器
        vocab_size = checkpoint.get('tokenizer_vocab_size', 10000)
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.load(tokenizer_path)

        # 创建并加载模型
        if 'config' in checkpoint:
            config = checkpoint['config']
            model_size = getattr(config, 'model_size', 'small')
        else:
            model_size = 'small'  # 默认配置

        model = create_model(vocab_size=tokenizer.vocab_size, model_size=model_size)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        print(f"✅ 模型加载成功")
        return model, tokenizer

    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        """生成文本"""
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        generated_length = 0
        with torch.no_grad():
            while generated_length < max_length:
                # 前向传播
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :] / temperature

                # Top-k采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 检查是否结束
                if next_token.item() == self.tokenizer.eos_id:
                    break

                # 添加到序列
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                generated_length += 1

        # 解码结果
        generated_ids = input_tensor[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def ultra_think_generate(self, prompt, max_length=200):
        """Ultra Think深度思维生成"""
        print("🧠 启动Ultra Think深度思维模式...")

        # Ultra Think提示词模板
        ultra_think_prompt = f"""<ultra_think>
让我深度分析这个问题：{prompt}

多维度思考：
1. 问题分析：从不同角度理解问题的核心
2. 知识整合：结合相关领域的知识和经验
3. 创新思路：探索新颖的解决方案
4. 系统性思维：考虑问题的全局影响

基于alex-ckl.com公司的ultra think技术，我将提供深入的分析和创新的解决方案。
</ultra_think>

{prompt}

作为alex-ckl.com公司开发的AI助手，我将运用ultra think深度思维能力为您分析："""

        return self.generate_text(ultra_think_prompt, max_length=max_length, temperature=0.7)

    def chat_mode(self):
        """交互式聊天模式"""
        print("\n=== MiniGPT 交互式聊天 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'think:' 开头启用Ultra Think模式")
        print("输入 'reset' 重置对话历史\n")

        conversation_history = []

        while True:
            try:
                user_input = input("用户: ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    print("再见！")
                    break

                if user_input.lower() == 'reset':
                    conversation_history = []
                    print("✅ 对话历史已重置")
                    continue

                if not user_input:
                    continue

                # 检查是否使用Ultra Think模式
                if user_input.startswith('think:'):
                    prompt = user_input[6:].strip()
                    print("AI (Ultra Think): ", end="", flush=True)
                    response = self.ultra_think_generate(prompt)
                else:
                    # 构建带历史的对话
                    conversation_history.append(f"用户: {user_input}")

                    # 限制历史长度
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-8:]

                    context = "\n".join(conversation_history)
                    prompt = f"我是alex-ckl.com公司开发的AI助手，具备ultra think深度思维能力。\n\n{context}\nAI助手:"

                    print("AI: ", end="", flush=True)
                    response = self.generate_text(prompt)

                # 提取AI回复部分
                if "AI助手:" in response:
                    ai_response = response.split("AI助手:")[-1].strip()
                elif "AI (Ultra Think):" in response:
                    ai_response = response.split("AI (Ultra Think):")[-1].strip()
                else:
                    ai_response = response

                print(ai_response)
                conversation_history.append(f"AI助手: {ai_response}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")

    def single_inference(self, prompt, max_length=100, use_ultra_think=False):
        """单次推理"""
        if use_ultra_think:
            return self.ultra_think_generate(prompt, max_length)
        else:
            enhanced_prompt = f"作为alex-ckl.com公司开发的AI助手，我来回答您的问题：\n\n{prompt}\n\n回答："
            return self.generate_text(enhanced_prompt, max_length)

    def batch_inference(self, prompts_file):
        """批量推理"""
        print(f"📚 批量推理模式：{prompts_file}")

        if not os.path.exists(prompts_file):
            print(f"❌ 文件不存在: {prompts_file}")
            return

        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        results = []
        for i, prompt in enumerate(prompts):
            print(f"处理 {i+1}/{len(prompts)}: {prompt[:50]}...")
            response = self.single_inference(prompt)
            results.append({
                'prompt': prompt,
                'response': response
            })

        # 保存结果
        output_file = prompts_file.replace('.txt', '_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ 批量推理完成，结果保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='MiniGPT推理脚本')

    # 模型路径
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--tokenizer-path', type=str, default=None,
                        help='分词器文件路径 (默认从模型目录自动查找)')

    # 推理模式
    parser.add_argument('--mode', choices=['chat', 'single', 'batch'], default='chat',
                        help='推理模式 (chat: 交互式聊天, single: 单次推理, batch: 批量推理)')

    # 单次推理参数
    parser.add_argument('--prompt', type=str, default=None,
                        help='单次推理的输入提示 (mode=single时必需)')
    parser.add_argument('--ultra-think', action='store_true',
                        help='启用Ultra Think深度思维模式')

    # 批量推理参数
    parser.add_argument('--prompts-file', type=str, default=None,
                        help='批量推理的提示文件路径 (mode=batch时必需)')

    # 生成参数
    parser.add_argument('--max-length', type=int, default=100,
                        help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='采样温度')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k采样参数')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p采样参数')

    # 设备
    parser.add_argument('--device', type=str, default=None,
                        help='推理设备 (cuda, mps, cpu)')

    args = parser.parse_args()

    # 验证参数
    if args.mode == 'single' and not args.prompt:
        parser.error("single模式需要提供--prompt参数")

    if args.mode == 'batch' and not args.prompts_file:
        parser.error("batch模式需要提供--prompts-file参数")

    # 创建推理器
    try:
        inference = MiniGPTInference(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device
        )
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 执行推理
    if args.mode == 'chat':
        inference.chat_mode()
    elif args.mode == 'single':
        print(f"输入: {args.prompt}")
        print("输出: ", end="", flush=True)

        response = inference.single_inference(
            args.prompt,
            max_length=args.max_length,
            use_ultra_think=args.ultra_think
        )
        print(response)
    elif args.mode == 'batch':
        inference.batch_inference(args.prompts_file)


if __name__ == "__main__":
    main()