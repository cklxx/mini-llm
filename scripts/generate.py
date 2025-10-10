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


class MiniGPTInference:
    """MiniGPT推理器，支持多种生成模式"""

    def __init__(self, model_path, tokenizer_path=None, device=None, generation_kwargs=None):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path, tokenizer_path)

        defaults = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        if generation_kwargs:
            defaults.update({k: v for k, v in generation_kwargs.items() if v is not None})
        self.generation_defaults = defaults

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

        expected_config = checkpoint.get('tokenizer_config')
        if expected_config:
            actual_config = tokenizer.get_config()
            mismatches = {
                key: (expected_config[key], actual_config.get(key))
                for key in expected_config
                if actual_config.get(key) != expected_config[key]
            }
            if mismatches:
                mismatch_info = ", ".join(
                    f"{k}: ckpt={v[0]} vs tokenizer={v[1]}" for k, v in mismatches.items()
                )
                raise ValueError(
                    "分词器配置与checkpoint不一致，请确认推理端使用的tokenizer文件正确。"
                    f"差异: {mismatch_info}"
                )

        expected_special = checkpoint.get('tokenizer_special_tokens')
        if expected_special:
            special_mismatches = tokenizer.diff_special_tokens(expected_special)
            if special_mismatches:
                mismatch_info = ", ".join(
                    f"{name}: ckpt={exp} vs tokenizer={act}" for name, (exp, act) in special_mismatches.items()
                )
                raise ValueError(
                    "分词器特殊token映射与checkpoint不一致，请检查 tokenizer.pkl 是否匹配训练输出。"
                    f"差异: {mismatch_info}"
                )

        expected_checksum = checkpoint.get('tokenizer_checksum')
        if expected_checksum:
            actual_checksum = tokenizer.checksum()
            if actual_checksum and actual_checksum != expected_checksum:
                raise ValueError(
                    "分词器校验失败：checksum不匹配。请确保使用训练时导出的tokenizer.pkl。"
                )
        print("✅ 分词器配置校验通过")

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

    def generate_text(self, prompt, **overrides):
        """生成文本"""

        config = self.generation_defaults.copy()
        config.update({k: v for k, v in overrides.items() if v is not None})

        max_new_tokens = int(config.get("max_new_tokens", 0))
        temperature = float(config.get("temperature", 1.0))
        top_k = int(config.get("top_k", 0))
        top_p = float(config.get("top_p", 1.0))
        repetition_penalty = float(config.get("repetition_penalty", 1.0))

        if max_new_tokens <= 0:
            return ""

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_length = len(input_ids)
        generated_ids = list(input_ids)
        input_tensor = torch.tensor([generated_ids], device=self.device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :]

                adjusted_logits = next_token_logits.clone()
                if temperature > 0:
                    adjusted_logits = adjusted_logits / temperature

                if repetition_penalty and repetition_penalty != 1.0:
                    for token_id in set(generated_ids):
                        logit = adjusted_logits[token_id]
                        if logit < 0:
                            adjusted_logits[token_id] *= repetition_penalty
                        else:
                            adjusted_logits[token_id] /= repetition_penalty

                if top_k and top_k > 0:
                    values, _ = torch.topk(adjusted_logits, min(top_k, adjusted_logits.size(-1)))
                    min_values = values[..., -1, None]
                    adjusted_logits[adjusted_logits < min_values] = float('-inf')

                if top_p and 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(adjusted_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    adjusted_logits[indices_to_remove] = float('-inf')

                if temperature <= 0:
                    next_token_id = int(torch.argmax(adjusted_logits).item())
                else:
                    probs = torch.softmax(adjusted_logits, dim=-1)
                    next_token_id = int(torch.multinomial(probs, num_samples=1).item())

                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
                input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

                if next_token_id == self.tokenizer.eos_id:
                    break

        new_token_ids = generated_ids[prompt_length:]
        generated_text = self.tokenizer.decode(new_token_ids)
        return generated_text.strip()

    def ultra_think_generate(self, prompt, **generation_kwargs):
        """Ultra Think深度思维生成"""
        print("🧠 启动Ultra Think深度思维模式...")

        ultra_think_prompt = f"""<ultra_think>
我将逐步深入分析该问题：{prompt}

思考方向：
1. 核心问题与背景
2. 相关知识与事实
3. 潜在方案与利弊
4. 长期影响与延伸思考
</ultra_think>

请根据以上分析给出最终回答："""

        max_tokens = generation_kwargs.pop(
            "max_new_tokens",
            max(self.generation_defaults.get("max_new_tokens", 128), 200),
        )
        return self.generate_text(
            ultra_think_prompt,
            max_new_tokens=max_tokens,
            **generation_kwargs,
        )

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
                    prompt = f"{context}\n助手:"

                    print("助手: ", end="", flush=True)
                    response = self.generate_text(prompt)

                # 提取AI回复部分
                ai_response = response.strip()
                if ai_response.lower().startswith("助手:"):
                    ai_response = ai_response.split("助手:", 1)[-1].strip()

                print(ai_response)
                conversation_history.append(f"助手: {ai_response}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")

    def single_inference(self, prompt, use_ultra_think=False, **generation_kwargs):
        """单次推理"""
        if use_ultra_think:
            return self.ultra_think_generate(prompt, **generation_kwargs)
        return self.generate_text(prompt, **generation_kwargs)

    def batch_inference(self, prompts_file, **generation_kwargs):
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
            response = self.single_inference(prompt, **generation_kwargs)
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
    parser.add_argument('--max-new-tokens', '--max-length', type=int, default=128, dest='max_new_tokens',
                        help='最大生成token数 (包含别名 --max-length)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k采样参数')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p采样参数')
    parser.add_argument('--repetition-penalty', type=float, default=1.05,
                        help='重复惩罚系数 (>1 会抑制重复)')

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
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }
        inference = MiniGPTInference(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            generation_kwargs=generation_kwargs,
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
            use_ultra_think=args.ultra_think,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(response)
    elif args.mode == 'batch':
        inference.batch_inference(
            args.prompts_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()