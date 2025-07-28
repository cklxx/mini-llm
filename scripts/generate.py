#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本
用于测试训练好的模型
"""
import os
import sys
import argparse
import torch

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import create_model
from tokenizer.bpe_tokenizer import BPETokenizer
from inference.generator import TextGenerator, GenerationConfig, ChatBot


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, device: str):
    """加载模型和分词器"""
    # 加载分词器
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # 创建模型
    model = create_model(vocab_size=tokenizer.get_vocab_size(), model_size="small")
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def interactive_chat(generator: TextGenerator):
    """交互式聊天"""
    print("\\n=== MiniGPT 交互式聊天 ===")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'reset' 重置对话历史")
    print("输入 'config' 修改生成参数\\n")
    
    # 默认生成配置
    config = GenerationConfig(
        max_length=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            
            if user_input.lower() == 'reset':
                conversation_history = []
                print("对话历史已重置")
                continue
            
            if user_input.lower() == 'config':
                print("\\n当前配置:")
                print(f"max_length: {config.max_length}")
                print(f"temperature: {config.temperature}")
                print(f"top_k: {config.top_k}")
                print(f"top_p: {config.top_p}")
                print("输入新值或按回车保持不变:")
                
                try:
                    new_max_length = input(f"max_length ({config.max_length}): ").strip()
                    if new_max_length:
                        config.max_length = int(new_max_length)
                    
                    new_temperature = input(f"temperature ({config.temperature}): ").strip()
                    if new_temperature:
                        config.temperature = float(new_temperature)
                    
                    new_top_k = input(f"top_k ({config.top_k}): ").strip()
                    if new_top_k:
                        config.top_k = int(new_top_k)
                    
                    new_top_p = input(f"top_p ({config.top_p}): ").strip()
                    if new_top_p:
                        config.top_p = float(new_top_p)
                    
                    print("配置已更新\\n")
                except ValueError:
                    print("输入格式错误，配置未更改\\n")
                continue
            
            if not user_input:
                continue
            
            # 构建对话上下文
            if conversation_history:
                context = "\\n".join(conversation_history + [f"用户: {user_input}", "助手: "])
            else:
                context = f"用户: {user_input}\\n助手: "
            
            # 生成回复
            print("助手: ", end="", flush=True)
            response = generator.generate(context, config)
            
            # 提取助手回复部分
            if "助手: " in response:
                assistant_response = response.split("助手: ")[-1].strip()
            else:
                assistant_response = response.strip()
            
            print(assistant_response)
            
            # 更新对话历史
            conversation_history.append(f"用户: {user_input}")
            conversation_history.append(f"助手: {assistant_response}")
            
            # 限制历史长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
        except KeyboardInterrupt:
            print("\\n\\n再见！")
            break
        except Exception as e:
            print(f"\\n生成过程中出错: {e}")
            continue


def batch_inference(generator: TextGenerator, test_prompts: list, config: GenerationConfig):
    """批量推理测试"""
    print("\\n=== 批量推理测试 ===\\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[测试 {i}]")
        print(f"输入: {prompt}")
        
        try:
            response = generator.generate(prompt, config)
            print(f"输出: {response}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="MiniGPT推理脚本")
    parser.add_argument("--model-path", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                       help="分词器路径")
    parser.add_argument("--mode", type=str, choices=["chat", "batch", "single"],
                       default="chat", help="推理模式")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下自己。",
                       help="单次推理的提示文本")
    parser.add_argument("--max-length", type=int, default=100,
                       help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="采样温度")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k采样参数")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p采样参数")
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    print("加载模型和分词器...")
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, args.tokenizer_path, device
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer, device)
    
    # 生成配置
    config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True
    )
    
    if args.mode == "chat":
        # 交互式聊天
        interactive_chat(generator)
        
    elif args.mode == "single":
        # 单次生成
        print(f"\\n输入: {args.prompt}")
        try:
            response = generator.generate(args.prompt, config)
            print(f"输出: {response}")
        except Exception as e:
            print(f"生成失败: {e}")
    
    elif args.mode == "batch":
        # 批量测试
        test_prompts = [
            "请介绍一下人工智能的发展历史。",
            "你好，请告诉我今天的天气怎么样？",
            "请解释一下什么是机器学习？",
            "写一个简单的Python函数来计算斐波那契数列。",
            "请推荐几本值得阅读的书籍。"
        ]
        
        batch_inference(generator, test_prompts, config)


if __name__ == "__main__":
    main()