#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medium模型测试脚本
"""
import os
import sys
import torch

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from model.transformer import MiniGPT
from tokenizer.bpe_tokenizer import BPETokenizer

def create_medium_model(vocab_size, device="cpu"):
    """创建与保存的medium模型匹配的配置"""
    config = {
        'vocab_size': vocab_size,
        'd_model': 640,        # medium配置
        'n_heads': 10,         # 10个注意力头
        'n_layers': 10,        # 10层
        'd_ff': 2560,          # 前馈网络维度
        'max_len': 1024,
        'dropout': 0.1
    }
    
    model = MiniGPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    )
    
    return model

def generate_text(model, tokenizer, prompt, max_length=40, temperature=0.8, device="cpu"):
    """生成文本序列"""
    # 编码输入
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        return prompt
    
    generated_ids = input_ids.copy()
    
    for _ in range(max_length):
        # 转换为tensor
        input_tensor = torch.tensor([generated_ids], device=device)
        
        with torch.no_grad():
            # 前向传播
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            
            # 确保在有效范围内
            actual_vocab_size = tokenizer.get_vocab_size()
            if next_token_logits.size(0) > actual_vocab_size:
                next_token_logits = next_token_logits[:actual_vocab_size]
            
            # 采样下一个token
            if temperature > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(next_token_logits).item()
            
            # 检查token有效性
            if next_token >= actual_vocab_size:
                next_token = next_token % actual_vocab_size
            
            # 添加到生成序列
            generated_ids.append(next_token)
    
    # 解码生成的文本
    try:
        return tokenizer.decode(generated_ids)
    except:
        return prompt + " [解码错误]"

def test_medium_model():
    """测试Medium模型"""
    print("=== MiniGPT Medium模型评测 ===\n")
    
    # 设置设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"使用设备: {device}")
    
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = BPETokenizer()
    tokenizer_path = os.path.join(project_root, "checkpoints/mac_medium/tokenizer.pkl")
    tokenizer.load(tokenizer_path)
    print(f"分词器词汇表大小: {tokenizer.get_vocab_size()}")
    
    # 使用模型配置的词汇表大小
    model_vocab_size = 8000  # from medium config
    print(f"使用模型词汇表大小: {model_vocab_size}")
    
    # 创建模型
    print("创建模型...")
    model = create_medium_model(model_vocab_size, device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载模型权重
    print("加载模型权重...")
    try:
        model_path = os.path.join(project_root, "checkpoints/mac_medium/final_model.pt")
        checkpoint = torch.load(model_path, 
                              map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型加载成功!")
        
        # 显示训练信息
        if 'step' in checkpoint:
            print(f"训练步数: {checkpoint['step']}")
        if 'loss' in checkpoint:
            print(f"最终损失: {checkpoint['loss']:.4f}")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # 生成测试
    print("\n=== 生成测试 ===")
    test_cases = [
        {"prompt": "你好", "max_length": 30, "temperature": 0.7},
        {"prompt": "请介绍一下人工智能", "max_length": 50, "temperature": 0.8},
        {"prompt": "什么是机器学习", "max_length": 40, "temperature": 0.6},
        {"prompt": "今天天气很好", "max_length": 35, "temperature": 0.9},
        {"prompt": "我想学习编程", "max_length": 40, "temperature": 0.7},
        {"prompt": "写一个Python程序", "max_length": 45, "temperature": 0.8},
        {"prompt": "解释深度学习的原理", "max_length": 50, "temperature": 0.6},
        {"prompt": "如何提高学习效率", "max_length": 40, "temperature": 0.75},
        {"prompt": "中国的首都是", "max_length": 25, "temperature": 0.5},
        {"prompt": "人工智能的未来发展", "max_length": 45, "temperature": 0.8},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case["prompt"]
        max_length = test_case["max_length"]
        temperature = test_case["temperature"]
        
        print(f"\n[测试 {i}]")
        print(f"输入: {prompt}")
        print(f"参数: max_length={max_length}, temperature={temperature}")
        
        try:
            generated_text = generate_text(
                model, tokenizer, prompt, 
                max_length=max_length, 
                temperature=temperature, 
                device=device
            )
            print(f"输出: {generated_text}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        print("-" * 80)
    
    # 模型性能总结
    print("\n=== Medium模型性能总结 ===")
    print("模型配置:")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,} (~{sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    print(f"  - 层数: 10层 Transformer")
    print(f"  - 维度: 640 (d_model)")
    print(f"  - 注意力头: 10个")
    print(f"  - 词汇表: {model_vocab_size}")
    print(f"  - 训练步数: 4000步")
    if 'loss' in checkpoint:
        print(f"  - 最终损失: {checkpoint['loss']:.4f}")
    
    # 与其他模型的对比
    print(f"\n模型规模对比:")
    print(f"  Tiny模型:   ~1.3M 参数")
    print(f"  Small模型:  ~29M 参数") 
    print(f"  Medium模型: ~59M 参数 (当前)")
    print(f"  Medium是Small的 {59/29:.1f}倍大小")
    
    # 交互式测试
    print("\n=== 交互式测试 ===")
    print("输入文本进行生成测试，输入 'quit' 退出")
    
    while True:
        try:
            user_input = input("\n> 请输入提示文本: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if not user_input:
                continue
            
            # 生成文本
            generated = generate_text(
                model, tokenizer, user_input, 
                max_length=40, temperature=0.8, device=device
            )
            print(f"生成结果: {generated}")
            
        except KeyboardInterrupt:
            print("\n测试结束")
            break
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    test_medium_model() 