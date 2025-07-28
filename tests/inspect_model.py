#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型checkpoint的结构
"""
import torch
import os

def inspect_checkpoint(checkpoint_path):
    """检查checkpoint的结构"""
    print(f"=== 检查模型: {checkpoint_path} ===\n")
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ 成功加载checkpoint")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 检查checkpoint结构
    print("\n=== Checkpoint 结构 ===")
    if isinstance(checkpoint, dict):
        print("Checkpoint 键值:")
        for key in checkpoint.keys():
            print(f"  - {key}")
            
        # 如果有config信息
        if 'config' in checkpoint:
            print(f"\n=== 配置信息 ===")
            config = checkpoint['config']
            print(f"配置类型: {type(config)}")
            if hasattr(config, '__dict__'):
                for attr, value in config.__dict__.items():
                    print(f"  {attr}: {value}")
        
        # 检查模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        print(f"\n=== 模型参数 ===")
        print(f"参数总数: {len(state_dict)}")
        
        # 分析参数形状
        print("\n主要参数形状:")
        for name, param in state_dict.items():
            if hasattr(param, 'shape'):
                print(f"  {name}: {param.shape}")
            if len(list(state_dict.keys())) > 20 and name.count('.') > 2:
                # 如果参数太多，只显示主要的
                continue
                
        # 尝试推断模型配置
        print(f"\n=== 推断的模型配置 ===")
        try:
            # 从参数形状推断配置
            if 'token_embedding.weight' in state_dict:
                vocab_size, d_model = state_dict['token_embedding.weight'].shape
                print(f"词汇表大小: {vocab_size}")
                print(f"模型维度: {d_model}")
                
            # 计算层数
            layer_count = 0
            for name in state_dict.keys():
                if 'transformer_blocks.' in name:
                    layer_num = int(name.split('.')[1])
                    layer_count = max(layer_count, layer_num + 1)
            print(f"Transformer层数: {layer_count}")
            
            # 检查注意力头数
            if 'transformer_blocks.0.attention.w_q.weight' in state_dict:
                q_weight = state_dict['transformer_blocks.0.attention.w_q.weight']
                print(f"注意力权重形状: {q_weight.shape}")
                
        except Exception as e:
            print(f"推断配置时出错: {e}")
            
    else:
        print(f"Checkpoint 类型: {type(checkpoint)}")

def main():
    """主函数"""
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    checkpoints_to_inspect = [
        os.path.join(project_root, "checkpoints/mac_small/final_model.pt"),
        os.path.join(project_root, "checkpoints/mac_tiny/final_model.pt")
    ]
    
    for checkpoint_path in checkpoints_to_inspect:
        if os.path.exists(checkpoint_path):
            inspect_checkpoint(checkpoint_path)
            print("\n" + "="*80 + "\n")
        else:
            print(f"文件不存在: {checkpoint_path}")

if __name__ == "__main__":
    main() 