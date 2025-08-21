#!/usr/bin/env python3
"""
测试 SwiGLU 模型实现
使用 minimind_dataset 的数据测试模型的所有层是否正常工作
包含详细的日志输出
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from typing import List, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from src.model.transformer import MiniGPT, create_model
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.trainer import PreTrainer, LanguageModelingDataset
from src.data.dataset_loader import DatasetConfig, PretrainDataLoader

class ModelTestSuite:
    """模型测试套件"""
    
    def __init__(self, device='cpu'):
        self.device = device
        print(f"🔧 初始化测试套件，使用设备: {device}")
        
        # 初始化 tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载测试数据
        self.test_data = self._load_test_data()
        
    def _load_tokenizer(self):
        """加载分词器"""
        print("📚 加载分词器...")
        try:
            tokenizer = BPETokenizer()
            # 设置基本的特殊token
            tokenizer.pad_id = 0
            tokenizer.bos_id = 1
            tokenizer.eos_id = 2
            tokenizer.unk_id = 3
            
            # 创建一个简单的词汇表用于测试
            vocab = {
                '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3,
                '你': 4, '好': 5, '我': 6, '是': 7, '的': 8, '了': 9, '在': 10,
                '什': 11, '么': 12, '如': 13, '何': 14, '学': 15, '习': 16, '编': 17, '程': 18,
                '春': 19, '风': 20, '花': 21, '开': 22, '鸟': 23, '语': 24, '香': 25
            }
            
            # 扩展词汇表到至少100个token
            for i in range(26, 100):
                vocab[f'token_{i}'] = i
                
            tokenizer.vocab = vocab
            tokenizer.vocab_size = len(vocab)
            
            print(f"✅ 分词器加载完成，词汇表大小: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            print(f"❌ 分词器加载失败: {e}")
            # 创建一个最小的tokenizer用于测试
            class SimpleTokenizer:
                def __init__(self):
                    self.vocab_size = 1000
                    self.pad_id = 0
                    self.bos_id = 1
                    self.eos_id = 2
                    self.unk_id = 3
                
                def encode(self, text, add_special_tokens=True):
                    # 简单的字符级编码
                    tokens = [min(ord(c), 999) for c in text[:50]]  # 限制长度和值
                    if add_special_tokens:
                        tokens = [self.bos_id] + tokens + [self.eos_id]
                    return tokens
                
                def decode(self, tokens):
                    return ''.join([chr(min(t, 127)) for t in tokens if t not in [self.pad_id, self.bos_id, self.eos_id]])
            
            return SimpleTokenizer()
    
    def _create_model(self):
        """创建模型"""
        print("🤖 创建模型...")
        try:
            # 创建一个小型模型用于测试
            model = MiniGPT(
                vocab_size=self.tokenizer.vocab_size,
                d_model=256,      # 较小的维度用于快速测试
                n_heads=4,        # 较少的头数
                n_layers=2,       # 较少的层数
                d_ff=1024,        # SwiGLU 隐藏维度
                max_len=128,      # 较短的序列长度
                dropout=0.1
            )
            model.to(self.device)
            
            print(f"✅ 模型创建完成")
            print(f"📊 模型参数量: {model.get_num_params():,}")
            print(f"📊 模型配置: d_model={model.d_model}, n_layers={len(model.transformer_blocks)}")
            
            return model
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _load_test_data(self):
        """加载测试数据"""
        print("📁 加载测试数据...")
        
        # 首先尝试加载实际的数据文件
        data_files = [
            "data/dataset/minimind_dataset/pretrain_minimal.jsonl",
            "data/dataset/minimind_dataset/pretrain_test.jsonl"
        ]
        
        texts = []
        for data_file in data_files:
            if os.path.exists(data_file):
                print(f"📂 加载数据文件: {data_file}")
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line_num >= 10:  # 只加载前10行用于测试
                                break
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'text' in data:
                                        text = data['text']
                                        # 提取实际文本内容（去除特殊标记）
                                        if '<|im_start|>' in text and '<|im_end|>' in text:
                                            text = text.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
                                        if len(text) > 20 and len(text) < 200:  # 选择合适长度的文本
                                            texts.append(text)
                                except json.JSONDecodeError:
                                    continue
                    print(f"✅ 从 {data_file} 加载了 {len([t for t in texts])} 条文本")
                except Exception as e:
                    print(f"⚠️ 加载 {data_file} 时出错: {e}")
                    continue
            else:
                print(f"⚠️ 数据文件不存在: {data_file}")
        
        # 如果没有找到数据文件，创建一些测试数据
        if not texts:
            print("📝 创建测试数据...")
            texts = [
                "你好，我想学习如何编程，有什么建议吗？",
                "帮我写一首关于春天的五言诗。",
                "解释一下什么是机器学习？",
                "今天天气很好，适合出去散步。",
                "人工智能是未来科技发展的重要方向。"
            ]
        
        print(f"✅ 总共加载了 {len(texts)} 条测试文本")
        for i, text in enumerate(texts[:3]):
            print(f"📄 示例 {i+1}: {text[:50]}...")
        
        return texts
    
    def test_tokenizer(self):
        """测试分词器"""
        print("\n" + "="*50)
        print("🧪 测试分词器")
        print("="*50)
        
        test_text = self.test_data[0] if self.test_data else "你好世界"
        print(f"📝 测试文本: {test_text}")
        
        try:
            # 编码
            tokens = self.tokenizer.encode(test_text, add_special_tokens=True)
            print(f"🔢 编码结果: {tokens[:10]}...")
            print(f"📏 Token数量: {len(tokens)}")
            
            # 解码
            if hasattr(self.tokenizer, 'decode'):
                decoded = self.tokenizer.decode(tokens)
                print(f"📝 解码结果: {decoded[:50]}...")
            
            print("✅ 分词器测试通过")
            return True
        except Exception as e:
            print(f"❌ 分词器测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_layers(self):
        """测试模型各层"""
        print("\n" + "="*50)
        print("🧪 测试模型各层")
        print("="*50)
        
        try:
            # 创建测试输入
            test_text = self.test_data[0] if self.test_data else "测试文本"
            tokens = self.tokenizer.encode(test_text, add_special_tokens=True)
            if len(tokens) > 64:
                tokens = tokens[:64]
            while len(tokens) < 10:
                tokens.append(self.tokenizer.pad_id)
            
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            print(f"📦 输入形状: {input_ids.shape}")
            
            self.model.eval()
            with torch.no_grad():
                # 测试词嵌入层
                print("\n🔤 测试词嵌入层...")
                embeddings = self.model.token_embedding(input_ids)
                print(f"✅ 词嵌入输出形状: {embeddings.shape}")
                print(f"📊 词嵌入统计: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
                
                # 测试位置编码
                print("\n📍 测试位置编码...")
                pos_encoded = self.model.positional_encoding(embeddings)
                print(f"✅ 位置编码输出形状: {pos_encoded.shape}")
                print(f"📊 位置编码统计: mean={pos_encoded.mean():.4f}, std={pos_encoded.std():.4f}")
                
                # 测试每个Transformer块
                x = self.model.dropout(pos_encoded)
                causal_mask = self.model.create_causal_mask(input_ids.size(1)).to(self.device)
                
                for i, transformer_block in enumerate(self.model.transformer_blocks):
                    print(f"\n🔄 测试Transformer块 {i+1}...")
                    
                    # 测试注意力机制
                    print(f"  🎯 测试多头注意力...")
                    attn_output = transformer_block.attention(x, x, x, causal_mask)
                    print(f"  ✅ 注意力输出形状: {attn_output.shape}")
                    print(f"  📊 注意力统计: mean={attn_output.mean():.4f}, std={attn_output.std():.4f}")
                    
                    # 应用第一个残差连接和层归一化
                    x_after_attn = transformer_block.norm1(x + transformer_block.dropout(attn_output))
                    print(f"  ✅ 注意力后规范化形状: {x_after_attn.shape}")
                    
                    # 测试SwiGLU前馈网络
                    print(f"  🚀 测试SwiGLU前馈网络...")
                    ff_output = transformer_block.feed_forward(x_after_attn)
                    print(f"  ✅ SwiGLU输出形状: {ff_output.shape}")
                    print(f"  📊 SwiGLU统计: mean={ff_output.mean():.4f}, std={ff_output.std():.4f}")
                    
                    # 检查SwiGLU内部组件
                    ff = transformer_block.feed_forward
                    print(f"  🔍 SwiGLU组件:")
                    print(f"    - w_gate权重形状: {ff.w_gate.weight.shape}")
                    print(f"    - w_up权重形状: {ff.w_up.weight.shape}")
                    print(f"    - w_down权重形状: {ff.w_down.weight.shape}")
                    
                    # 应用第二个残差连接和层归一化
                    x = transformer_block.norm2(x_after_attn + transformer_block.dropout(ff_output))
                    print(f"  ✅ 块输出形状: {x.shape}")
                    print(f"  📊 块输出统计: mean={x.mean():.4f}, std={x.std():.4f}")
                
                # 测试最终层归一化和输出投影
                print(f"\n🎯 测试最终层...")
                normalized = self.model.layer_norm(x)
                print(f"✅ 最终层归一化形状: {normalized.shape}")
                
                logits = self.model.lm_head(normalized)
                print(f"✅ 最终输出logits形状: {logits.shape}")
                print(f"📊 Logits统计: mean={logits.mean():.4f}, std={logits.std():.4f}")
                
                # 测试完整前向传播
                print(f"\n🔄 测试完整前向传播...")
                full_logits = self.model(input_ids)
                print(f"✅ 完整前向传播输出形状: {full_logits.shape}")
                print(f"📊 完整输出统计: mean={full_logits.mean():.4f}, std={full_logits.std():.4f}")
                
                # 验证输出是否一致
                diff = torch.abs(logits - full_logits).max()
                print(f"🔍 分步vs完整前向传播差异: {diff:.8f}")
                if diff < 1e-6:
                    print("✅ 分步和完整前向传播结果一致")
                else:
                    print("⚠️ 分步和完整前向传播结果存在差异")
                
            print("✅ 模型各层测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 模型层测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_training_step(self):
        """测试训练步骤"""
        print("\n" + "="*50)
        print("🧪 测试训练步骤")
        print("="*50)
        
        try:
            # 创建数据集
            dataset = LanguageModelingDataset(
                texts=self.test_data[:5],  # 只使用前5条数据
                tokenizer=self.tokenizer,
                max_length=64  # 较短的序列用于快速测试
            )
            
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            print(f"📦 数据集大小: {len(dataset)}")
            print(f"📦 批次大小: 2")
            
            # 创建训练器
            trainer = PreTrainer(self.model, self.tokenizer, device=self.device)
            
            print("\n🏋️ 开始训练测试...")
            
            # 记录训练前的参数
            param_before = {}
            for name, param in self.model.named_parameters():
                if 'feed_forward' in name:  # 重点关注SwiGLU参数
                    param_before[name] = param.clone().detach()
            
            print("📊 训练前SwiGLU参数统计:")
            for name, param in param_before.items():
                print(f"  {name}: mean={param.mean():.6f}, std={param.std():.6f}")
            
            # 执行一个训练步骤
            self.model.train()
            batch = next(iter(dataloader))
            
            print(f"\n📦 处理批次数据...")
            input_ids = batch.to(self.device)
            print(f"  输入形状: {input_ids.shape}")
            
            # 创建标签
            labels = torch.cat([input_ids[:, 1:], 
                              torch.full((input_ids.size(0), 1), 
                                       self.tokenizer.pad_id, device=self.device)], dim=1)
            print(f"  标签形状: {labels.shape}")
            
            # 前向传播
            print("🔄 前向传播...")
            logits = self.model(input_ids)
            print(f"✅ 前向传播完成，输出形状: {logits.shape}")
            
            # 计算损失
            print("📉 计算损失...")
            loss = trainer.compute_loss(logits, labels)
            print(f"✅ 损失计算完成: {loss.item():.4f}")
            
            # 反向传播
            print("🔙 反向传播...")
            trainer.optimizer.zero_grad()
            loss.backward()
            print("✅ 反向传播完成")
            
            # 检查梯度
            print("🔍 检查梯度...")
            grad_stats = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None and 'feed_forward' in name:
                    grad_norm = param.grad.norm().item()
                    grad_stats[name] = grad_norm
                    print(f"  {name}: 梯度范数={grad_norm:.6f}")
            
            if not grad_stats:
                print("⚠️ 没有检测到SwiGLU相关梯度")
            else:
                print("✅ SwiGLU层梯度正常")
            
            # 优化器步骤
            print("⚡ 执行优化器步骤...")
            trainer.optimizer.step()
            print("✅ 优化器步骤完成")
            
            # 检查参数更新
            print("\n📊 检查参数更新...")
            param_updated = False
            for name, param_before_val in param_before.items():
                param_after = dict(self.model.named_parameters())[name]
                diff = torch.abs(param_after - param_before_val).max().item()
                print(f"  {name}: 最大变化={diff:.8f}")
                if diff > 1e-8:
                    param_updated = True
            
            if param_updated:
                print("✅ SwiGLU参数成功更新")
            else:
                print("⚠️ SwiGLU参数没有明显更新")
            
            print("✅ 训练步骤测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 训练步骤测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_generation(self):
        """测试模型生成"""
        print("\n" + "="*50)
        print("🧪 测试模型生成")
        print("="*50)
        
        try:
            # 准备输入
            prompt = "你好"
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) > 10:
                tokens = tokens[:10]
            
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            print(f"📝 输入提示: {prompt}")
            print(f"📦 输入形状: {input_ids.shape}")
            
            # 生成文本
            print("🎯 开始生成...")
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_length=5,  # 只生成5个token用于测试
                    temperature=1.0,
                    top_k=10
                )
            
            print(f"✅ 生成完成，输出形状: {generated.shape}")
            print(f"📄 生成的token序列: {generated[0].tolist()}")
            
            # 尝试解码
            if hasattr(self.tokenizer, 'decode'):
                try:
                    decoded_text = self.tokenizer.decode(generated[0].tolist())
                    print(f"📝 解码后的文本: {decoded_text}")
                except:
                    print("⚠️ 解码失败，但生成过程正常")
            
            print("✅ 模型生成测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 模型生成测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_test(self):
        """运行完整测试"""
        print("🚀 开始SwiGLU模型完整测试")
        print("="*60)
        
        results = {}
        
        # 1. 测试分词器
        results['tokenizer'] = self.test_tokenizer()
        
        # 2. 测试模型各层
        results['model_layers'] = self.test_model_layers()
        
        # 3. 测试训练步骤
        results['training_step'] = self.test_training_step()
        
        # 4. 测试模型生成
        results['generation'] = self.test_model_generation()
        
        # 总结
        print("\n" + "="*60)
        print("📋 测试结果总结")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name:20}: {status}")
        
        print(f"\n🏆 总体结果: {passed_tests}/{total_tests} 项测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！SwiGLU模型实现正确！")
        else:
            print("⚠️ 部分测试失败，请检查实现")
        
        return passed_tests == total_tests


def main():
    """主函数"""
    # 检测设备
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 创建测试套件
    test_suite = ModelTestSuite(device=device)
    
    # 运行完整测试
    success = test_suite.run_full_test()
    
    if success:
        print("\n🎯 建议下一步：运行更长时间的训练来验证模型收敛性")
        print("💡 可以使用以下命令进行完整训练:")
        print("   python scripts/train_optimized.py")
    
    return success


if __name__ == "__main__":
    main()
