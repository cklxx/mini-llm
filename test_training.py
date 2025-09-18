#!/usr/bin/env python3
"""
完整训练测试脚本
验证GPU优化配置和训练流程
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm

from config.training_config import get_config
from src.model.transformer import create_model


def create_dummy_data(config, num_samples=1000):
    """创建虚拟训练数据"""
    print("创建虚拟训练数据...")

    # 生成随机token序列
    input_ids = torch.randint(0, config.vocab_size, (num_samples, config.max_seq_len))

    # 标签是输入向右偏移一位
    labels = torch.cat([input_ids[:, 1:], torch.zeros(num_samples, 1, dtype=torch.long)], dim=1)

    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    print(f"数据集大小: {num_samples} 样本")
    print(f"批量大小: {config.batch_size}")
    print(f"批次数量: {len(dataloader)}")

    return dataloader


def test_training_step(model, dataloader, config):
    """测试训练步骤"""
    print("\n=== 开始训练测试 ===")

    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 训练模式
    model.train()

    # 记录时间和损失
    start_time = time.time()
    total_loss = 0
    num_batches = min(10, len(dataloader))  # 只训练10个批次用于测试

    print(f"训练 {num_batches} 个批次...")

    for i, (input_ids, labels) in enumerate(tqdm(dataloader, total=num_batches)):
        if i >= num_batches:
            break

        # 移动数据到设备
        if config.device != "cpu":
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

        # 前向传播
        optimizer.zero_grad()

        outputs = model(input_ids)

        # 计算损失
        # 重塑输出和标签用于损失计算
        vocab_size = outputs.size(-1)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = criterion(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 优化器步骤
        optimizer.step()

        total_loss += loss.item()

        # 显存监控
        if config.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"批次 {i+1}: 损失={loss.item():.4f}, 显存={allocated:.1f}GB/{cached:.1f}GB")
        elif config.device == "mps":
            print(f"批次 {i+1}: 损失={loss.item():.4f}")
        else:
            print(f"批次 {i+1}: 损失={loss.item():.4f}")

    end_time = time.time()
    avg_loss = total_loss / num_batches
    training_time = end_time - start_time

    print(f"\n=== 训练测试完成 ===")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"每批次时间: {training_time/num_batches:.2f} 秒")

    return avg_loss


def test_inference(model, config):
    """测试推理"""
    print("\n=== 开始推理测试 ===")

    model.eval()

    # 创建测试输入
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    if config.device != "cpu":
        prompt = prompt.to(config.device)

    start_time = time.time()

    with torch.no_grad():
        # 测试生成
        generated = model.generate(
            prompt,
            max_length=50,
            temperature=config.temperature,
            top_k=config.top_k
        )

    end_time = time.time()
    inference_time = end_time - start_time

    print(f"推理时间: {inference_time:.3f} 秒")
    print(f"生成长度: {generated.size(1)} tokens")
    print(f"生成速度: {generated.size(1)/inference_time:.1f} tokens/秒")

    return inference_time


def main():
    """主测试函数"""
    print("🚀 开始完整的训练和推理测试")

    # 获取配置
    config = get_config("tiny")  # 使用tiny模型快速测试

    # 创建模型
    print(f"\n=== 创建模型 ===")

    # 使用配置创建模型配置对象
    from src.model.config import MiniGPTConfig
    model_config = MiniGPTConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.d_model,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        intermediate_size=config.d_ff,
        max_position_embeddings=config.max_seq_len,
        dropout=config.dropout,
        rms_norm_eps=1e-6
    )

    model = create_model(config=model_config)

    # 移动模型到设备
    if config.device != "cpu":
        model = model.to(config.device)
        print(f"模型已移动到 {config.device}")

    # 可选：编译模型 (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("模型编译成功")
        except Exception as e:
            print(f"模型编译失败: {e}")

    # 创建训练数据
    dataloader = create_dummy_data(config, num_samples=500)

    # 训练测试
    training_loss = test_training_step(model, dataloader, config)

    # 推理测试
    inference_time = test_inference(model, config)

    # 总结
    print(f"\n🎉 测试完成！")
    print(f"✓ 设备: {config.device}")
    print(f"✓ 模型参数: {model.get_num_params():,}")
    print(f"✓ 训练损失: {training_loss:.4f}")
    print(f"✓ 推理时间: {inference_time:.3f}s")

    # 内存使用情况
    if config.device == "cuda":
        print(f"✓ GPU显存使用: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    elif config.device == "mps":
        print(f"✓ MPS已启用")

    print("\n所有测试通过！训练和推理系统工作正常。")


if __name__ == "__main__":
    main()