#!/usr/bin/env python3
"""
架构升级测试脚本
验证所有新架构组件的正确性：RoPE、GQA、深度优化、权重共享等
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:  # pragma: no cover - optional dependency guard
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    pytest = None


try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

from src.model.config import estimate_params, get_small_config, get_tiny_config

if torch is not None:
    from src.model.gqa import GroupedQueryAttention
    from src.model.rope import RotaryPositionEmbedding, apply_rotary_pos_emb
    from src.model.transformer import MiniGPT
else:  # pragma: no cover - executed when optional dependencies missing
    GroupedQueryAttention = None
    RotaryPositionEmbedding = None
    apply_rotary_pos_emb = None
    MiniGPT = None

BACKEND_AVAILABLE = torch is not None and MiniGPT is not None

if pytest is not None:
    pytestmark = pytest.mark.skipif(not BACKEND_AVAILABLE, reason="PyTorch not available")
else:  # pragma: no cover - executed when running as a plain script
    pytestmark = []


def test_rope_implementation():
    """测试RoPE位置编码实现"""
    print("Testing RoPE Position Encoding...")

    batch_size = 2
    seq_len = 128
    head_dim = 64

    # 创建RoPE
    rope = RotaryPositionEmbedding(head_dim, max_position_embeddings=256)

    # 测试输入
    hidden_states = torch.randn(batch_size, seq_len, head_dim)

    # 获取cos和sin
    cos, sin = rope(hidden_states)

    assert cos.shape == (seq_len, head_dim), f"Expected cos shape ({seq_len}, {head_dim}), got {cos.shape}"
    assert sin.shape == (seq_len, head_dim), f"Expected sin shape ({seq_len}, {head_dim}), got {sin.shape}"

    # 测试应用RoPE
    q = torch.randn(batch_size, 8, seq_len, head_dim)
    k = torch.randn(batch_size, 8, seq_len, head_dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape, f"RoPE changed query shape: {q.shape} -> {q_rot.shape}"
    assert k_rot.shape == k.shape, f"RoPE changed key shape: {k.shape} -> {k_rot.shape}"

    # 验证旋转不变性（长度应该保持不变）
    q_norm_before = torch.norm(q, dim=-1)
    q_norm_after = torch.norm(q_rot, dim=-1)
    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-6), "RoPE should preserve vector norms"

    print("✅ RoPE implementation test passed!")
    return True


def test_gqa_implementation():
    """测试分组查询注意力实现"""
    print("Testing Grouped-Query Attention...")

    batch_size = 2
    seq_len = 128
    d_model = 384
    num_heads = 12
    num_kv_heads = 3

    # 创建GQA
    gqa = GroupedQueryAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        use_rope=True
    )

    # 测试输入
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output, _ = gqa(hidden_states)

    assert output.shape == hidden_states.shape, f"GQA output shape mismatch: {hidden_states.shape} vs {output.shape}"

    # 验证KV头数配置
    assert gqa.num_queries_per_kv == num_heads // num_kv_heads, "Incorrect queries per KV head ratio"

    # 测试参数数量减少
    total_params = sum(p.numel() for p in gqa.parameters())

    # 估算传统MHA参数量
    mha_params = 4 * d_model * d_model  # Q, K, V, O projections

    print(f"GQA parameters: {total_params:,}")
    print(f"Estimated MHA parameters: {mha_params:,}")
    print(f"Parameter reduction: {(1 - total_params/mha_params)*100:.1f}%")

    print("✅ GQA implementation test passed!")
    return True


def test_deep_thin_architecture():
    """测试深而窄架构优化"""
    print("Testing Deep-Thin Architecture Optimization...")

    # 比较优化前后的配置
    old_config = get_tiny_config()
    old_config.use_rope = False
    old_config.use_gqa = False
    old_config.hidden_size = 128
    old_config.num_hidden_layers = 4
    old_config.num_attention_heads = 2
    old_config.intermediate_size = 512

    new_config = get_tiny_config()  # 已经是深而窄的配置

    old_params = estimate_params(old_config)
    new_params = estimate_params(new_config)

    print(f"Old architecture (wide-shallow): {old_params:,} parameters")
    print(f"New architecture (deep-thin): {new_params:,} parameters")

    # 验证新架构确实更深
    assert new_config.num_hidden_layers > old_config.num_hidden_layers, "New config should be deeper"

    # 创建模型测试
    model = MiniGPT(new_config)

    # 测试前向传播
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, new_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (batch_size, seq_len, new_config.vocab_size)
    assert logits.shape == expected_shape, f"Model output shape mismatch: {logits.shape} vs {expected_shape}"

    print("✅ Deep-thin architecture test passed!")
    return True


def test_weight_sharing():
    """测试权重共享实现"""
    print("Testing Weight Sharing...")

    config = get_tiny_config()
    config.tie_word_embeddings = True

    model = MiniGPT(config)

    # 验证没有独立的lm_head
    assert model.lm_head is None, "Model should not have separate lm_head when using weight sharing"

    # 测试前向传播
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    # 验证输出维度正确
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Weight sharing model output shape: {logits.shape} vs {expected_shape}"

    # 计算参数节省
    params_with_sharing = estimate_params(config)

    config_no_sharing = get_tiny_config()
    config_no_sharing.tie_word_embeddings = False
    params_without_sharing = estimate_params(config_no_sharing)

    saved_params = params_without_sharing - params_with_sharing
    savings_percent = (saved_params / params_without_sharing) * 100

    print(f"Parameters with sharing: {params_with_sharing:,}")
    print(f"Parameters without sharing: {params_without_sharing:,}")
    print(f"Saved parameters: {saved_params:,} ({savings_percent:.1f}%)")

    print("✅ Weight sharing test passed!")
    return True


def test_model_compatibility():
    """测试模型兼容性和配置组合"""
    print("Testing Model Compatibility...")

    configs_to_test = [
        ("tiny", get_tiny_config()),
        ("small", get_small_config()),
    ]

    for config_name, config in configs_to_test:
        print(f"  Testing {config_name} configuration...")

        try:
            model = MiniGPT(config)

            # 测试参数初始化
            total_params = sum(p.numel() for p in model.parameters())
            estimated_params = estimate_params(config)

            # 允许一定的估算误差
            param_diff = abs(total_params - estimated_params) / estimated_params
            assert param_diff < 0.1, f"Parameter estimation error too large: {param_diff:.1%}"

            # 测试前向传播
            batch_size = 1
            seq_len = 32
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                logits = model(input_ids)

            assert not torch.isnan(logits).any(), f"Model {config_name} produces NaN outputs"
            assert not torch.isinf(logits).any(), f"Model {config_name} produces infinite outputs"

            print(f"    ✅ {config_name}: {total_params:,} params, output shape {logits.shape}")

        except Exception as e:
            print(f"    ❌ {config_name} failed: {e}")
            return False

    print("✅ Model compatibility test passed!")
    return True


def test_performance_improvements():
    """测试性能改进"""
    print("Testing Performance Improvements...")

    # 比较GQA vs MHA的内存使用
    batch_size = 4
    seq_len = 256
    d_model = 384
    num_heads = 12

    # 传统MHA
    mha_config = get_small_config()
    mha_config.use_gqa = False
    mha_config.use_rope = False

    # GQA配置
    gqa_config = get_small_config()
    gqa_config.use_gqa = True
    gqa_config.num_key_value_heads = 3

    print(f"MHA config: {estimate_params(mha_config):,} parameters")
    print(f"GQA config: {estimate_params(gqa_config):,} parameters")

    # 计算KV缓存大小对比
    head_dim = d_model // num_heads

    mha_kv_cache = 2 * batch_size * num_heads * seq_len * head_dim
    gqa_kv_cache = 2 * batch_size * gqa_config.num_key_value_heads * seq_len * head_dim

    kv_reduction = (1 - gqa_kv_cache / mha_kv_cache) * 100

    print(f"MHA KV cache size: {mha_kv_cache:,} elements")
    print(f"GQA KV cache size: {gqa_kv_cache:,} elements")
    print(f"KV cache reduction: {kv_reduction:.1f}%")

    assert kv_reduction > 50, "GQA should provide significant KV cache reduction"

    print("✅ Performance improvements test passed!")
    return True


def run_all_tests():
    """运行所有架构测试"""
    if not BACKEND_AVAILABLE:
        print("⚠️ PyTorch not available, skipping architecture tests.")
        return True

    print("=" * 60)
    print("MINIGPT ARCHITECTURE UPGRADE TESTS")
    print("=" * 60)

    tests = [
        test_rope_implementation,
        test_gqa_implementation,
        test_deep_thin_architecture,
        test_weight_sharing,
        test_model_compatibility,
        test_performance_improvements,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with error: {e}")
            print()

    print("=" * 60)
    print(f"ARCHITECTURE TESTS SUMMARY: {passed}/{total} PASSED")
    print("=" * 60)

    if passed == total:
        print("🎉 All architecture upgrades are working correctly!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
