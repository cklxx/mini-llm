#!/usr/bin/env python3
"""One-click debugging utility for inspecting the MiniGPT training stack."""

import argparse
import os
import sys
from typing import Any, Iterable

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from config.training_config import get_config
from training.pipeline.pipeline import TrainingPipeline
from training.pipeline.training_loop import TrainingLoopRunner


def _describe_tensor(name: str, tensor: torch.Tensor, limit: int = 8) -> None:
    shape = tuple(tensor.shape)
    device = tensor.device
    dtype = tensor.dtype
    flat = tensor.detach().reshape(-1)
    if flat.numel() > 0:
        if tensor.is_floating_point():
            preview = ", ".join(f"{v:.4f}" for v in flat[:limit].tolist())
        else:
            preview = ", ".join(str(int(v)) for v in flat[:limit].tolist())
    else:
        preview = "<empty>"
    print(f"  {name}: shape={shape}, dtype={dtype}, device={device}")
    print(f"    sample=[{preview}]")


def _describe_padding(name: str, tensor: torch.Tensor, pad_id: int) -> None:
    if tensor.dim() < 2:
        return
    pad_mask = tensor.eq(pad_id)
    if not pad_mask.any():
        return
    per_row = pad_mask.sum(dim=-1)
    avg = per_row.float().mean().item()
    min_pad = int(per_row.min().item())
    max_pad = int(per_row.max().item())
    print(
        f"    padding[{name}]: min={min_pad}, max={max_pad}, avg={avg:.2f} tokens (pad_id={pad_id})"
    )


def _extract_raw_example(dataset: Any) -> str:
    if hasattr(dataset, "texts") and dataset.texts:
        return str(dataset.texts[0])
    if hasattr(dataset, "conversations") and dataset.conversations:
        conv = dataset.conversations[0]
        if isinstance(conv, list):
            return "\n".join(
                f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}" for turn in conv
            )
        if isinstance(conv, dict):
            user = conv.get("input") or conv.get("prompt") or ""
            reply = conv.get("output") or conv.get("response") or ""
            return f"user: {user}\nassistant: {reply}".strip()
        return str(conv)
    return "(示例) 你好，MiniGPT 全栈调试。"


def _encode_text(tokenizer: Any, text: str, max_length: int) -> torch.Tensor:
    if callable(tokenizer):
        try:
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        except TypeError:
            encoding = None
        else:
            input_ids = getattr(encoding, "input_ids", None)
            if input_ids is None and isinstance(encoding, dict):
                input_ids = encoding.get("input_ids")
            if input_ids is not None:
                tensor = torch.as_tensor(input_ids, dtype=torch.long)
                return tensor.squeeze(0)
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:max_length]
        pad_id = getattr(tokenizer, "pad_id", 0)
        if len(tokens) < max_length:
            tokens.extend([pad_id] * (max_length - len(tokens)))
        return torch.tensor(tokens, dtype=torch.long)
    raise TypeError("Unsupported tokenizer interface")


def _decode_tokens(tokenizer: Any, token_ids: Iterable[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(list(token_ids))
        except TypeError:
            pass
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids), skip_special_tokens=False)
        return "".join(tokens)
    return " ".join(str(t) for t in token_ids)


def run_fullstack_debug(mode: str, model_size: str, prompt: str, max_new_tokens: int) -> None:
    config = get_config(model_size)
    config.max_steps = 1
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.eval_steps = 1
    config.save_steps = 1
    config.enable_tensorboard = False
    config.validation_split = 0.0
    config.validation_min_samples = 0
    config.num_workers = 0
    config.persistent_workers = False
    config.prefetch_factor = 2
    config.use_high_performance_data_loading = False
    config.benchmark_eval_enabled = False
    config.dataset_global_sample_ratio = min(1.0, getattr(config, "dataset_global_sample_ratio", 1.0))

    print("=== MiniGPT 全栈调试 ===")
    print(f"模式: {mode}, 模型规模: {model_size}")

    pipeline = TrainingPipeline(config, mode=mode)
    tokenizer = pipeline.setup_tokenizer()
    pipeline.latest_tokenizer = tokenizer

    train_loader, _ = pipeline.setup_data_loader(tokenizer)
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("未在数据集中找到样本，请检查数据配置。")

    raw_text = _extract_raw_example(dataset)
    print("\n📄 原始样本预览:")
    print(raw_text[:400] + ("..." if len(raw_text) > 400 else ""))

    batch_iter = iter(train_loader)
    try:
        batch = next(batch_iter)
    except StopIteration as exc:
        raise RuntimeError("数据加载器未返回样本，无法调试。") from exc

    device = pipeline.device
    pad_id = getattr(tokenizer, "pad_id", None)
    bos_id = getattr(tokenizer, "bos_id", None)
    eos_id = getattr(tokenizer, "eos_id", None)
    print("\n🔤 分词器特殊 token:")
    print(f"  pad_id={pad_id}, bos_id={bos_id}, eos_id={eos_id}")

    if isinstance(batch, dict):
        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)
            attention = batch.get("attention_mask")
            if attention is not None:
                attention = attention.to(device)
            print("\n🧮 Batch 结构 (dict)")
            _describe_tensor("input_ids", input_ids)
            if pad_id is not None:
                _describe_padding("input_ids", input_ids, pad_id)
            if labels is not None:
                _describe_tensor("labels", labels)
            if attention is not None:
                _describe_tensor("attention_mask", attention)
        elif "chosen_input_ids" in batch:
            input_ids = batch["chosen_input_ids"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            print("\n🧮 Batch 结构 (DPO)")
            _describe_tensor("chosen_input_ids", input_ids)
            _describe_tensor("rejected_input_ids", rejected_ids)
            labels = None
            attention = None
        else:
            raise ValueError("无法识别的批次格式")
    else:
        inputs, targets, loss_mask = [tensor.to(device) for tensor in batch]
        print("\n🧮 Batch 结构 (tuple)")
        _describe_tensor("inputs", inputs)
        if pad_id is not None:
            _describe_padding("inputs", inputs, pad_id)
        _describe_tensor("targets", targets)
        _describe_tensor("loss_mask", loss_mask)
        input_ids = inputs
        labels = targets
        attention = None

    model = pipeline._build_model(tokenizer)
    optimizer = pipeline._create_optimizer(model)
    scheduler = pipeline._build_scheduler(optimizer)
    criterion = pipeline._create_criterion(tokenizer)

    print("\n🔍 前向传播检查")
    model.eval()
    with torch.no_grad():
        preview_inputs = input_ids if input_ids.dim() == 3 else input_ids[:1]
        if preview_inputs.dim() == 2:
            preview_inputs = preview_inputs[:1]
        token_embeds = model.token_embedding(preview_inputs)
        _describe_tensor("token_embedding", token_embeds)

        debug_snapshots: list[tuple[str, torch.Tensor]] = []
        call_counters: dict[str, int] = {}
        hook_handles: list[Any] = []

        def register(name: str, module: torch.nn.Module | None) -> None:
            if module is None:
                return

            def _hook(_mod, _inputs, output):
                primary = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(primary, torch.Tensor):
                    return
                slot = call_counters.get(name, 0)
                call_counters[name] = slot + 1
                label = name if slot == 0 else f"{name}#{slot}"
                debug_snapshots.append((label, primary.detach().cpu()))

            hook_handles.append(module.register_forward_hook(_hook))

        register("embedding_dropout", getattr(model, "dropout", None))
        if getattr(model, "transformer_blocks", None):
            first_block = model.transformer_blocks[0]
            register("block0.norm1", getattr(first_block, "norm1", None))
            register("block0.attention", getattr(first_block, "attention", None))
            register("block0.dropout", getattr(first_block, "dropout", None))
            register("block0.norm2", getattr(first_block, "norm2", None))
            register("block0.feed_forward", getattr(first_block, "feed_forward", None))
        register("final_layer_norm", getattr(model, "layer_norm", None))

        logits = model(preview_inputs)

        for handle in hook_handles:
            handle.remove()

        for name, tensor in debug_snapshots:
            _describe_tensor(name, tensor)

        _describe_tensor(
            "logits_preview",
            logits[:, : min(logits.size(1), 2), : min(logits.size(-1), 8)],
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    runner = TrainingLoopRunner(
        config,
        device,
        pipeline.checkpoints,
        mode,
        reference_model=pipeline.reference_model,
        dpo_beta=getattr(config, "dpo_beta", 0.1),
    )
    loss = runner._forward_backward(
        model,
        tokenizer,
        batch,
        criterion,
        scaler=None,
        accumulation_steps=1,
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    print("\n⚙️  训练步完成")
    print(f"  loss={loss.item():.4f}, grad_norm={float(grad_norm):.4f}")

    model.eval()
    prompt_ids_full = _encode_text(tokenizer, prompt, config.max_seq_len)
    print("\n🧾 推理前准备")
    _describe_tensor("prompt_input_ids(padded)", prompt_ids_full.unsqueeze(0))
    if pad_id is not None:
        _describe_padding("prompt_input_ids", prompt_ids_full.unsqueeze(0), pad_id)
    decoded_prompt = _decode_tokens(tokenizer, prompt_ids_full.tolist())
    print(f"  经过分词器解码的提示: {decoded_prompt.strip()}")

    if pad_id is not None:
        prompt_ids = prompt_ids_full[prompt_ids_full != pad_id]
    else:
        prompt_ids = prompt_ids_full
    if prompt_ids.numel() == 0:
        fill_id = bos_id if bos_id is not None else (0 if pad_id is None else pad_id)
        prompt_ids = torch.tensor([fill_id], dtype=torch.long)
    prompt_tensor = prompt_ids.unsqueeze(0).to(device)
    print(f"  实际用于推理的 token 数: {prompt_tensor.size(1)}")
    _describe_tensor("prompt_tensor", prompt_tensor)
    with torch.no_grad():
        logits_prompt = model(prompt_tensor)
        last_logits = logits_prompt[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        topk = min(5, probs.size(-1))
        topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)
        decoded_tokens = []
        for idx in topk_indices[0].tolist():
            decoded_tokens.append(_decode_tokens(tokenizer, [int(idx)]).strip() or repr(int(idx)))
        print(
            "  首个生成步 top-k 预测: "
            + ", ".join(
                f"{tok}({prob:.3f})" for tok, prob in zip(decoded_tokens, topk_probs[0].tolist())
            )
        )

    max_length = min(
        config.max_generate_length,
        prompt_tensor.size(1) + max_new_tokens,
        config.max_seq_len,
    )
    with torch.no_grad():
        generated = model.generate(prompt_tensor, max_length=max_length)
    decoded = _decode_tokens(tokenizer, generated[0].tolist())

    print("\n💬 推理输出")
    _describe_tensor("generated_token_ids", generated)
    print(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniGPT 全栈调试脚本")
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "sft", "dpo"], help="调试阶段")
    parser.add_argument("--model-size", default="tiny", help="模型规模，对应 training_config 配置")
    parser.add_argument("--prompt", default="你好，MiniGPT！", help="用于推理阶段的提示")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="生成的最大新token数")
    args = parser.parse_args()

    run_fullstack_debug(args.mode, args.model_size, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
