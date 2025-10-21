"""Standalone training loop logic for the MiniGPT trainer."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class TrainingControl:
    """Shared stop-flag between the trainer and the signal handler."""

    interrupted: bool = False


class TrainingLoopRunner:
    """Execute the gradient update loop and associated evaluations."""

    def __init__(
        self,
        config,
        device: str,
        checkpoints,
        mode: str,
        *,
        reference_model=None,
        dpo_beta: float = 0.1,
    ):
        self.config = config
        self.device = device
        self.checkpoints = checkpoints
        self.mode = mode
        self.reference_model = reference_model
        self.dpo_beta = dpo_beta

    # ------------------------------------------------------------------
    def run(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        scaler,
        monitor,
        control: TrainingControl,
        start_step: int,
        start_time: float,
        memory_hooks=None,
        regression_suite=None,
        benchmark_evaluator=None,
    ) -> str:
        model.train()
        step = start_step
        accumulation_steps = self.config.gradient_accumulation_steps

        if self.mode == "dpo" and self.reference_model is None:
            raise RuntimeError("DPO训练需要提供参考模型权重")
        if self.mode == "dpo" and self.reference_model is not None:
            self.reference_model.eval()

        print(f"开始训练，最大步数: {self.config.max_steps}")
        print(
            f"Batch size: {self.config.batch_size}, 梯度累积: {accumulation_steps}, "
            f"有效batch: {self.config.batch_size * accumulation_steps}"
        )

        if memory_hooks is not None:
            memory_hooks.on_train_start()

        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 初始GPU内存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")

        best_val_loss = float("inf")
        no_improve_steps = 0

        for epoch in range(1000):
            epoch_loss = 0.0
            epoch_steps = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                if control.interrupted:
                    print(f"\n⚠️  训练被用户中断（步骤 {step}）")
                    break
                if step >= self.config.max_steps:
                    break

                seq_length = self._resolve_sequence_length(batch)
                if seq_length < 2:
                    continue

                try:
                    loss = self._forward_backward(
                        model,
                        tokenizer,
                        batch,
                        criterion,
                        scaler,
                        accumulation_steps,
                    )
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    if memory_hooks is not None:
                        memory_hooks.on_oom()
                    raise

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
                    train_loader
                ):
                    step, grad_norm = self._optimizer_step(
                        model, optimizer, scheduler, scaler, step
                    )

                    actual_loss = loss.item() * accumulation_steps
                    epoch_loss += actual_loss
                    epoch_steps += 1

                    current_batch_size = self._resolve_batch_size(batch)
                    monitor.log_step(
                        step=step,
                        epoch=epoch,
                        loss=actual_loss,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        batch_size=current_batch_size * accumulation_steps,
                        grad_norm=grad_norm,
                    )

                    if memory_hooks is not None:
                        memory_hooks.on_step_end(step)

                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    elapsed = time.time() - start_time
                    current_lr = optimizer.param_groups[0]["lr"]
                    lr_phase = "Warmup" if step < self.config.warmup_steps else "Decay"
                    lr_progress = (
                        f"{step}/{self.config.warmup_steps}"
                        if step < self.config.warmup_steps
                        else f"{step}/{self.config.max_steps}"
                    )
                    print(
                        f"Step {step:5d} | Loss: {actual_loss:.4f} | Avg: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} ({lr_phase} {lr_progress}) | Time: {elapsed/60:.1f}min"
                    )

                    if step % 100 == 0:
                        self.checkpoints.save(model, optimizer, step, actual_loss, tokenizer)

                    if self.mode != "dpo" and step % self.config.eval_steps == 0:
                        eval_metrics = self._evaluate(
                            model,
                            tokenizer,
                            val_loader,
                            criterion,
                            monitor,
                            step,
                            regression_suite,
                        )
                        if eval_metrics and "val_loss" in eval_metrics:
                            best_val_loss, no_improve_steps = self._maybe_update_best(
                                model,
                                optimizer,
                                tokenizer,
                                step,
                                eval_metrics["val_loss"],
                                best_val_loss,
                                no_improve_steps,
                                control,
                            )

                    if step >= self.config.max_steps:
                        break

                    if benchmark_evaluator is not None:
                        benchmark_evaluator.maybe_run(model, step, monitor)

            if step >= self.config.max_steps or control.interrupted:
                break

        if control.interrupted:
            print("\n💾 正在保存中断checkpoint...")
            self.checkpoints.save(
                model,
                optimizer,
                step,
                epoch_loss / max(epoch_steps, 1) if epoch_steps else 0.0,
                tokenizer,
            )
            print("💡 可使用 --auto-resume 从此处恢复训练")

        final_path = self.checkpoints.save_final(model, tokenizer, step)

        if benchmark_evaluator is not None:
            benchmark_evaluator.maybe_run(model, step, monitor, force=True)

        if control.interrupted:
            print("\n⚠️  训练被用户中断")
            print("✅ 已成功保存中断时的模型状态")
        else:
            print(f"\n🎉 {self.mode}训练完成！")

        print(f"总步数: {step}")
        print(f"训练时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"最终模型已保存: {final_path}")
        return final_path

    # ------------------------------------------------------------------
    def _forward_backward(self, model, tokenizer, batch, criterion, scaler, accumulation_steps):
        if self.mode == "dpo":
            return self._forward_backward_dpo(
                model, tokenizer, batch, scaler, accumulation_steps
            )
        try:
            if isinstance(batch, dict):
                full_input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                if full_input_ids.size(1) < 2:
                    raise ValueError("批次序列长度过短")

                if "labels" in batch:
                    full_target_ids = batch["labels"].to(self.device, non_blocking=True)
                else:
                    full_target_ids = torch.cat(
                        [
                            full_input_ids[:, 1:],
                            torch.full(
                                (full_input_ids.size(0), 1),
                                tokenizer.pad_id,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ],
                        dim=1,
                    )

                input_ids = full_input_ids[:, :-1]
                target_ids = full_target_ids[:, 1:]

                attention_mask = None
                if "attention_mask" in batch:
                    attention_mask = batch["attention_mask"].to(
                        self.device, non_blocking=True
                    )
                    attention_mask = attention_mask[:, :-1]

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        model_kwargs = {}
                        if attention_mask is not None:
                            model_kwargs["attention_mask"] = attention_mask
                        outputs = model(input_ids, **model_kwargs)
                        loss = criterion(
                            outputs.reshape(-1, outputs.size(-1)),
                            target_ids.reshape(-1),
                        )
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    model_kwargs = {}
                    if attention_mask is not None:
                        model_kwargs["attention_mask"] = attention_mask
                    outputs = model(input_ids, **model_kwargs)
                    loss = criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        target_ids.reshape(-1),
                    )
                    loss = loss / accumulation_steps
                    loss.backward()

                return loss

            if isinstance(batch, (list, tuple)):
                if len(batch) != 3:
                    raise ValueError("预训练批次应包含 (X, Y, loss_mask)")

                inputs, targets, loss_mask = [
                    tensor.to(self.device, non_blocking=True) for tensor in batch
                ]

                if inputs.size(1) == 0 or targets.size(1) == 0:
                    raise ValueError("批次序列长度过短")

                valid_mask = loss_mask.to(dtype=torch.float32)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        per_token_loss = F.cross_entropy(
                            outputs.reshape(-1, outputs.size(-1)),
                            targets.reshape(-1),
                            ignore_index=tokenizer.pad_id,
                            reduction="none",
                            label_smoothing=getattr(
                                self.config, "label_smoothing", 0.0
                            ),
                        )
                        per_token_loss = per_token_loss.view_as(targets)
                        denom = valid_mask.sum().clamp_min(1.0)
                        loss = (per_token_loss * valid_mask).sum() / denom
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(inputs)
                    per_token_loss = F.cross_entropy(
                        outputs.reshape(-1, outputs.size(-1)),
                        targets.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                        reduction="none",
                        label_smoothing=getattr(
                            self.config, "label_smoothing", 0.0
                        ),
                    )
                    per_token_loss = per_token_loss.view_as(targets)
                    denom = valid_mask.sum().clamp_min(1.0)
                    loss = (per_token_loss * valid_mask).sum() / denom
                    loss = loss / accumulation_steps
                    loss.backward()

                return loss

            batch = batch.to(self.device, non_blocking=True)
            if batch.size(1) < 2:
                raise ValueError("批次序列长度过短")
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            attention_mask = None

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids)
                    loss = criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        target_ids.reshape(-1),
                    )
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    target_ids.reshape(-1),
                )
                loss = loss / accumulation_steps
                loss.backward()

            return loss

        except torch.cuda.OutOfMemoryError:
            self._handle_oom(batch, tokenizer)
            raise

    def _forward_backward_dpo(self, model, tokenizer, batch, scaler, accumulation_steps):
        chosen_input_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
        chosen_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
        rejected_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

        chosen_targets = chosen_labels[:, 1:]
        rejected_targets = rejected_labels[:, 1:]

        if chosen_targets.size(1) == 0 or rejected_targets.size(1) == 0:
            raise ValueError("DPO 样本序列长度不足以计算对数概率")

        chosen_mask = (chosen_targets != tokenizer.pad_id).float()
        rejected_mask = (rejected_targets != tokenizer.pad_id).float()

        autocast_ctx = torch.amp.autocast("cuda") if scaler is not None else nullcontext()
        with autocast_ctx:
            with torch.no_grad():
                ref_chosen_logits = self._extract_logits(
                    self.reference_model(chosen_input_ids)
                )[:, :-1, :]
                ref_rejected_logits = self._extract_logits(
                    self.reference_model(rejected_input_ids)
                )[:, :-1, :]

            policy_chosen_logits = self._extract_logits(model(chosen_input_ids))[:, :-1, :]
            policy_rejected_logits = self._extract_logits(model(rejected_input_ids))[:, :-1, :]

            ref_chosen_logps = self._sequence_log_probs(
                ref_chosen_logits, chosen_targets, chosen_mask
            )
            ref_rejected_logps = self._sequence_log_probs(
                ref_rejected_logits, rejected_targets, rejected_mask
            )
            policy_chosen_logps = self._sequence_log_probs(
                policy_chosen_logits, chosen_targets, chosen_mask
            )
            policy_rejected_logps = self._sequence_log_probs(
                policy_rejected_logits, rejected_targets, rejected_mask
            )

            policy_diff = policy_chosen_logps - policy_rejected_logps
            ref_diff = ref_chosen_logps - ref_rejected_logps
            loss = -F.logsigmoid(self.dpo_beta * (policy_diff - ref_diff)).mean()
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    @staticmethod
    def _sequence_log_probs(logits, labels, mask):
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        masked = selected * mask
        token_counts = mask.sum(dim=-1).clamp(min=1.0)
        return masked.sum(dim=-1) / token_counts

    def _resolve_sequence_length(self, batch):
        if isinstance(batch, dict):
            if "input_ids" in batch:
                return batch["input_ids"].size(1)
            if "chosen_input_ids" in batch:
                return batch["chosen_input_ids"].size(1)
            raise ValueError("无法从批次数据中解析序列长度")
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("空批次无法解析序列长度")
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.size(1)
            raise ValueError("批次元素不是张量，无法解析序列长度")
        return batch.size(1)

    def _resolve_batch_size(self, batch):
        if isinstance(batch, dict):
            if "input_ids" in batch:
                return batch["input_ids"].size(0)
            if "chosen_input_ids" in batch:
                return batch["chosen_input_ids"].size(0)
            raise ValueError("无法从批次数据中解析批次大小")
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("空批次无法解析批次大小")
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.size(0)
            raise ValueError("批次元素不是张量，无法解析批次大小")
        return batch.size(0)

    def _optimizer_step(self, model, optimizer, scheduler, scaler, step: int) -> tuple[int, float]:
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        if isinstance(grad_norm, torch.Tensor):
            grad_norm_value = float(grad_norm.detach().cpu().item())
        else:
            grad_norm_value = float(grad_norm)

        return step + 1, grad_norm_value

    def _maybe_update_best(
        self,
        model,
        optimizer,
        tokenizer,
        step: int,
        val_loss: float,
        best_val_loss: float,
        no_improve_steps: int,
        control: TrainingControl,
    ) -> tuple[float, int]:
        delta = getattr(self.config, "early_stopping_delta", 0.0)
        patience = getattr(self.config, "early_stopping_patience", 0)

        if val_loss + delta < best_val_loss:
            best_val_loss = val_loss
            no_improve_steps = 0
            self.checkpoints.save(model, optimizer, step, val_loss, tokenizer)
            print("💾 验证集指标提升，已保存最佳模型")
        else:
            no_improve_steps += 1
            if patience > 0 and no_improve_steps >= patience:
                print(f"🛑 触发早停: 验证集在连续 {no_improve_steps} 次评估后未改进")
                control.interrupted = True

        return best_val_loss, no_improve_steps

    # ------------------------------------------------------------------
    def _evaluate(
        self, model, tokenizer, val_loader, criterion, monitor, step, regression_suite=None
    ):
        metrics: dict[str, Any] = {}
        if self.mode == "dpo":
            print("ℹ️ DPO 模式暂不执行验证评估，跳过。")
            try:
                self._smoke_generation(model, tokenizer, step)
            except Exception as exc:
                print(f"⚠️  生成验证失败: {exc}")
            return metrics
        if val_loader:
            model.eval()
            total_loss = 0.0
            total_tokens = 0.0
            pad_id = tokenizer.pad_id
            label_smoothing = getattr(self.config, "label_smoothing", 0.0)
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        full_input_ids = batch["input_ids"].to(
                            self.device, non_blocking=True
                        )
                        full_target_ids = batch["labels"].to(
                            self.device, non_blocking=True
                        )
                        if full_input_ids.size(1) < 2:
                            continue
                        input_ids = full_input_ids[:, :-1]
                        target_ids = full_target_ids[:, 1:]
                        logits = model(input_ids)
                        loss = criterion(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                        )
                        valid_tokens = torch.count_nonzero(target_ids != pad_id).item()
                        if valid_tokens == 0:
                            continue
                        total_loss += loss.item() * valid_tokens
                        total_tokens += float(valid_tokens)
                    elif isinstance(batch, (list, tuple)):
                        if len(batch) != 3:
                            continue
                        inputs, targets, loss_mask = [
                            tensor.to(self.device, non_blocking=True) for tensor in batch
                        ]
                        if inputs.size(1) == 0 or targets.size(1) == 0:
                            continue
                        logits = model(inputs)
                        per_token_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1),
                            ignore_index=pad_id,
                            reduction="none",
                            label_smoothing=label_smoothing,
                        )
                        per_token_loss = per_token_loss.view_as(targets)
                        mask = loss_mask.to(dtype=torch.float32)
                        tokens = mask.sum().item()
                        if tokens <= 0:
                            continue
                        total_loss += (per_token_loss * mask).sum().item()
                        total_tokens += tokens
                    else:
                        batch = batch.to(self.device, non_blocking=True)
                        if batch.size(1) < 2:
                            continue
                        input_ids = batch[:, :-1]
                        target_ids = batch[:, 1:]
                        logits = model(input_ids)
                        loss = criterion(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                        )
                        valid_tokens = torch.count_nonzero(target_ids != pad_id).item()
                        if valid_tokens == 0:
                            continue
                        total_loss += loss.item() * valid_tokens
                        total_tokens += float(valid_tokens)
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            perplexity = (
                math.exp(min(20, avg_loss))
                if avg_loss not in (float("inf"), float("nan"))
                else float("inf")
            )
            metrics = {
                "val_loss": avg_loss,
                "perplexity": perplexity,
                "val_tokens": total_tokens,
            }
            monitor.log_validation(step, avg_loss, perplexity, {"ValTokens": total_tokens})
            print(
                f"🔍 验证 Step {step}: loss={avg_loss:.4f}, ppl={perplexity:.2f}, tokens={total_tokens}"
            )
            model.train()
        else:
            print("ℹ️ 当前未配置验证集，跳过验证损失计算，仅进行生成检查")
        try:
            self._smoke_generation(model, tokenizer, step)
        except Exception as exc:
            print(f"⚠️  生成验证失败: {exc}")

        if regression_suite is not None:
            try:
                regression_suite.maybe_run(model, tokenizer, step, monitor=monitor)
            except Exception as exc:
                print(f"⚠️  Regression suite failed: {exc}")
        return metrics

    def _smoke_generation(
        self, model, tokenizer, step, prompt: str = "你好，我是", max_new_tokens: int = 32
    ):
        model.eval()
        try:
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=self.device)
            generated = input_tensor
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(generated)
                    next_token_logits = outputs[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            decoded = tokenizer.decode(generated[0].tolist())
            print(f"🧪 Step {step} 生成样例: {decoded[:200]}")
        finally:
            model.train()

    # ------------------------------------------------------------------
    def _handle_oom(self, batch, tokenizer) -> None:
        print("\n❌ CUDA OOM错误!")
        if isinstance(batch, dict):
            batch_size = batch["input_ids"].size(0)
            seq_length = batch["input_ids"].size(1)
        elif isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            batch_size = first.size(0)
            seq_length = first.size(1) if first.dim() > 1 else 0
        else:
            batch_size = batch.size(0)
            seq_length = batch.size(1) if batch.dim() > 1 else 0
        print(f"   当前批次大小: {batch_size}")
        print(f"   序列长度: {seq_length}")
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU内存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
            torch.cuda.empty_cache()
        print("\n💡 建议解决方案:")
        print(f"   1. 降低batch_size: --batch-size {self.config.batch_size // 2}")
        print(
            f"   2. 增加梯度累积: 当前={self.config.gradient_accumulation_steps}, 建议={self.config.gradient_accumulation_steps * 2}"
        )
        print(f"   3. 减小序列长度: 当前max_seq_len={self.config.max_seq_len}")
        print("   4. 启用梯度检查点 (gradient checkpointing)")
        print("   5. 设置环境变量: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
