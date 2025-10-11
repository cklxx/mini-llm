"""Prompt regression harness for the MiniGPT training pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RegressionPrompt:
    prompt_id: str
    prompt: str
    expect_substrings: list[str]
    max_new_tokens: int


class RegressionSuite:
    """Run deterministic prompt regressions during or after training."""

    def __init__(self, config: Any, output_dir: str, device: str):
        self.enabled: bool = bool(getattr(config, "regression_eval_enabled", False))
        self.interval: int = int(getattr(config, "regression_eval_interval", 500))
        self.prompts_path: str | None = getattr(config, "regression_eval_prompts", None)
        self.max_new_tokens: int = int(getattr(config, "regression_eval_max_new_tokens", 96))
        self.temperature: float = float(getattr(config, "regression_eval_temperature", 0.8))
        self.top_p: float = float(getattr(config, "regression_eval_top_p", 0.95))
        self.output_dir = os.path.join(output_dir, "regression")
        self._device = torch.device(device)
        self._cached_prompts: list[RegressionPrompt] | None = None
        self._last_step_run: int = -1
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _load_prompts(self) -> list[RegressionPrompt]:
        if self._cached_prompts is not None:
            return self._cached_prompts
        prompts: list[RegressionPrompt] = []
        if not self.prompts_path or not os.path.exists(self.prompts_path):
            print(f"⚠️  Regression prompt file missing: {self.prompts_path}")
            self._cached_prompts = []
            return self._cached_prompts
        with open(self.prompts_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                prompt_id = payload.get("id") or payload.get("prompt_id")
                prompt_text = payload["prompt"]
                expected = (
                    payload.get("expect_substrings") or payload.get("expected_substrings") or []
                )
                if isinstance(expected, str):
                    expected = [expected]
                max_new_tokens = int(payload.get("max_new_tokens", self.max_new_tokens))
                prompts.append(
                    RegressionPrompt(
                        prompt_id=prompt_id,
                        prompt=prompt_text,
                        expect_substrings=expected,
                        max_new_tokens=max_new_tokens,
                    )
                )
        self._cached_prompts = prompts
        return prompts

    def should_run(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step == self._last_step_run:
            return False
        if self.interval <= 0:
            return True
        return step % self.interval == 0

    # ------------------------------------------------------------------
    def maybe_run(self, model, tokenizer, step: int, monitor=None) -> None:
        if not self.should_run(step):
            return
        prompts = self._load_prompts()
        if not prompts:
            return

        model_was_training = model.training
        model.eval()

        results: list[dict[str, Any]] = []
        pass_count = 0
        with torch.no_grad():
            for item in prompts:
                input_ids = tokenizer.encode(item.prompt, add_special_tokens=True)
                input_tensor = torch.tensor([input_ids], device=self._device)
                generated = input_tensor
                for _ in range(item.max_new_tokens):
                    outputs = model(generated)
                    next_token_logits = outputs[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    if next_token.item() == tokenizer.eos_id:
                        break
                    generated = torch.cat([generated, next_token], dim=1)
                decoded = tokenizer.decode(generated[0].tolist())
                passed = all(substring in decoded for substring in item.expect_substrings)
                if passed:
                    pass_count += 1
                results.append(
                    {
                        "id": item.prompt_id,
                        "prompt": item.prompt,
                        "expect_substrings": item.expect_substrings,
                        "response": decoded,
                        "passed": passed,
                    }
                )

        pass_rate = pass_count / len(results) if results else 0.0
        timestamp_path = os.path.join(self.output_dir, f"regression_step_{step:06d}.json")
        with open(timestamp_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "step": step,
                    "pass_rate": pass_rate,
                    "total": len(results),
                    "results": results,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(
            f"🧪 Regression suite completed @ step {step}: pass_rate={pass_rate:.2%} -> {timestamp_path}"
        )
        if monitor is not None and hasattr(monitor, "log_regression"):
            monitor.log_regression(step, pass_rate, results)

        if model_was_training:
            model.train()
        self._last_step_run = step
