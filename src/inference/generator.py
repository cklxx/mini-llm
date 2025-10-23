"""High-level text generation helpers aligned with project defaults."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Generation configuration shared across decoding strategies."""

    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    history_turns: int = 0
    use_chat_template: bool = True


class TextGenerator:
    """Text generator supporting common decoding strategies."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        *,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        self._device_type = torch.device(device).type
        self.autocast_dtype = autocast_dtype if self._device_type == "cuda" else None

    # ------------------------------------------------------------------
    def apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.1
    ) -> torch.Tensor:
        """Apply a repetition penalty to discourage repeated tokens."""

        if penalty == 1.0:
            return logits

        token_mask = torch.zeros_like(logits)
        token_mask.scatter_(1, input_ids, 1.0)
        positive_logits = torch.where(token_mask.bool(), logits.clamp(min=0), torch.zeros_like(logits))
        negative_logits = torch.where(token_mask.bool(), logits.clamp(max=0), torch.zeros_like(logits))
        logits = logits.clone()
        logits += positive_logits * (1.0 / penalty - 1.0)
        logits += negative_logits * (penalty - 1.0)
        return logits

    # ------------------------------------------------------------------
    def top_k_filtering(self, logits: torch.Tensor, top_k: int = 50) -> torch.Tensor:
        """Filter logits using top-k sampling."""

        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.size(-1))
        top_k_scores, top_k_indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, -float("inf"))
        mask.scatter_(1, top_k_indices, top_k_scores)
        return mask

    # ------------------------------------------------------------------
    def top_p_filtering(self, logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling."""

        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, -float("inf"))

    # ------------------------------------------------------------------
    def greedy_search(self, input_ids: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """Greedy decoding."""

        eos_id = getattr(self.tokenizer, "eos_id", getattr(self.tokenizer, "eos_token_id", None))
        with torch.inference_mode():
            for _ in range(max_length):
                with self._autocast_context():
                    outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if eos_id is not None and next_token.item() == eos_id:
                    break
        return input_ids

    # ------------------------------------------------------------------
    def sample_generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stochastic decoding with project-aligned heuristics."""

        eos_id = getattr(self.tokenizer, "eos_id", getattr(self.tokenizer, "eos_token_id", None))
        device = input_ids.device
        attn = attention_mask.to(device) if attention_mask is not None else None

        with torch.inference_mode():
            for _ in range(config.max_length):
                with self._autocast_context():
                    model_kwargs = {"attention_mask": attn} if attn is not None else {}
                    outputs = self.model(input_ids, **model_kwargs)

                next_token_logits = outputs[:, -1, :]
                temperature = max(config.temperature, 1e-5)
                next_token_logits = next_token_logits / temperature

                if config.repetition_penalty != 1.0:
                    next_token_logits = self.apply_repetition_penalty(
                        next_token_logits, input_ids, config.repetition_penalty
                    )

                if config.top_k > 0:
                    next_token_logits = self.top_k_filtering(next_token_logits, config.top_k)

                if config.top_p < 1.0:
                    next_token_logits = self.top_p_filtering(next_token_logits, config.top_p)

                probs = F.softmax(next_token_logits, dim=-1)

                if config.do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if attn is not None:
                    attn = torch.cat(
                        [attn, torch.ones((attn.size(0), 1), device=device, dtype=attn.dtype)],
                        dim=-1,
                    )

                if eos_id is not None and next_token.item() == eos_id:
                    break

        return input_ids

    # ------------------------------------------------------------------
    def beam_search(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Standard beam-search decoding."""

        beam_size = max(1, config.num_beams)
        expanded_input_ids = input_ids.repeat(beam_size, 1)
        beam_scores = torch.zeros(beam_size, device=self.device)

        with torch.inference_mode():
            for step in range(config.max_length):
                with self._autocast_context():
                    outputs = self.model(expanded_input_ids)
                next_token_logits = outputs[:, -1, :]
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)

                if step == 0:
                    next_token_scores = next_token_scores[0:1, :]
                    beam_scores = beam_scores[0:1]
                    expanded_input_ids = expanded_input_ids[0:1, :]

                scores = beam_scores.unsqueeze(1) + next_token_scores
                scores = scores.reshape(-1)
                top_scores, top_indices = torch.topk(scores, beam_size)

                beam_indices = top_indices // next_token_logits.size(-1)
                token_indices = top_indices % next_token_logits.size(-1)

                expanded_input_ids = torch.cat(
                    [expanded_input_ids[beam_indices], token_indices.unsqueeze(-1)], dim=-1
                )
                beam_scores = top_scores

        best_index = int(torch.argmax(beam_scores).item())
        return expanded_input_ids[best_index : best_index + 1]

    # ------------------------------------------------------------------
    def generate_chat_response(
        self,
        prompt: str,
        *,
        history: Iterable[dict[str, str]] | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Build a chat prompt following the project template and decode."""

        config = config or GenerationConfig()
        history = list(history or [])

        if config.history_turns > 0 and history:
            history = history[-config.history_turns :]

        if config.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = history + [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generated = self.sample_generate(input_ids, config, attention_mask=attention_mask)
        start_idx = input_ids.size(1)
        response_tokens = generated[:, start_idx:]
        return self.tokenizer.decode(response_tokens[0].tolist(), skip_special_tokens=True)

    # ------------------------------------------------------------------
    def _autocast_context(self):
        if self._device_type == "cuda" and self.autocast_dtype is not None:
            return torch.cuda.amp.autocast(dtype=self.autocast_dtype)
        return nullcontext()

