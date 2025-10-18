"""Conversation dataset helpers."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    """Dataset that masks non-assistant turns for supervised fine-tuning."""

    def __init__(
        self,
        conversations: list[Any],
        tokenizer,
        max_length: int = 512,
        role_tokens: dict[str, str] | None = None,
        augmentation: dict[str, Any] | None = None,
        seed: int = 42,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.role_tokens = role_tokens or {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "turn_separator": "<|endofturn|>",
        }
        self.turn_separator_ids = (
            self.tokenizer.encode(
                self.role_tokens.get("turn_separator", ""), add_special_tokens=False
            )
            if self.role_tokens.get("turn_separator")
            else []
        )
        self.augmentation = augmentation or {}
        self.turn_truncate_prob = max(0.0, float(self.augmentation.get("turn_truncate_prob", 0.0)))
        self.max_turn_truncate = max(0, int(self.augmentation.get("max_turn_truncate", 0)))
        self.seed = seed
        self.ignore_index = self.tokenizer.pad_id

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self._normalize_conversation(self.conversations[idx])
        messages = self._apply_augmentation(messages, idx)

        return self._build_sample(messages)

    def _build_sample(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        pad_id = self.tokenizer.pad_id

        input_ids: list[int] = [bos_id]
        labels: list[int] = [pad_id]
        attention_mask: list[int] = [1]

        last_role = None

        for turn_idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            role_token = self.role_tokens.get(role, self.role_tokens.get("user", "<|user|>"))

            text = f"{role_token}\n{content.strip()}".strip()
            if not text:
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            input_ids.extend(token_ids)
            attention_mask.extend([1] * len(token_ids))

            if role == "assistant":
                labels.extend(token_ids)
            else:
                labels.extend([pad_id] * len(token_ids))

            last_role = role

            if self.turn_separator_ids and turn_idx < len(messages) - 1:
                input_ids.extend(self.turn_separator_ids)
                attention_mask.extend([1] * len(self.turn_separator_ids))
                labels.extend([pad_id] * len(self.turn_separator_ids))

        input_ids.append(eos_id)
        attention_mask.append(1)
        labels.append(eos_id if last_role == "assistant" else pad_id)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([pad_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([pad_id] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _normalize_conversation(self, conv: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if isinstance(conv, dict) and "input" in conv and "output" in conv:
            user_content = str(conv.get("input", "")).strip()
            assistant_content = str(conv.get("output", "")).strip()
            if user_content:
                messages.append({"role": "user", "content": user_content})
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
        elif isinstance(conv, list):
            for message in conv:
                if not isinstance(message, dict):
                    continue
                role = self._canonical_role(message.get("role", "user"))
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                messages.append({"role": role, "content": content})
        else:
            text = str(conv)
            if text.strip():
                messages.append({"role": "user", "content": text.strip()})

        if not any(msg["role"] == "assistant" for msg in messages):
            messages.append({"role": "assistant", "content": ""})

        return messages

    def _canonical_role(self, role: str) -> str:
        role = (role or "").lower()
        if role in ("assistant", "bot", "gpt"):
            return "assistant"
        if role in ("system", "context"):
            return "system"
        return "user"

    def _apply_augmentation(self, messages: list[dict[str, str]], idx: int) -> list[dict[str, str]]:
        if not messages or self.max_turn_truncate == 0 or self.turn_truncate_prob <= 0.0:
            return messages

        rng = random.Random(self.seed + idx)
        if rng.random() >= self.turn_truncate_prob:
            return messages

        assistant_turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if len(assistant_turns) <= 1:
            return messages

        max_truncate = min(self.max_turn_truncate, len(assistant_turns) - 1)
        if max_truncate <= 0:
            return messages

        truncate_rounds = rng.randint(1, max_truncate)

        truncated = list(messages)
        removed = 0
        while removed < truncate_rounds and truncated:
            while truncated and truncated[-1]["role"] != "assistant":
                truncated.pop()
            if truncated and truncated[-1]["role"] == "assistant":
                truncated.pop()
                removed += 1
                if truncated and truncated[-1]["role"] == "user":
                    truncated.pop()

        return truncated if truncated else messages


class DPODataset(Dataset):
    """Preference dataset that pairs chosen/rejected conversations for DPO."""

    def __init__(
        self,
        records: list[Any],
        tokenizer,
        max_length: int = 1024,
        role_tokens: dict[str, str] | None = None,
        seed: int = 42,
    ):
        valid_records = [
            record
            for record in records
            if isinstance(record, dict) and "chosen" in record and "rejected" in record
        ]
        if not valid_records:
            raise ValueError("DPO数据集中缺少包含 'chosen' 和 'rejected' 字段的样本")

        self.records = valid_records
        # 复用 ConversationDataset 的格式化逻辑，但禁用数据增强。
        self.formatter = ConversationDataset(
            [],
            tokenizer,
            max_length=max_length,
            role_tokens=role_tokens,
            augmentation=None,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]

        chosen_messages = self.formatter._normalize_conversation(record["chosen"])
        rejected_messages = self.formatter._normalize_conversation(record["rejected"])

        chosen_sample = self.formatter._build_sample(chosen_messages)
        rejected_sample = self.formatter._build_sample(rejected_messages)

        return {
            "chosen_input_ids": chosen_sample["input_ids"],
            "chosen_labels": chosen_sample["labels"],
            "chosen_attention_mask": chosen_sample["attention_mask"],
            "rejected_input_ids": rejected_sample["input_ids"],
            "rejected_labels": rejected_sample["labels"],
            "rejected_attention_mask": rejected_sample["attention_mask"],
        }
