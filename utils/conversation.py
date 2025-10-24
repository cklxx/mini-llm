"""Conversation normalization helpers shared across MiniLLM data tooling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence


@dataclass
class Message:
    """Normalized chat message."""

    role: str
    content: str

    def to_dict(self) -> Mapping[str, str]:
        return {"role": self.role, "content": self.content}


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
    "observation": "assistant",
}


def _coerce_role(raw: str) -> str:
    role = ROLE_ALIASES.get(raw.lower(), raw.lower())
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Unsupported role '{raw}' in conversation record")
    return role


def normalize_conversation(record: Mapping[str, object]) -> List[Message]:
    """Return a normalized conversation list for a wide range of data formats."""

    def _from_messages(messages: Sequence[Mapping[str, object]]) -> List[Message]:
        normalized: List[Message] = []
        for item in messages:
            if not isinstance(item, Mapping):
                raise TypeError("Conversation message must be a mapping")
            role_raw = item.get("role") or item.get("from") or item.get("speaker")
            if role_raw is None:
                raise KeyError("Conversation message is missing 'role'/'from'/'speaker'")
            content_raw = item.get("content") or item.get("value") or item.get("text")
            if content_raw is None:
                raise KeyError("Conversation message is missing text field")
            normalized.append(Message(role=_coerce_role(str(role_raw)), content=str(content_raw).strip()))
        return normalized

    if "conversations" in record:
        conv = record["conversations"]
        if isinstance(conv, Mapping):  # some datasets nest messages under key "messages"
            conv = conv.get("messages", [])
        if not isinstance(conv, Sequence):
            raise TypeError("'conversations' must be a sequence of messages")
        return _from_messages(conv)

    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, Sequence):
            raise TypeError("'messages' must be a sequence")
        return _from_messages(messages)

    if "instruction" in record and "output" in record:
        system_prompt = str(record.get("system", "")).strip()
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        instruction = str(record["instruction"]).strip()
        input_text = str(record.get("input", "")).strip()
        if input_text:
            instruction = f"{instruction}\n\n{input_text}"
        messages.append(Message(role="user", content=instruction))
        messages.append(Message(role="assistant", content=str(record["output"]).strip()))
        return messages

    if "question" in record and ("answer" in record or "response" in record):
        messages = []
        system_prompt = str(record.get("system", "")).strip()
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=str(record["question"]).strip()))
        answer = record.get("answer", record.get("response"))
        messages.append(Message(role="assistant", content=str(answer).strip()))
        return messages

    if "prompt" in record and ("completion" in record or "response" in record):
        messages = []
        system_prompt = str(record.get("system", "")).strip()
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=str(record["prompt"]).strip()))
        completion = record.get("completion", record.get("response"))
        messages.append(Message(role="assistant", content=str(completion).strip()))
        return messages

    raise ValueError("Unsupported conversation record format")


def conversation_to_template(messages: Sequence[Message], *, include_bos: bool = False) -> str:
    """Render a conversation to a ChatML-like template using special tokens."""

    segments: List[str] = []
    if include_bos:
        segments.append("<|bos|>")
    for msg in messages:
        content = msg.content.strip()
        if msg.role == "system":
            segments.append(content)
            continue
        if msg.role == "user":
            segments.append(f"<|user_start|>{content}<|user_end|>")
            continue
        if msg.role == "assistant":
            segments.append(f"<|assistant_start|>{content}<|assistant_end|>")
            continue
        raise ValueError(f"Unsupported role in template rendering: {msg.role}")
    return "\n".join(segments)


def flatten_conversation(messages: Sequence[Message]) -> str:
    """Convert a conversation to a plain text transcript (for pretraining)."""

    lines: List[str] = []
    for msg in messages:
        prefix = {"system": "[系统]", "user": "[用户]", "assistant": "[助手]"}[msg.role]
        lines.append(f"{prefix} {msg.content.strip()}")
    return "\n".join(lines)
