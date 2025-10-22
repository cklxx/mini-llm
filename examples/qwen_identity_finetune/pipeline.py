"""Utilities for fine-tuning Qwen models with persona-aligned identity data.

This module provides a complete pipeline that mirrors the accompanying
`docs/case_studies/qwen_identity_finetune.md` tutorial.  It can download and prepare the
Delicate Medical R1 dataset, generate additional persona-aligned samples via
inference, mix them with the training data, and launch a supervised fine-tuning
run that tracks metrics in SwanLab.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset
from modelscope import MsDataset, snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import swanlab

LOGGER = logging.getLogger(__name__)


@dataclass
class PersonaSpecification:
    """Configuration for generating persona-aligned identity data."""

    persona_name: str
    identity_statement: str
    mix_ratio: float = 0.2
    max_identity_samples: Optional[int] = None
    warmup_questions: Optional[Sequence[str]] = None


@dataclass
class IdentityFineTuneConfig:
    """High level configuration for the identity fine-tuning pipeline."""

    work_dir: str
    persona: PersonaSpecification
    dataset_name: str = "krisfu/delicate_medical_r1_data"
    dataset_subset: str = "default"
    dataset_split: str = "train"
    seed: int = 42
    prompt: str = (
        "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    )
    base_model_repo: str = "Qwen/Qwen3-1.7B"
    base_model_revision: str = "master"
    cache_dir: str = "./pretrained"
    max_length: int = 2048
    train_split_ratio: float = 0.9
    per_device_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    train_epochs: int = 1
    learning_rate: float = 1e-4
    eval_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 400
    gradient_checkpointing: bool = True
    run_name: str = "qwen3-identity"
    output_subdir: str = "outputs"
    identity_generation_temperature: float = 0.7
    identity_generation_top_p: float = 0.95
    predictions_to_log: int = 3

    def __post_init__(self) -> None:
        self.work_dir = os.path.abspath(self.work_dir)
        self.output_dir = os.path.join(self.work_dir, self.output_subdir)
        self.cache_dir = os.path.abspath(self.cache_dir)
        if not 0 < self.train_split_ratio < 1:
            raise ValueError("train_split_ratio must be between 0 and 1 (exclusive)")
        if not 0 <= self.persona.mix_ratio <= 1:
            raise ValueError("persona mix_ratio must be within [0, 1]")
        if self.predictions_to_log < 0:
            raise ValueError("predictions_to_log must be non-negative")


def _ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonl_dump(records: Iterable[Dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonl_load(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _download_dataset(config: IdentityFineTuneConfig) -> List[Dict[str, str]]:
    LOGGER.info(
        "Downloading dataset %s (subset=%s split=%s)",
        config.dataset_name,
        config.dataset_subset,
        config.dataset_split,
    )
    dataset = MsDataset.load(
        config.dataset_name,
        subset_name=config.dataset_subset,
        split=config.dataset_split,
    )
    return list(dataset)


def _split_dataset(
    data: Sequence[Dict[str, str]], split_ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * split_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def _parse_identity_response(text: str) -> Optional[Tuple[str, str]]:
    cleaned = text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
            think = payload.get("think", "").strip()
            answer = payload.get("answer", "").strip()
            if think and answer:
                return think, answer
        except json.JSONDecodeError:
            LOGGER.debug("Failed to parse JSON response: %s", cleaned)

    if "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.index("<think>") + len("<think>")
        end = cleaned.index("</think>")
        think = cleaned[start:end].strip()
        answer = cleaned[end + len("</think>") :].strip()
        if think and answer:
            return think, answer

    lines = cleaned.splitlines()
    if len(lines) >= 2:
        think = lines[0].strip()
        answer = "\n".join(lines[1:]).strip()
        if think and answer:
            return think, answer
    return None


def _generate_identity_samples(
    config: IdentityFineTuneConfig,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    base_records: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    persona = config.persona
    if persona.mix_ratio <= 0:
        LOGGER.info("Persona mix ratio <= 0. Skipping identity generation.")
        return []

    if not base_records:
        LOGGER.warning("Base dataset is empty. No persona samples will be generated.")
        return []

    rng = random.Random(config.seed)
    target_identity = max(1, int(len(base_records) * persona.mix_ratio))
    if persona.max_identity_samples is not None:
        target_identity = min(target_identity, persona.max_identity_samples)

    candidate_indices = list(range(len(base_records)))
    rng.shuffle(candidate_indices)
    selected_indices = candidate_indices[:target_identity]

    prompts: List[Tuple[str, Optional[int]]] = [
        (base_records[idx]["question"], idx) for idx in selected_indices
    ]

    if persona.warmup_questions:
        prompts.extend((question, None) for question in persona.warmup_questions)

    LOGGER.info(
        "Generating %d persona-aligned samples for persona '%s'",
        len(prompts),
        persona.persona_name,
    )

    identity_records: List[Dict[str, str]] = []
    for question, source_index in prompts:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{config.prompt}\n\nPersona Identity: {persona.identity_statement}\n"
                    "请严格输出JSON，格式为 {\"think\": \"...\", \"answer\": \"...\"}."
                ),
            },
            {"role": "user", "content": question},
        ]
        chat_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([chat_template], return_tensors="pt")
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=config.identity_generation_temperature,
                top_p=config.identity_generation_top_p,
                do_sample=True,
            )
        generated_ids = generated[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = _parse_identity_response(response)
        if not parsed:
            LOGGER.warning("Skipping persona sample due to parse failure: %s", response)
            continue
        think, answer = parsed
        record = {
            "question": question,
            "think": think,
            "answer": answer,
        }
        if source_index is not None:
            record["source_index"] = source_index
        identity_records.append(record)

    LOGGER.info("Generated %d persona-aligned samples", len(identity_records))
    return identity_records


def _mix_identity_data(
    base_records: List[Dict[str, str]],
    identity_records: List[Dict[str, str]],
    mix_ratio: float,
    seed: int,
) -> List[Dict[str, str]]:
    if not identity_records or mix_ratio <= 0:
        return base_records

    rng = random.Random(seed)
    mixed_records: List[Dict[str, str]] = []
    replacements = 0

    # Map normalized questions to persona records when those questions exist in the base dataset.
    normalized_mapping: Dict[str, Dict[str, str]] = {}
    for record in identity_records:
        key = record["question"].strip()
        if key in normalized_mapping:
            LOGGER.debug("Duplicate persona sample detected for question: %s", key)
            continue
        normalized_mapping[key] = record

    for base_record in base_records:
        key = base_record["question"].strip()
        persona_record = normalized_mapping.get(key)
        if persona_record:
            merged = dict(base_record)
            merged["answer"] = persona_record["answer"]
            if persona_record.get("think"):
                merged["think"] = persona_record["think"]
            mixed_records.append(merged)
            replacements += 1
        else:
            mixed_records.append(base_record)

    # Append persona samples that do not correspond to existing dataset questions.
    base_questions = {item["question"].strip() for item in base_records}
    unmatched = [
        record
        for record in identity_records
        if record["question"].strip() not in base_questions
    ]
    if unmatched:
        mixed_records.extend(unmatched)

    rng.shuffle(mixed_records)
    LOGGER.info(
        "Persona data merged into training set (replaced=%d, appended=%d).",
        replacements,
        len(unmatched),
    )
    return mixed_records


def _convert_to_instruction_format(
    prompt: str, records: Sequence[Dict[str, str]]
) -> List[Dict[str, str]]:
    formatted: List[Dict[str, str]] = []
    for item in records:
        question = item["question"].strip()
        think = item.get("think", "").strip()
        answer = item.get("answer", "").strip()
        output = f"<think>{think}</think>\n{answer}"
        formatted.append(
            {
                "instruction": prompt,
                "input": question,
                "output": output,
            }
        )
    return formatted


def _tokenize_dataset(
    tokenizer: AutoTokenizer,
    prompt: str,
    dataset: Dataset,
    max_length: int,
) -> Dataset:
    def _process_func(example: Dict[str, str]) -> Dict[str, List[int]]:
        instruction = tokenizer(
            f"<|im_start|>system\n{prompt}<|im_end|>\n"
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = tokenizer(example["output"], add_special_tokens=False)
        input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        )
        attention_mask = (
            instruction["attention_mask"]
            + response["attention_mask"]
            + [1]
        )
        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
        )
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(_process_func, remove_columns=dataset.column_names)


def _build_trainer(
    config: IdentityFineTuneConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        num_train_epochs=config.train_epochs,
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="swanlab",
        run_name=config.run_name,
        save_on_each_node=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def _log_predictions(
    config: IdentityFineTuneConfig,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    formatted_records: Sequence[Dict[str, str]],
) -> None:
    preview = list(formatted_records)[: config.predictions_to_log]
    if not preview:
        return

    logs: List[swanlab.Text] = []
    for record in preview:
        messages = [
            {"role": "system", "content": record["instruction"]},
            {"role": "user", "content": record["input"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_length,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        preview_text = (
            f"Question: {record['input']}\n\n"
            f"Ground Truth: {record['output']}\n\n"
            f"Model Output: {decoded}"
        )
        logs.append(swanlab.Text(preview_text))
    if logs:
        swanlab.log({"predictions": logs})


def run_identity_finetune_pipeline(config: IdentityFineTuneConfig) -> None:
    """Execute the full persona-aligned fine-tuning workflow."""
    logging.basicConfig(level=logging.INFO)

    _set_seed(config.seed)
    _ensure_directory(config.work_dir)
    _ensure_directory(config.output_dir)
    _ensure_directory(config.cache_dir)

    raw_dataset_path = os.path.join(config.work_dir, "raw_dataset.jsonl")
    train_dataset_path = os.path.join(config.work_dir, "train.jsonl")
    val_dataset_path = os.path.join(config.work_dir, "val.jsonl")
    identity_dataset_path = os.path.join(config.work_dir, "identity.jsonl")
    mixed_dataset_path = os.path.join(config.work_dir, "train_mixed.jsonl")
    train_formatted_path = os.path.join(config.work_dir, "train_format.jsonl")
    val_formatted_path = os.path.join(config.work_dir, "val_format.jsonl")

    swanlab.config.update(
        {
            "model": config.base_model_repo,
            "prompt": config.prompt,
            "persona": config.persona.persona_name,
            "data_max_length": config.max_length,
        }
    )

    if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
        LOGGER.info("Reusing existing dataset splits from %s", config.work_dir)
        train_records = _jsonl_load(train_dataset_path)
        val_records = _jsonl_load(val_dataset_path)
        if os.path.exists(raw_dataset_path):
            dataset_records = _jsonl_load(raw_dataset_path)
        else:
            dataset_records = train_records + val_records
            _jsonl_dump(dataset_records, raw_dataset_path)
    else:
        dataset_records = _download_dataset(config)
        _jsonl_dump(dataset_records, raw_dataset_path)
        train_records, val_records = _split_dataset(
            dataset_records, config.train_split_ratio, config.seed
        )
        _jsonl_dump(train_records, train_dataset_path)
        _jsonl_dump(val_records, val_dataset_path)

    local_model_dir = snapshot_download(
        config.base_model_repo,
        cache_dir=config.cache_dir,
        revision=config.base_model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()

    model.eval()
    identity_records = _generate_identity_samples(
        config, tokenizer, model, train_records
    )
    model.train()
    export_identity_records = [
        {k: v for k, v in record.items() if k != "source_index"}
        for record in identity_records
    ]
    _jsonl_dump(export_identity_records, identity_dataset_path)

    mixed_train_records = _mix_identity_data(
        train_records,
        identity_records,
        config.persona.mix_ratio,
        config.seed,
    )
    mixed_train_records = [
        {k: v for k, v in record.items() if k != "source_index"}
        for record in mixed_train_records
    ]
    _jsonl_dump(mixed_train_records, mixed_dataset_path)

    train_formatted = _convert_to_instruction_format(config.prompt, mixed_train_records)
    val_formatted = _convert_to_instruction_format(config.prompt, val_records)
    _jsonl_dump(train_formatted, train_formatted_path)
    _jsonl_dump(val_formatted, val_formatted_path)

    train_df = pd.read_json(train_formatted_path, lines=True)
    val_df = pd.read_json(val_formatted_path, lines=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenized_train = _tokenize_dataset(
        tokenizer, config.prompt, train_dataset, config.max_length
    )
    tokenized_val = _tokenize_dataset(
        tokenizer, config.prompt, val_dataset, config.max_length
    )

    trainer = _build_trainer(
        config, model, tokenizer, tokenized_train, tokenized_val
    )

    swanlab.init(
        project=os.environ.get("SWANLAB_PROJECT", config.run_name),
        run_name=config.run_name,
    )
    try:
        trainer.train()
        _log_predictions(config, tokenizer, model, train_formatted)
    finally:
        swanlab.finish()


__all__ = [
    "IdentityFineTuneConfig",
    "PersonaSpecification",
    "run_identity_finetune_pipeline",
]
