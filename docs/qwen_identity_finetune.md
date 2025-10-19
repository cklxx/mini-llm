# Qwen 系列模型身份认同微调教程

本文档介绍如何基于 Qwen3-1.7B 模型完成一次身份认同增强的微调流程。目标是在保持原有对话能力的同时，让模型在回答用户问题时展现新的身份设定。整体流程如下：准备环境 → 下载与清洗数据 → 预生成模型回复 → 构建身份认同增强数据集 → 配置并启动训练 → 观察与评估。

## 1. 环境准备

### 1.1 硬件与基础依赖

- Python ≥ 3.8
- 至少 1 张 NVIDIA/昇腾 GPU（建议显存 ≥ 32 GB）
- PyTorch 与匹配的 CUDA 运行环境

### 1.2 Python 依赖

```bash
pip install swanlab modelscope==1.22.0 "transformers>=4.50.0" datasets==3.2.0 accelerate pandas addict
```

> 本教程测试使用的版本：`modelscope==1.22.0`、`transformers==4.51.3`、`datasets==3.2.0`、`peft==0.11.1`、`accelerate==1.6.0`、`swanlab==0.5.7`。

## 2. 数据准备

### 2.1 下载原始医学数据集

本案例使用 [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data) 数据集，适用于医疗问答场景。数据集字段包括 `Instruction`、`question`、`think`、`answer`、`metrics`。我们关注其中的 `question`、`think`、`answer` 字段。

```python
from modelscope.msdatasets import MsDataset
import json
import random

random.seed(42)

ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')
data_list = list(ds)
random.shuffle(data_list)

split_idx = int(len(data_list) * 0.9)

train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
```

### 2.2 构建身份认同增强数据

我们的目标是让模型在回答问题时体现特定身份，例如“资深三甲医院内科主任”。直接在小规模数据集上全参训练可能导致过拟合。为缓解这一问题，先使用基础模型进行推理，生成一批未加入身份设定的参考答案，再与身份认同数据进行混合。

#### 2.2.1 预生成模型回复

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import random

model_path = "./Qwen/Qwen3-1.7B"  # 后续会在第 3 节介绍如何下载

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

PROMPT = "你是一位资深三甲医院内科主任，请以专业、耐心的语气回答患者问题。"

raw_dataset = Dataset.from_json("train.jsonl")

sample_ratio = 0.2
sample_size = max(1, int(len(raw_dataset) * sample_ratio))
selected_indices = random.sample(range(len(raw_dataset)), sample_size)

identity_prompt = (
    PROMPT
    + "\n\nPersona Identity: 你是一位善于共情、用语温和的医学专家。"
    + "请严格输出JSON，格式为 {\"think\": \"...\", \"answer\": \"...\"}."
)

inference_records = []
for idx in selected_indices:
    sample = raw_dataset[idx]
    messages = [
        {"role": "system", "content": identity_prompt},
        {"role": "user", "content": sample["question"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    response = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True
    )[0]
    try:
        payload = json.loads(response)
        think = payload.get("think", "")
        answer = payload.get("answer", "")
    except json.JSONDecodeError:
        think, answer = response.split("\n", 1) if "\n" in response else ("", response)
    inference_records.append({
        "question": sample["question"],
        "think": think,
        "answer": answer
    })

with open("identity_inference.jsonl", "w", encoding="utf-8") as f:
    for record in inference_records:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
```

上述脚本会生成 `identity_inference.jsonl` 文件，用于与原始数据混合。数量可根据资源调节，建议覆盖 10%~20% 的训练样本。

#### 2.2.2 混合身份认同数据

将推理生成的身份认同数据与原始数据进行合并，并保证 `think` 字段保留原始推理或空字符串，`answer` 字段使用带身份风格的输出。

```python
import json
import random

identity_ratio = 0.2

with open("train.jsonl", "r", encoding="utf-8") as f:
    base_records = [json.loads(line) for line in f]

with open("identity_inference.jsonl", "r", encoding="utf-8") as f:
    identity_records = [json.loads(line) for line in f]

random.shuffle(identity_records)
identity_map = {
    record["question"]: record for record in identity_records[: int(len(base_records) * identity_ratio)]
}

mixed_records = []
for record in base_records:
    persona_record = identity_map.get(record["question"])
    if persona_record:
        updated = dict(record)
        updated["answer"] = persona_record.get("answer", "")
        if persona_record.get("think"):
            updated["think"] = persona_record["think"]
        mixed_records.append(updated)
    else:
        mixed_records.append(record)

with open("train_mixed.jsonl", "w", encoding="utf-8") as f:
    for record in mixed_records:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
```

`train_mixed.jsonl` 便是包含身份认同强化后的训练集。若需要额外补充完全新建的身份问题，可以向 `identity_records` 中追加自定义问题，然后在上述流程中一起写入。验证集保持原始形式，便于评估身份设定对泛化能力的影响。

## 3. 下载与加载 Qwen 模型

```python
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="./", revision="master")

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
```

## 4. 数据格式转换与预处理

我们将数据转换为 SFT 所需的 `instruction`/`input`/`output` 结构，并在 `output` 中显式区分 `think` 和 `answer`。

```python
PROMPT = "你是一位资深三甲医院内科主任，请根据患者问题给出带思考过程的回答。"
MAX_LENGTH = 2048


def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            output = f"<think>{data['think']}</think>\n{data['answer']}"
            messages.append({
                "instruction": PROMPT,
                "input": data["question"],
                "output": output
            })
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

## 5. 训练配置与启动

使用 Transformers 的 `Trainer` 进行全参微调，同时接入 SwanLab 追踪实验指标。

```python
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab
import os

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical-identity"

train_path = "train_mixed.jsonl"
val_path = "val.jsonl"

train_format_path = "train_mixed_format.jsonl"
val_format_path = "val_format.jsonl"

for src, dst in [(train_path, train_format_path), (val_path, val_format_path)]:
    if not os.path.exists(dst):
        dataset_jsonl_transfer(src, dst)

train_df = pd.read_json(train_format_path, lines=True)
val_df = pd.read_json(val_format_path, lines=True)

train_dataset = Dataset.from_pandas(train_df).map(process_func, remove_columns=train_df.columns)
val_dataset = Dataset.from_pandas(val_df).map(process_func, remove_columns=val_df.columns)

args = TrainingArguments(
    output_dir="./output/Qwen3-1.7B-identity",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-1.7B-identity",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
```

> 仅训练 1 个 epoch，避免在小数据集上过拟合。如需更长训练，可结合验证集损失和 SwanLab 可视化判断是否出现过拟合趋势。

## 6. 推理与评估

完成训练后，可选取若干条验证集样本验证身份设定效果。

```python
from swanlab import Text

val_preview = val_df.head(3)
preview_logs = []

for _, row in val_preview.iterrows():
    messages = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row["input"]},
    ]
    response = predict(messages, model, tokenizer)
    preview_logs.append(Text(f"Question: {row['input']}\n\nLLM: {response}"))

swanlab.log({"IdentityPreview": preview_logs})
swanlab.finish()
```

## 7. 关键注意事项

1. **身份认同混合策略**：预生成的回答用于“打底”，再混入显式身份数据，可引导模型稳定输出身份特征，同时降低仅凭少量身份数据直接训练的过拟合风险。
2. **数据质量**：确保身份设定描述一致、逻辑连贯，避免冲突信息导致模型输出混乱。
3. **训练监控**：关注 SwanLab 中的 `train_loss` 与 `eval_loss`，若验证损失上升，应考虑减少 epoch 或加入更多多样化样本。
4. **安全合规**：医学场景需注意输出免责声明和就医建议的边界，避免误导用户。

按照上述流程，即可完成一次针对 Qwen 系列模型的身份认同微调，并通过推理生成与数据混合策略缓解过拟合问题。

## 8. 自动化脚本与示例代码

文档中的所有步骤已经整理为仓库内的可复用代码：

- `examples/qwen_identity_finetune/`：包含完整的 Python 管道实现，可直接在其他项目中导入使用。
- `scripts/run_qwen_identity_finetune.py`：一键拉起数据准备 → 人格样本生成 → 数据混合 → 训练 → 评估的脚本。

默认情况下，只需执行以下命令即可在本地启动全流程（会自动在 `./qwen_identity_workspace` 目录下产生日志与数据）：

```bash
python scripts/run_qwen_identity_finetune.py \
  --persona-name "Dr. Grace" \
  --persona-statement "你是一位善于共情、用语温和的医学专家，擅长通过先思考再回答的方式提供循序渐进的指导。" \
  --identity-mix-ratio 0.25
```

脚本会自动对原始训练集中抽样一定比例的问题进行身份推理、混入并启动训练。若需要调整训练/验证划分比例，可使用 `--train-split-ratio` 指定。

若需要额外注入全新的人格问答样本，可搭配 `--warmup-question`（可多次传入）、`--max-identity-samples` 等参数，为自动抽样之外的问题生成身份回复后追加到训练集。
