# 20-100M MiniGPT 语料筹备与统计

## 1. 目标与当前状态
- 面向 20–100M 参数模型，保留高质量、许可友好的通用语料；现阶段仅使用本地已下载数据，不再新增下载。
- 已完成维基百科 (parts 1–6) 与 ChineseCorpus (shards 1–31) 的转换、去重与占位符清洗，并保留原有 `pretrain_hq` 与 `sft_mini_512` 语料。
- 目前可用于预训练阶段的近似 token 量约 **4.47B**（按字符≈token 估算），距离 5B token 目标仅剩约 0.5B，可视需求追加英语/代码语料。

## 2. 可用语料摘要
| 语料 | 来源 | 许可 | 记录数 | 近似 token 数 |
| --- | --- | --- | --- | --- |
| `data/final/wiki_zh_full.simdedup.jsonl` | yuhuanstudio/wikipedia-pretrain-zh (parts 1–6) | Apache-2.0 | 3,498,017 | ≈905,316,530 (deduped) |
| `data/final/chinacorpus_full.simdedup.jsonl` | ticoAg/ChineseCorpus-Kaggle-fanti (shards 1–31) | Apache-2.0 | 15,466,499 | ≈806,153,064 (deduped) |
| `data/final/pretrain_hq.cleaned.jsonl` | minimind_dataset/pretrain_hq | 未标注 | 1,374,285 | ≈600,007,577 (cleaned) |
| `data/final/slimpajama_chunk1_part0_49.cleaned.jsonl` | cerebras/SlimPajama-627B (chunk1 files 0–49) | Apache-2.0 | 499,000 | ≈2,161,762,249 (deduped subset) |
| `data/final/sft_mini_512.cleaned.jsonl` (SFT 阶段) | minimind_dataset/sft_mini_512 | 未标注 | 1,149,770 | ≈335,939,596 (cleaned) |

更多元数据见 `data/final/datasets_manifest.json`。

## 3. 处理流水线
1. **转换**：使用 `scripts/convert_to_jsonl.py` 将 JSON 数组流式拆分成 JSONL，并保留 `title`/`text` 或 `text` 字段，自动添加语言标签。
2. **合并**：`cat data/processed/wiki_zh_part{1..6}.jsonl > wiki_zh_full.jsonl`；`cat data/processed/chinacorpus_0*.jsonl > chinacorpus_full.jsonl`。
3. **清洗**：
   ```bash
   python scripts/clean_datasets.py \
     --input data/processed/wiki_zh_full.jsonl \
     --output data/processed/wiki_zh_full.cleaned.jsonl \
     --dataset-type pretrain --dedupe --drop-placeholders --strip-think

   python scripts/clean_datasets.py \
     --input data/processed/chinacorpus_full.jsonl \
     --output data/processed/chinacorpus_full.cleaned.jsonl \
     --dataset-type pretrain --dedupe --drop-placeholders --strip-think
   ```
   - 维基百科：初步 exact 去重 + 清洗后，额外通过 SimHash 去除 32,429 条近重复，保留 3,498,017 条。
   - ChineseCorpus：初步 exact 去重 + 清洗后，再使用 SimHash 去除 46,489 条近重复，保留 15,466,499 条。
4. **统计**：`python` 脚本遍历 JSONL，按字符计数估算 token，总结输出详见上表与 `data/final/datasets_manifest.json`。

## 4. Token 规模与缺口
- 预训练阶段语料（Wiki + ChineseCorpus dedup + pretrain_hq.cleaned + SlimPajama 子集）共计 **≈4.47B tokens**。
- 若将 `sft_mini_512.cleaned` 全部文本纳入统计，总量约 **≈4.81B tokens**，但其中 0.34B 通常保留给微调。
- 与 ≥5B token 目标相比，仍需增补 **约 0.2B–0.5B tokens**，可通过追加更多 SlimPajama 分片或高质量代码数据实现。

## 5. 英文/代码语料整合方案（后续执行）
1. **扩充 SlimPajama**：已下载 chunk1 的 50 个文件（≈2.16B tokens），可继续按批下载更多分片并复用 `--state` 去重流程。
2. **代码语料（目标 ≥0.7B tokens）**：
   - 备选数据：`the-stack-smol`、`codeparrot/codeparrot-clean-train` 等，优先选择许可证友好的语言（Python/JS/C++）。
   - 清洗要点：过滤超长文件、移除含专有许可证或大段注释的样本。
3. **混合策略**：将现有语料配置采样权重，例如：
   ```json
   {
     "wiki_zh_clean": {"path": "data/final/wiki_zh_full.simdedup.jsonl", "target_tokens": 0.9e9},
     "chinacorpus_clean": {"path": "data/final/chinacorpus_full.simdedup.jsonl", "target_tokens": 0.8e9},
     "pretrain_hq": {"path": "data/final/pretrain_hq.cleaned.jsonl", "target_tokens": 0.6e9},
     "slimpajama_en": {"path": "data/final/slimpajama_chunk1_part0_49.cleaned.jsonl", "target_tokens": 2.1e9},
     "code_corpus": {"path": "TBD", "target_tokens": 0.7e9}
   }
   ```
   后续新增语料可直接补充该配置并重新混合抽样。

## 6. 后续建议
- 进一步对 ChineseCorpus 进行相似度去重（minhash/simhash）以降低 near-duplicate；必要时对 Wiki 亦可执行相同策略。
- 对清洗后的 `sft_mini_512.cleaned` 抽样质检，排查是否仍存在占位或反复追问，决定是否构建更大规模 SFT。
- 规划 tokenizer：基于当前 ≈4.81B 字符训练 SentencePiece/BPE，后续加入代码数据后再迭代更新以避免 OOV。

## 7. 近重复清理与质检
- 已实现 `scripts/dedupe_simhash.py`，基于 64-bit SimHash + LSH 的近重复过滤。对 20 万条维基样本测试，去除率约 0.3%。
- 由于全量语料较大，建议拆分为分片运行或在资源充足的环境中执行（预计 Wiki 全集约 4 小时、ChineseCorpus 约 20 小时）。
- 清洗后的 SFT (`sft_mini_512.cleaned`) 统计显示：平均轮次 4.18，助手平均 118.9 字，仍有 5.5 万条含“请提供”，8.4 万条含“无法”，提示后续需补充更高质量示例。
- 新增 `scripts/train_sentencepiece.py` 可在安装 sentencepiece 后训练初版 tokenizer。

## 8. SFT 数据优化计划
1. **模板样本筛除**：编写关键词过滤脚本（如含 “请提供”“无法”“作为一个AI” 等）并结合长度阈值，剔除模板化回复或将其权重降低。
2. **补充高质量任务**：收集多轮、多工具调用以及更具操作性的任务日志；可优先融合公开的中文开源对话集（BELLE、ShareGPT-cn 等），并手工挑选 3–5 万条高质量问答。
3. **人工重写关键样本**：对高频触发模板的类别（如摘要、提取类）组织人工或众包重写，确保回答包含具体事实、示例或步骤。
4. **RLHF/DPO 增强**：利用 `dpo.jsonl` 筛选高质量 chosen/rejected 对并扩充，结合奖励模型或规则引导，进一步压制拒答语。
5. **评估闭环**：建立关键词统计 + 人工抽检的双通道 QA；在每次数据迭代后跟踪模板词频、拒答率与平均对话长度。

## 9. 训练对接
- 预训练语料清单：`configs/data/pretrain_manifest.json`，包含权重归一化后的四个语料（Wiki、ChineseCorpus、pretrain_hq、SlimPajama 子集），`total_target_tokens` ≈ 4.4e9。
- 监督微调语料：`configs/data/sft_manifest.json`，当前指向 `sft_mini_512.cleaned.jsonl`，可在后续扩充更高质量 SFT 时更新。
- 训练脚本可按 manifest 中的 `path` 与 `weight`/`target_tokens` 实现流式采样，避免一次性合并超大单文件。

- 一键准备：`python scripts/prepare_training_data.py --workers 8` 将自动下载、转换、SimHash 去重，并生成 `data/final` 与 `configs/data` 下的 manifest。
