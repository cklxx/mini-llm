# MiniMind 参数量核对

本文记录了对 MiniMind `trainer/train_pretrain.py` 默认配置（`hidden_size=512`、`num_hidden_layers=8`、`use_moe=True`）的参数量推导过程，解释为何脚本在启动时会打印 `LLM可训练总参数量：95.052 百万`。

## 架构要点

- 词嵌入与输出层权重共享：`nn.Embedding(6400, 512)` 与 `nn.Linear(512, 6400, bias=False)` 共用 1 组参数。
- 自注意力模块：每层包含无偏置的 `q_proj`、`k_proj`、`v_proj`、`o_proj`，其中 `num_attention_heads=8`、`num_key_value_heads=2`、`head_dim=64`。
- 前馈层默认使用 4 专家 + 1 共享专家的 MoE 结构，每个专家都是无偏置的三线性层（`gate_proj`、`up_proj`、`down_proj`），隐藏扩展维度经过 `64` 对齐后为 `intermediate_size=1408`。
- 每个 Transformer block 还包含两组 `RMSNorm`（仅权重参数）以及 MoE 门控矩阵（`n_routed_experts × hidden_size`）。

## 逐层参数统计

| 模块 | 参数个数 | 说明 |
| --- | --- | --- |
| 词嵌入/输出共享矩阵 | 3,276,800 | `6400 × 512` |
| 单层注意力权重 | 655,360 | `512×512`（q、o） + `512×128`（k、v） |
| 单层归一化权重 | 1,024 | 两个 `RMSNorm(512)` |
| 单个前馈专家 | 2,162,688 | 三个 `512×1408`/`1408×512` 矩阵 |
| 单层 MoE 总计 | 10,815,488 | `4` 个路由专家 + `1` 个共享专家 + 门控矩阵 `4×512` |
| 单层合计 | 11,471,872 | 注意力 + 归一化 + MoE |
| 8 层堆叠 | 91,774,976 | `11,471,872 × 8` |
| 最终 `RMSNorm` | 512 | 输出前归一化 |

## 汇总

总参数量 = `91,774,976（Transformer blocks） + 3,276,800（词嵌入/输出） + 512（末端 RMSNorm） = 95,052,288`，与训练脚本打印的 `≈95.052M` 保持一致。

因此，当不调整命令行参数直接运行 `trainer/train_pretrain.py` 时，模型规模约为 95M，而非 README 中列出的 104M；后者对应的是 768 维、16 层的密集架构，需要手动覆盖默认超参方能复现。

## 26M `MiniMind2-Small` 如何复现

README 中给出的 26M 级别模型（`MiniMind2-Small`/`minimind-v1-small`）实际上使用与上节相同的 512 维、8 层骨干，只是关闭了 MoE 并沿用默认的 GQA 设置。需要注意的是，仓库当前的命令行参数写成 `parser.add_argument('--use_moe', default=True, type=bool)`，这会让 `--use_moe False` 或 `--use_moe false` 都被解释为布尔 `True`，因此脚本仍会初始化 MoE 并打印 `≈95M` 的参数量。将默认值改成 `False` 并保留 `type=bool` 也无济于事，因为 `argparse` 会把任何非空字符串（包括 `'False'`）转换成 `True`。要得到 README 所述的密集模型，需要先修正参数解析，再运行训练脚本。

1. **修正 CLI**：在 `trainer/train_pretrain.py` 中，将参数定义改成下面的形式，使其支持显式关闭 MoE 并避免 `type=bool` 带来的困惑：

   ```python
   parser.add_argument('--use_moe', dest='use_moe', action='store_true')
   parser.add_argument('--no_moe', dest='use_moe', action='store_false')
   parser.set_defaults(use_moe=True)
   ```

   如果不想修改原文件，也可以在命令行里通过 `python - <<'PY'` 等方式临时注入同等逻辑。

2. **重新运行脚本**：完成上述调整后，可在 `trainer` 目录下运行：

   ```bash
   python train_pretrain.py \
     --no_moe \
     --hidden_size 512 \
     --num_hidden_layers 8 \
     --batch_size 32 \
     --accumulation_steps 8 \
     --data_path ../dataset/pretrain_hq.jsonl
   ```

关闭 MoE 后，单层 Transformer block 的参数量下降为：

| 模块 | 参数个数 | 说明 |
| --- | --- | --- |
| 单层注意力权重 | 655,360 | `512×512`（q、o） + `512×128`（k、v） |
| 单层归一化权重 | 1,024 | 两个 `RMSNorm(512)` |
| 单层前馈网络 | 2,162,688 | 三个密集矩阵，无专家重复 |
| 单层合计 | 2,819,072 | 注意力 + 归一化 + MLP |

乘以 8 层得到 `22,552,576`，加上词嵌入/输出共享矩阵 `3,276,800` 和末端 `RMSNorm` 的 512 个参数，总计约为 `25.83M`。完成上文的 CLI 修正后，脚本会打印 `LLM可训练总参数量：25.830 百万` 左右的日志，与 README 所述“26M”一致（舍入差异）。
