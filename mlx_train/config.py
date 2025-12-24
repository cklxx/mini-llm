from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class MiniLLMConfig:
    # Keep names aligned with `model/model_minillm.py::MiniLLMConfig`.
    dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 32768
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    num_key_value_heads: Optional[int] = 2
    vocab_size: int = 6400
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    inference_rope_scaling: bool = False
    flash_attn: bool = True

    # Custom Metal fused kernels (MLX path)
    use_metal_kernels: bool = True

    # LoRA (MLX path)
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    # MoE (not implemented in MLX path for now)
    use_moe: bool = False
    num_experts_per_tok: int = 2
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.1
    seq_aux: bool = True
    norm_topk_prob: bool = True

    def finalize(self) -> "MiniLLMConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) must be divisible by num_attention_heads({self.num_attention_heads})"
            )

        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * 8 / 3)
            self.intermediate_size = 64 * ((intermediate + 64 - 1) // 64)
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MiniLLMConfig":
        """
        Construct a config from a dict while ignoring unknown keys.

        This makes checkpoint `config.json` forward/backward compatible across
        versions (newer keys won't break older code and vice versa).
        """
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered).finalize()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def minillm_200mb() -> MiniLLMConfig:
    """
    ~200MB weights in fp16 (≈100M params), aligned with MiniLLM (LLaMA-style).
    Roughly: hidden=768, layers=15, heads=12, kv_heads=3, vocab=6400.
    """

    return MiniLLMConfig(
        hidden_size=768,
        num_hidden_layers=15,
        num_attention_heads=12,
        num_key_value_heads=3,
        vocab_size=6400,
        max_position_embeddings=32768,
        rope_theta=1_000_000.0,
        dropout=0.0,
        use_moe=False,
    ).finalize()
