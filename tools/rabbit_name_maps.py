#!/usr/bin/env python3
"""HuggingFace MoE tensor name variants used by Rabbit converters.

Families covered (weight suffixes):
  Mixtral / Mistral-MoE:
    model.layers.{L}.block_sparse_moe.gate.weight
    model.layers.{L}.block_sparse_moe.experts.{E}.w1|w2|w3.weight

  Qwen2-MoE / Qwen3-MoE:
    model.layers.{L}.mlp.gate.weight
    model.layers.{L}.mlp.experts.{E}.gate_proj|up_proj|down_proj.weight

  DeepSeek-style (routed experts under mlp):
    model.layers.{L}.mlp.gate.weight  (or .moe.gate)
    model.layers.{L}.mlp.experts.{E}.gate_proj|up_proj|down_proj.weight

  Alternate markers also scanned by classify_tensors / collect_experts_per_layer:
    .experts.  block_sparse_moe.experts  mlp.experts  moe.experts
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# Markers that identify expert weight tensors in flat key lists
EXPERT_MARKERS: Tuple[str, ...] = (
    ".experts.",
    "block_sparse_moe.experts",
    "mlp.experts",
    "moe.experts",
    "sparse_moe.experts",
    "feed_forward.experts",
)

# Router / gate (per-layer) candidate path templates: {layer}
ROUTER_TEMPLATES: Tuple[str, ...] = (
    "model.layers.{layer}.block_sparse_moe.gate.weight",
    "model.layers.{layer}.mlp.gate.weight",
    "model.layers.{layer}.mlp.router.weight",
    "model.layers.{layer}.moe.gate.weight",
    "model.layers.{layer}.moe.router.weight",
    "model.layers.{layer}.sparse_moe.gate.weight",
    "layers.{layer}.block_sparse_moe.gate.weight",
    "layers.{layer}.mlp.gate.weight",
)

# Expert MLP triple: (gate/w1, up/w3, down/w2) path templates
# Placeholders: {layer}, {expert}
EXPERT_TRIPLE_TEMPLATES: Tuple[Tuple[str, str, str], ...] = (
    # Mixtral
    (
        "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight",
        "model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight",
        "model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight",
    ),
    (
        "layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight",
        "layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight",
        "layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight",
    ),
    # Qwen / DeepSeek-style under mlp.experts
    (
        "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
    ),
    (
        "layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        "layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        "layers.{layer}.mlp.experts.{expert}.down_proj.weight",
    ),
    # moe.experts
    (
        "model.layers.{layer}.moe.experts.{expert}.gate_proj.weight",
        "model.layers.{layer}.moe.experts.{expert}.up_proj.weight",
        "model.layers.{layer}.moe.experts.{expert}.down_proj.weight",
    ),
    (
        "model.layers.{layer}.moe.experts.{expert}.w1.weight",
        "model.layers.{layer}.moe.experts.{expert}.w3.weight",
        "model.layers.{layer}.moe.experts.{expert}.w2.weight",
    ),
    # sparse_moe
    (
        "model.layers.{layer}.sparse_moe.experts.{expert}.w1.weight",
        "model.layers.{layer}.sparse_moe.experts.{expert}.w3.weight",
        "model.layers.{layer}.sparse_moe.experts.{expert}.w2.weight",
    ),
)

# Attention / norm path variants used when probing layer tensors
ATTN_Q_TEMPLATES = (
    "model.layers.{layer}.self_attn.q_proj.weight",
    "layers.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.attention.wq.weight",
)
ATTN_K_TEMPLATES = (
    "model.layers.{layer}.self_attn.k_proj.weight",
    "layers.{layer}.self_attn.k_proj.weight",
    "model.layers.{layer}.attention.wk.weight",
)
ATTN_V_TEMPLATES = (
    "model.layers.{layer}.self_attn.v_proj.weight",
    "layers.{layer}.self_attn.v_proj.weight",
    "model.layers.{layer}.attention.wv.weight",
)
ATTN_O_TEMPLATES = (
    "model.layers.{layer}.self_attn.o_proj.weight",
    "layers.{layer}.self_attn.o_proj.weight",
    "model.layers.{layer}.attention.wo.weight",
)
INPUT_NORM_TEMPLATES = (
    "model.layers.{layer}.input_layernorm.weight",
    "layers.{layer}.input_layernorm.weight",
    "model.layers.{layer}.attention_norm.weight",
)
POST_NORM_TEMPLATES = (
    "model.layers.{layer}.post_attention_layernorm.weight",
    "layers.{layer}.post_attention_layernorm.weight",
    "model.layers.{layer}.ffn_norm.weight",
)

EMBED_TEMPLATES = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "model.wte.weight",
    "transformer.wte.weight",
)
LM_HEAD_TEMPLATES = (
    "lm_head.weight",
    "model.lm_head.weight",
)
FINAL_NORM_TEMPLATES = (
    "model.norm.weight",
    "model.model.norm.weight",
    "transformer.norm.weight",
    "norm.weight",
)


def first_present(getter: Callable[[str], Optional[object]], names: Iterable[str]):
    """Return (name, value) for the first name where getter returns non-None."""
    for name in names:
        val = getter(name)
        if val is not None:
            return name, val
    return None, None


def resolve_router_name(layer: int) -> List[str]:
    return [t.format(layer=layer) for t in ROUTER_TEMPLATES]


def resolve_expert_triple_names(layer: int, expert: int) -> List[Tuple[str, str, str]]:
    return [
        (g.format(layer=layer, expert=expert),
         u.format(layer=layer, expert=expert),
         d.format(layer=layer, expert=expert))
        for g, u, d in EXPERT_TRIPLE_TEMPLATES
    ]


def detect_expert_scheme(has_tensor: Callable[[str], bool], layer: int = 0, expert: int = 0) -> Optional[str]:
    """Return a short label for the first matching expert naming scheme."""
    labels = (
        "mixtral-block_sparse_moe",
        "mixtral-layers-prefix",
        "qwen-mlp.experts",
        "qwen-layers-prefix",
        "moe.experts-proj",
        "moe.experts-w",
        "sparse_moe",
    )
    for (g, u, d), label in zip(resolve_expert_triple_names(layer, expert), labels):
        if has_tensor(g) and has_tensor(u) and has_tensor(d):
            return label
    return None
