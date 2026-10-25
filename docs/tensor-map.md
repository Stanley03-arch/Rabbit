# Tensor maps (Phase A)

This document freezes the **exact** placement of weights for the correctness spine.
Phase A does not claim bit-exact reconstruction of a full transformer — only that
the Rabbit pipeline (load → route → fetch → GEMV) is checkable against a known
contract.

## 1. `rabbit-ref-v1` (synthetic, frozen)

| Field | Value |
|-------|-------|
| layers | 2 |
| hidden_dim | 32 |
| vocab_size | 64 |
| n_experts | 4 |
| experts_per_token | 2 |
| expert layout | `[float32 scale][int8 HxH]` |
| dense | Opaque int8 blob (not used by MoE parity core) |

Routing and expert apply match `tools/ref_forward.py` and `rabbit_ref_forward`.
Validated by `make parity` (C vs NumPy within 1e-4).

Generator: `make ref-model` → `models/ref-v1.rabbit` (seed=1).

---

## 2. Real small HF MoE — OLMoE family

**Reference checkpoint:** `allenai/OLMoE-1B-7B-0924` (or any OLMoE-compatible
export that keeps the same tensor names).

This is a **real** public MoE. Rabbit does not yet run full OLMoE language quality
(Phase B: real SwiGLU + RoPE). The Phase A goal is an **exact, documented**
mapping from HF tensor names into the Rabbit file so conversion is reproducible
and reviewable — not heuristic-only.

### Config fields used

| HF `config.json` | Rabbit header |
|------------------|---------------|
| `hidden_size` | `hidden_dim` |
| `num_hidden_layers` | `n_layers` |
| `vocab_size` | `vocab_size` |
| `num_experts` | `n_experts` |
| `num_experts_per_tok` | `experts_per_token` |
| `intermediate_size` / `moe_intermediate_size` | `expert_dim` (metadata) |

### Exact tensor name → Rabbit placement

Patterns below use `{L}` = layer index, `{E}` = expert index.

#### Dense (loaded into RAM blob)

| HF name pattern | Role |
|-----------------|------|
| `model.embed_tokens.weight` | Token embeddings |
| `model.layers.{L}.self_attn.q_proj.weight` | Attention Q |
| `model.layers.{L}.self_attn.k_proj.weight` | Attention K |
| `model.layers.{L}.self_attn.v_proj.weight` | Attention V |
| `model.layers.{L}.self_attn.o_proj.weight` | Attention O |
| `model.layers.{L}.input_layernorm.weight` | Pre-attn norm |
| `model.layers.{L}.post_attention_layernorm.weight` | Pre-MLP norm |
| `model.layers.{L}.mlp.gate.weight` | Router (gate) |
| `model.norm.weight` | Final norm |
| `lm_head.weight` | LM head (if present; may be tied) |

Any tensor **not** matching an expert pattern below is treated as dense.
Dense weights are absmax-quantized to int8 and concatenated in **sorted name order**
into the dense blob. The engine currently treats the dense region as opaque for
the MoE parity path; attention/router use is Phase B.

#### Routed experts (mmap + LRU)

| HF name pattern | Role |
|-----------------|------|
| `model.layers.{L}.mlp.experts.{E}.gate_proj.weight` | Expert gate (SwiGLU) |
| `model.layers.{L}.mlp.experts.{E}.up_proj.weight` | Expert up |
| `model.layers.{L}.mlp.experts.{E}.down_proj.weight` | Expert down |

**Expert index** is taken from `{E}` only. Layer index is not stored separately in
the v1 file format (experts are a flat pool of size `n_experts`). When multiple
layers share the same expert id space, the converter keeps the first occurrence
of each global expert id, or the user may pass `--max-experts`.

**Phase A packing rule (documented, intentional):**

1. Collect `gate_proj`, `up_proj`, `down_proj` for expert `{E}` (missing tensors → zeros).
2. Concatenate in that order as float32, ravelled.
3. Pad or truncate to `hidden_dim * hidden_dim`.
4. Reshape to `[hidden_dim, hidden_dim]`.
5. Store as `[float32 scale][packed int4 or int8]` per `docs/file-format.md`.

This matches the current square GEMV kernels. Phase B will replace the square
collapse with a real SwiGLU expert record layout.

### Converter command (explicit arch)

```bash
pip install safetensors numpy

# After downloading OLMoE (or a subset) to a local directory:
python3 tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/OLMoE-1B-7B-0924 \
  --out models/olmoe.rabbit \
  --arch olmoe \
  --quant 4 \
  --layout square \
  --max-experts 32
```

`--arch olmoe` uses the patterns above instead of generic heuristics.

### Verification checklist (manual)

1. Header `n_experts` / `hidden_dim` match config (or the `--max-experts` limit).
2. `./rabbit probe --model models/olmoe.rabbit --prompt hi` prints a PROBE line
   (pipeline loads and runs MoE steps).
3. File size ≈ header + dense_size + n_experts × expert_size.

Bit-exact HF forward parity for full OLMoE is **out of scope** for Phase A
(requires real attention + SwiGLU + correct router). Phase A only guarantees the
mapping is exact and the engine path is exercised.

---

## 3. Other families

Mixtral / DeepSeek / Qwen-MoE remain on the **heuristic** path
(`--arch auto`, default). Adding further frozen maps is encouraged once a small
public checkpoint is chosen and the same packing rules are written down.
