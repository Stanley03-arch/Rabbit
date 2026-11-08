# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md).

## Phase B — Engine depth (mostly complete)

Memory + SwiGLU + multi-head RoPE done.

## Phase C — Product surface (mostly complete)

OpenAI API + SSE, `rabbit stats`, `rabbit bench`, dashboard engine metrics.

## Phase D — Quality spine

See [docs/PHASE_D.md](docs/PHASE_D.md).

### M1 — micro oracle
- [x] `rabbit-ref-v2` + NumPy oracle + C `forward_v2.c`
- [x] `make parity-v2` + CI
- [x] Chat uses ref-v2 path for matching models

### M2 — HF-shaped conversion
- [x] `--layout v2` exact dense + gate/up/down
- [x] Deterministic HF-like micro fixture (`make micro-hf`)
- [x] `make m2` / CI + per-layer experts + f32 path
- [x] Real BPE tokenizer (RBTTOK01)
- [x] Group-scaled int4 (group=32)
- [x] GQA dense layout (Q/K/V separate, header.reserved metadata)
- [x] Mixtral naming (`w1/w3/w2`, `block_sparse_moe`) + `make mixtral-micro` / `stream-mixtral`
- [x] Qwen2/3-MoE naming (`gate_proj/up_proj/down_proj`, `mlp.gate`) + `make stream-qwen`
- [x] DeepSeek shared experts (`mlp.shared_experts`, `n_shared_experts` in header) + `make stream-deepseek`
- [x] Shared name maps (`tools/rabbit_name_maps.py`) for Mixtral / Qwen / moe.* / sparse_moe / shared_experts
- [x] Streaming converter multi-shard + quant 0|4 (`tools/convert_hf_stream.py`)
- [ ] Live public MoE end-to-end (TinyBrainBot 360M) — run `make live-moe` on a machine with HF access
  - Hardened (2026-11-08): download size validation + curl fallback for Xet 0-byte files; converter warns on large shards / low RAM
  - Streaming reader (2026-11-08): convert_hf_stream no longer mmaps the full checkpoint; peak RAM ≈ largest tensor
  - Needs enough free RAM for the largest tensors during convert (embed/lm_head ~100 MB for 360M model)
  - Target still pending a successful convert + probe on a host that can download the 1.5 GB checkpoint

## Non-goals

- Claiming Colibrì parity without real-model transformers parity
- Copying Colibrì engines, UI, or containers
