# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md).

## Phase B — Engine depth (mostly complete)

Memory + SwiGLU + multi-head RoPE done.

## Phase C — Product surface (mostly complete)

OpenAI API + SSE + CORS, `rabbit stats`, `rabbit bench`, dashboard engine metrics, nucleus sampling + seed.

## Phase D — Quality spine (core complete)

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

## Next priorities (post Phase D)

Ordered by value for real-world usefulness:

1. **Live public MoE validation**  
   Finish and document the TinyBrainBot (or similar small public MoE) path so newcomers can chat with a real model, not only synthetic weights.

2. **Optional CUDA expert GEMV**  
   Accelerate the hottest path (expert matrix-vector) while keeping pure-CPU as the default. See the older open PR #3 for a starting design; needs rebase onto current main.

3. **Stronger tokenizer backends**  
   SentencePiece / HF tokenizers behind the existing RBTTOK01 API so converted models produce readable text out of the box.

4. **Clearer load / conversion errors**  
   Surface missing tensors, shape mismatches, and quant issues with actionable messages instead of generic failures.

5. **Multitier memory hierarchy**  
   Explicit VRAM / RAM / NVMe placement + better expert limiting for larger MoEs on 4–8 GB hosts.

6. **Dashboard polish**  
   React/Vite or improved Flask UI, better expert-cache visualization, and live metrics.

## Older open PRs

PRs #1–#3 (Oct 2025) sketched a “Colibri quality” sequence (validation harness, frontier models, CUDA). Much of the correctness work landed directly on `main` via Phase D. Those PRs are now stale relative to current code and should either be rebased/rewritten or closed in favor of the priorities above.

## Non-goals

- Claiming Colibrì parity without real-model transformers parity
- Copying Colibrì engines, UI, or containers
- Production chat quality on purely synthetic weights
