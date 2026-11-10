# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md).

## Phase B — Engine depth (mostly complete)

Memory + SwiGLU + multi-head RoPE done.

Expert cache (2026-11-08): max 64 slots, RABBIT_CACHE_SLOTS override, pin shared experts, eviction/byte stats.

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
- [x] Live public MoE end-to-end (TinyBrainBot 360M) — convert + probe + chat verified 2026-11-08
  - Streaming convert of 1.53 GB safetensors → 485 MB `.rabbit` (int4) in ~5s on ~1 GB free RAM
  - Chat: 12L / H=768 / 24 experts top-4 / 32k BPE; expert cache 99%+ hit rate with 32 slots
  - Download: prefer `curl` when HF Xet leaves 0-byte placeholders
  - Dashboard parses cache evictions / bytes_copied / pinned (v0.4.18+)
  - Tokenizer export: pure tokenizer.json merges (v0.4.20); GQA ref-v2 detection

## Phase E — Scale & polish (complete)

- [x] Dashboard: cache evictions / bytes_copied / pinned (v0.4.22)
- [x] `RABBIT_REF_V2_CTX` configurable KV window for low-RAM ref-v2 (v0.4.22)
- [x] Incremental ref-v2 decode — `RabbitRefV2Ctx` prefill/step, sliding KV (v0.4.23)
- [x] Optional mmap-free expert load path for broken FUSE/CDN hosts (v0.4.24)
- [x] Larger public MoE (≥1B) convert+chat recipe — `make live-olmoe` (OLMoE-1B-7B, v0.4.25)

## Phase F — Production hardening (complete)

- [x] Multi-token streaming SSE metrics (tok/s, cache hit/s) in dashboard (v0.4.26)
- [x] Bench harness: `rabbit bench --json` with latency + cache deltas (v0.4.27)
- [x] Expert-cache warm-up: `rabbit warm` (v0.4.27)
- [x] Windows / WSL pread notes: `docs/WINDOWS.md` (v0.4.27)

## Phase G — Observability & API fidelity (in progress)

- [x] OpenAI `usage` on chat completions + final stream chunk (v0.4.27)
- [x] Machine-readable `rabbit bench --json` for CI / dashboards (v0.4.27)
- [x] Multi-model registry / dashboard model switcher (v0.4.28)
- [ ] Structured request log (JSONL) with tok/s and cache deltas
- [ ] Baseline bench table for micro + TinyBrainBot + OLMoE recipes

## Non-goals

- Claiming Colibrì parity without real-model transformers parity
- Copying Colibrì engines, UI, or containers
