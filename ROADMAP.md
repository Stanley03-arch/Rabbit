# Roadmap — making Rabbit serious

Rabbit is an early systems project. This document tracks what “serious” means here
and what is deliberately out of scope for now.

## Current maturity

| Layer | Status |
|-------|--------|
| File format + loader | Usable |
| Expert mmap + LRU | Usable |
| int4/int8 expert GEMV | Basic |
| Attention / FFN | Toy |
| Tokenizer | Byte / simple word |
| HF converter | Heuristic |
| Dashboard UI | Basic |
| Real LM quality | **Not claimed** |

## Near-term (engineering)

1. **Tests** — quant round-trip, cache behavior, header parse (started: `make test`)
2. **CI** — build + unit + smoke on every push
3. **Warnings-clean builds** under `-Wall -Wextra`
4. **Stable public API** in `rabbit.h` with documented ownership rules
5. **Deterministic smoke** fixtures (fixed seed)

## Medium-term (capability)

1. Faithful layout for **one** small open MoE (document exact tensor map)
2. Real tokenizer backend (SentencePiece or HF tokenizer JSON)
3. Measured memory budget report at load time
4. Optional OpenAI-compatible `/v1/chat/completions` shim for the dashboard

## Explicit non-goals (for now)

- Matching Colibrì feature-for-feature
- GPU kernels
- Multi-hundred-GB frontier checkpoints on 4 GB laptops without severe limits
- Bit-exact parity with every upstream architecture

## Definition of “serious v1”

Rabbit v1 is serious when:

- `make check` is green in CI
- One documented real model converts and runs greedily with **known** limitations
- Memory use at load is reported and bounded by flags
- Public C API is stable enough for external tools
