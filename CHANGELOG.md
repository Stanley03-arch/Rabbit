# Changelog

## [0.4.6] - 2026-10-25

### GQA + Mixtral path
- `forward_v2` separate Q/K/V with GQA (`n_kv_heads`); up to 32 layers
- Header `reserved[0:12]` stores `n_heads`, `n_kv_heads`, `head_dim`
- Converter packs real Q/K/V shapes; Mixtral expert map **w1=gate, w3=up, w2=down**
- Offline fixture `tools/create_mixtral_micro.py` + `make mixtral-micro`
- Target live model: TinyBrainBot 360M (`make live-moe`)

## [0.4.5] - 2026-10-25

### Group-scaled int4
- `rabbit_pack_int4_grouped` / dequant / GEMV (group=32)
- Converter `--quant 4` + `forward_v2` expert path

## [0.4.4] - 2026-10-25

### Real BPE tokenizer
- RBTTOK01 + `export_hf_tokenizer.py` + CLI `--tokenizer`

## [0.4.3] - 2026-10-25

### Exact parity path
- `--quant 0` float32 experts + `make exact-f32`

## [0.4.2] - 2026-10-25

### Per-layer expert tables
- `n_expert_records = L×E`

## [0.4.1] - 2026-10-25

### Phase D M1 / M2 start
- Faithful `ref-v2` + HF micro fixture

## [0.4.0] - 2026-10-25

- OpenAI API + SSE, `rabbit stats`
