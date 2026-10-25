# Changelog

## [0.2.2] - 2026-10-25

### Added
- Expert cache hit/miss/insert statistics printed on shutdown
- `make smoke` non-interactive pipeline check
- `.gitignore` for build artifacts and local model files

### Changed
- More robust expert region mapping (full-file mmap)
- Version set to 0.2.2

## [0.2.1] - 2026-10-25

### Added
- CLI flags: `--vocab`, `--tokens`, `--temp`, `--help`
- `rabbit_chat_ex()` for configurable generation
- Sample word vocabulary
- Grouped attention and gated dense residual
- CONTRIBUTING.md, structured docs

## [0.2.0] - 2026-10-25

### Added
- int4 pack/dequant and GEMV path
- Expert layout `[scale][packed weights]`
- Hugging Face converter with `--quant` / `--layout`
- Byte and word tokenizer APIs

## [0.1.x] - 2026-10-25

### Added
- Initial engine: load, mmap, LRU, MoE, attention, chat loop
