# Changelog

All notable changes to Rabbit are documented in this file.

## [0.2.1] - 2026-10-25

### Added
- CLI flags: `--vocab`, `--tokens`, `--temp`, `--help`
- `rabbit_chat_ex()` for configurable generation length and temperature
- Sample word vocabulary (`models/sample_vocab.txt`)
- Grouped attention path and gated dense residual update

### Changed
- Version string set to `0.2.1-dev`
- Documentation restructured for clearer onboarding

## [0.2.0] - 2026-10-25

### Added
- int4 pack/dequant and `gemv_int4` expert path
- Expert on-disk layout: `[float32 scale][packed weights]`
- Hugging Face → Rabbit converter with `--quant` and `--layout`
- Byte-level and simple word tokenizer APIs

### Changed
- Demo model generation defaults to int4 experts

## [0.1.x] - 2026-10-25

### Added
- Initial engine skeleton: load, mmap experts, LRU cache
- MoE step with int8 GEMV
- Attention + dense residual layer step
- Synthetic test model tool
- Basic interactive chat loop
