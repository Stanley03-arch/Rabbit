# Changelog

## [0.2.3] - 2026-10-25

### Added
- Unit tests (`make test`) for int4/int8 quant and byte tokenizer
- GitHub Actions CI (build, unit tests, smoke)
- ROADMAP.md with explicit maturity and non-goals

### Fixed
- Compiler warnings in quant pack path
- `strdup` portability in tokenizer (local `xstrdup`)
- Removed unused helpers in MoE path
- Safer scale handling in expert apply

## [0.2.2] - 2026-10-25

- Cache statistics, smoke target, mmap base pointer fix, dashboard UI

## [0.2.1] - 2026-10-25

- CLI flags, docs polish, grouped attention

## [0.2.0] - 2026-10-25

- int4 path, HF converter, tokenizer APIs
