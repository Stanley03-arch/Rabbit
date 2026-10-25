# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Full layer path working — Attention → Dense FFN → MoE → Sampling.

## Goal

- Keep only the dense / shared parts of an MoE model in RAM
- Stream routed experts from disk on demand via `mmap`
- Extreme quantization support
- Pure C core, minimal dependencies
- Target machines with as little as **4 GB total RAM**

## Quick Start

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit

make
make test-model
./rabbit chat --model models/tiny-test.rabbit
```

Type a message. Rabbit runs a complete (synthetic) transformer-style step:

1. **Attention** over a short KV cache
2. **Dense residual FFN** using the dense blob
3. **MoE** (router → fetch experts from mmap/cache → int8 matmul)
4. **Sample** next token and continue

## What works today

| Feature                         | Status |
|---------------------------------|--------|
| Rabbit file format              | ✅     |
| Dense weights → RAM             | ✅     |
| Experts memory-mapped           | ✅     |
| Tiny LRU expert cache           | ✅     |
| Expert fetch + int8 matmul      | ✅     |
| Attention + KV cache            | ✅     |
| Dense residual path             | ✅     |
| Full layer step                 | ✅     |
| Byte-level tokenizer            | ✅     |
| Sampling + generation loop      | ✅     |
| Real model conversion           | ❌     |

## Architecture (current)

```
Token → hidden
         │
         ▼
   ┌────────────────┐
   │  Attention (KV)   │
   ├────────────────┤
   │  Dense residual   │
   ├────────────────┤
   │  MoE (streamed)   │
   └────────────────┘
         │
         ▼
      LM head → sample
```

## Next steps

1. Better quantization + fused kernels
2. Real tokenizer (SentencePiece / tiktoken style)
3. Converter for open MoE models (DeepSeek Flash, Kimi, …)
4. Multi-head / MLA-style attention

## License

Apache-2.0
