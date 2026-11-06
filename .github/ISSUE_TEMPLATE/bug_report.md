---
name: Bug report
about: Report a crash, incorrect result, or unexpected behavior
labels: bug
---

**Describe the bug**
A clear and concise description of what went wrong.

**Command / steps to reproduce**
```bash
# exact command(s)
./rabbit ...
```

**Model**
- Source: synthetic (tiny-test / ref-v2 / micro) / converted HF / other
- Path and approx size if relevant
- Quant: f32 / int4 / int8

**Environment**
- OS:
- Compiler (`gcc --version` or equivalent):
- Rabbit version (`./rabbit version`):

**Expected vs actual**
What you expected, and what happened instead. Paste relevant stderr/stdout.

**Additional context**
Anything else that helps (converter flags, memory budget, tokenizer, etc.).
