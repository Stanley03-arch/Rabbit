# Engine dashboard

Lightweight **static HTML/CSS/JS** UI (not a Vite/React SPA). Keeps Rabbit easy to run on constrained hosts and Colab.

```bash
make && make test-model
pip install flask
python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
# http://127.0.0.1:7860/
```

## Views

| View | Purpose |
|------|--------|
| **Chat** | Drive `rabbit chat`, show reply |
| **Expert Atlas** | Layer × expert grid; warm/hot from recent routing heuristics |

## APIs

- `GET /api/info` — model + memory + cache metrics (`rabbit stats`)
- `GET /api/atlas` — structured atlas cells
- `POST /api/chat` — `{message, tokens, temperature}`

## Design notes

- Metrics are **engine-sourced**, not a separate “Brain” product UI.
- Atlas is an **activity map**, not a scientific visualization of neural activations.
