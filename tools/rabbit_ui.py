#!/usr/bin/env python3
"""
Rabbit web UI (Gradio)

Gives a simple browser chat interface similar in spirit to demo UIs
(Colibrì-style), while the heavy lifting stays in the pure-C `rabbit` binary.

Usage (local):
  pip install gradio
  python tools/rabbit_ui.py --model models/tiny-test.rabbit
  python tools/rabbit_ui.py --model models/m.rabbit --tokenizer models/m.tok

Usage (Colab):
  See notebooks/rabbit_colab_ui.ipynb
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_rabbit_bin(repo_root: Path) -> Path:
    candidates = [
        repo_root / "rabbit",
        Path.cwd() / "rabbit",
        Path("/content/Rabbit/rabbit"),
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    which = shutil.which("rabbit")
    if which:
        return Path(which)
    raise FileNotFoundError(
        "rabbit binary not found. Build with `make` in the repo root first."
    )


def run_stats(rabbit_bin: Path, model: Path) -> dict:
    try:
        proc = subprocess.run(
            [str(rabbit_bin), "stats", "--model", str(model)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (proc.stdout or "").strip()
        start, end = out.find("{"), out.rfind("}")
        if start >= 0 and end > start:
            return json.loads(out[start : end + 1])
    except Exception:
        pass
    return {}


def run_once(
    rabbit_bin: Path,
    model: Path,
    prompt: str,
    tokens: int,
    temp: float,
    vocab: str | None,
    tokenizer: Optional[Path] = None,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> str:
    """Run a single non-interactive generation and capture stdout."""
    cmd = [
        str(rabbit_bin),
        "chat",
        "--model",
        str(model),
        "--tokens",
        str(tokens),
        "--temp",
        str(temp),
        "--top-p",
        str(top_p),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if vocab:
        cmd.extend(["--vocab", vocab])
    if tokenizer and tokenizer.is_file():
        cmd.extend(["--tokenizer", str(tokenizer)])

    stdin_data = (prompt.strip() + "\nquit\n").encode("utf-8", errors="replace")

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "[error] generation timed out"

    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")

    lines = out.splitlines()
    generated: list[str] = []
    for line in lines:
        if line.startswith(">"):
            continue
        if line.startswith("Rabbit ") or line.startswith("--") or line.startswith("  "):
            continue
        if line.startswith("Enter text"):
            continue
        if line.strip() == "":
            if generated:
                break
            continue
        if not line.startswith("cache ") and "hit rate" not in line:
            generated.append(line)

    text = "\n".join(generated).strip()
    if not text:
        text = (out or err or "[no output]").strip()
        if len(text) > 2000:
            text = text[-2000:]

    note = (
        "\n\n---\n"
        "*Synthetic / early weights: output is not meaningful language. "
        "This UI validates that the engine runs end-to-end.*"
    )
    return text + note


def format_stats(st: dict) -> str:
    if not st:
        return "_No stats yet — load a model and generate once._"
    lines = [
        f"**{st.get('model_name', '?')}** · Rabbit `{st.get('version', '?')}`",
        f"- Layers: `{st.get('layers')}` · Hidden: `{st.get('hidden_dim')}` · Vocab: `{st.get('vocab_size')}`",
        f"- Experts: `{st.get('n_experts')}` (top-{st.get('experts_per_token')}) · Shared: `{st.get('n_shared_experts', 0)}`",
        f"- Quant: dense=`{st.get('dense_quant')}` expert=`{st.get('expert_quant')}` (0=f32)",
        f"- Cache slots: `{st.get('cache_slots')}` · Over budget: `{st.get('over_budget')}`",
    ]
    return "\n".join(lines)


def build_ui(
    rabbit_bin: Path,
    default_model: Path,
    share: bool,
    default_tokenizer: Optional[Path] = None,
):
    import gradio as gr

    def respond(message, history, model_path, tokens, temperature, top_p, seed, tok_path):
        model_path = (model_path or str(default_model)).strip()
        mp = Path(model_path)
        if not mp.is_file():
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"[error] model not found: {mp}"},
            ], format_stats({})

        tok: Optional[Path] = None
        if tok_path and str(tok_path).strip():
            tok = Path(str(tok_path).strip())
            if not tok.is_file():
                tok = None

        seed_i = None
        if seed is not None and str(seed).strip() != "":
            try:
                seed_i = int(seed)
            except (TypeError, ValueError):
                seed_i = None

        reply = run_once(
            rabbit_bin,
            mp,
            message,
            int(tokens),
            float(temperature),
            None,
            tok,
            float(top_p),
            seed_i,
        )
        history = history or []
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return history, format_stats(run_stats(rabbit_bin, mp))

    with gr.Blocks(
        title="Rabbit",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    ) as demo:
        gr.Markdown(
            """
            # Rabbit
            Local MoE engine — dense weights in RAM, experts streamed from disk.

            Browser UI over the C binary. With synthetic test weights, text looks random;
            the important part is that load → route → fetch → generate completes.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                model_in = gr.Textbox(
                    label="Model path",
                    value=str(default_model),
                )
                tok_in = gr.Textbox(
                    label="Tokenizer path (optional)",
                    value=str(default_tokenizer) if default_tokenizer else "",
                )
            with gr.Column(scale=2):
                tokens_in = gr.Slider(8, 256, value=48, step=8, label="Max tokens")
                temp_in = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
                top_p_in = gr.Slider(0.05, 1.0, value=1.0, step=0.05, label="Top-p (1 = off)")
                seed_in = gr.Number(label="Seed (empty = random)", value=None, precision=0)

        stats_md = gr.Markdown(format_stats(run_stats(rabbit_bin, default_model)))

        chatbot = gr.Chatbot(label="Chat", height=420, type="messages")
        msg = gr.Textbox(label="Message", placeholder="Type a prompt and press Enter")

        with gr.Row():
            clear = gr.Button("Clear")
            refresh = gr.Button("Refresh stats")

        gr.Examples(
            examples=[
                ["Hello from Rabbit"],
                ["Explain Mixture of Experts in one sentence."],
                ["ping"],
            ],
            inputs=[msg],
        )

        def on_submit(message, history, model_path, tokens, temperature, top_p, seed, tok_path):
            if not message or not message.strip():
                return history, "", stats_md.value if hasattr(stats_md, "value") else ""
            new_hist, st = respond(
                message, history, model_path, tokens, temperature, top_p, seed, tok_path
            )
            return new_hist, "", st

        msg.submit(
            on_submit,
            [msg, chatbot, model_in, tokens_in, temp_in, top_p_in, seed_in, tok_in],
            [chatbot, msg, stats_md],
        )
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

        def on_refresh(model_path):
            mp = Path((model_path or str(default_model)).strip())
            if not mp.is_file():
                return "_Model not found._"
            return format_stats(run_stats(rabbit_bin, mp))

        refresh.click(on_refresh, inputs=[model_in], outputs=[stats_md])

        gr.Markdown(
            "Repo: [Stanley03-arch/Rabbit](https://github.com/Stanley03-arch/Rabbit)"
        )

    demo.launch(share=share, debug=False)


def main():
    ap = argparse.ArgumentParser(description="Rabbit Gradio UI")
    ap.add_argument(
        "--model",
        default="models/tiny-test.rabbit",
        help="Path to .rabbit model",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Optional BPE tokenizer (.tok)",
    )
    ap.add_argument(
        "--repo",
        default=".",
        help="Repo root (where `rabbit` binary lives)",
    )
    ap.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (useful on Colab)",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    rabbit_bin = find_rabbit_bin(repo)
    model = Path(args.model)
    if not model.is_file():
        alt = repo / args.model
        if alt.is_file():
            model = alt

    tokenizer: Optional[Path] = None
    if args.tokenizer:
        tokenizer = Path(args.tokenizer)
        if not tokenizer.is_file():
            alt = repo / args.tokenizer
            if alt.is_file():
                tokenizer = alt
        if not tokenizer.is_file():
            print(f"warning: tokenizer not found: {args.tokenizer}")
            tokenizer = None

    print(f"rabbit binary : {rabbit_bin}")
    print(f"model         : {model}")
    if tokenizer:
        print(f"tokenizer     : {tokenizer}")
    build_ui(rabbit_bin, model, share=args.share, default_tokenizer=tokenizer)


if __name__ == "__main__":
    main()
