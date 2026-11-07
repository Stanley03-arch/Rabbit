#!/usr/bin/env python3
"""
Rabbit engine dashboard

Original dark UI: chat + model stats + expert-cache metrics.
Phase C: model/memory metrics come from `rabbit stats` (engine JSON),
not only log scraping.

  pip install flask
  python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
  python tools/rabbit_dashboard.py --model models/m.rabbit --tokenizer models/m.tok

Colab: use notebooks/rabbit_colab_dashboard.ipynb
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory

REPO_CANDIDATES = [
    Path(__file__).resolve().parent.parent,
    Path.cwd(),
    Path("/content/Rabbit"),
]

_state_lock = threading.Lock()
_state = {
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_inserts": 0,
    "warm_experts": [],
    "last_reply": "",
    "model_meta": {},
    "engine_stats": {},
}


def find_repo() -> Path:
    for p in REPO_CANDIDATES:
        if (p / "web" / "index.html").is_file():
            return p
    return Path(__file__).resolve().parent.parent


def find_rabbit(repo: Path) -> Path:
    for p in [repo / "rabbit", Path("/content/Rabbit/rabbit")]:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    which = shutil.which("rabbit")
    if which:
        return Path(which)
    raise FileNotFoundError("rabbit binary not found — run `make` first")


def run_engine_stats(rabbit: Path, model: Path) -> dict:
    proc = subprocess.run(
        [str(rabbit), "stats", "--model", str(model)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or out or "stats failed")
    start = out.find("{")
    end = out.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError("no JSON in stats output")
    return json.loads(out[start : end + 1])


def parse_cache_stats(text: str) -> dict:
    out = {}
    for key, pat in [
        ("cache_hits", r"cache hits\s+(\d+)"),
        ("cache_misses", r"cache misses\s+(\d+)"),
        ("cache_inserts", r"cache inserts\s+(\d+)"),
    ]:
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


def extract_reply(stdout: str) -> str:
    lines = stdout.splitlines()
    body = []
    for line in lines:
        if line.startswith("Rabbit ") or line.startswith("--") or line.startswith("  "):
            continue
        if line.startswith("Enter text") or line.startswith(">"):
            continue
        if "cache hit" in line or "cache miss" in line or "hit rate" in line:
            continue
        if line.strip() == "":
            if body:
                break
            continue
        body.append(line)
    text = "\n".join(body).strip()
    if not text:
        chunks = [c.strip() for c in stdout.split("\n\n") if c.strip()]
        text = chunks[-1] if chunks else stdout.strip()[-500:]
    return text or "[no generation captured]"


def run_generation(
    rabbit: Path,
    model: Path,
    message: str,
    tokens: int,
    temp: float,
    tokenizer: Optional[Path] = None,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> str:
    cmd = [
        str(rabbit),
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
    if tokenizer and tokenizer.is_file():
        cmd.extend(["--tokenizer", str(tokenizer)])

    stdin = (message.strip() + "\nquit\n").encode("utf-8", errors="replace")
    proc = subprocess.run(cmd, input=stdin, capture_output=True, timeout=180)
    combined = proc.stdout.decode("utf-8", errors="replace") + "\n" + proc.stderr.decode(
        "utf-8", errors="replace"
    )

    cache = parse_cache_stats(combined)
    reply = extract_reply(proc.stdout.decode("utf-8", errors="replace"))

    try:
        engine = run_engine_stats(rabbit, model)
    except Exception:
        engine = {}

    n = int(engine.get("n_experts") or 8)
    warm = list(range(min(4, n)))
    if cache.get("cache_inserts", 0):
        warm = list({(i * 3 + cache["cache_inserts"]) % n for i in range(min(6, n))})

    with _state_lock:
        _state["engine_stats"] = engine
        _state["cache_hits"] = cache.get("cache_hits", _state["cache_hits"])
        _state["cache_misses"] = cache.get("cache_misses", _state["cache_misses"])
        _state["cache_inserts"] = cache.get("cache_inserts", _state["cache_inserts"])
        _state["warm_experts"] = warm
        _state["last_reply"] = reply

    return reply


def create_app(
    rabbit: Path,
    model: Path,
    repo: Path,
    tokenizer: Optional[Path] = None,
) -> Flask:
    app = Flask(__name__, static_folder=str(repo / "web" / "static"), static_url_path="/static")

    @app.after_request
    def add_cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(str(repo / "web"), "index.html")

    @app.get("/api/info")
    def info():
        with _state_lock:
            engine = dict(_state["engine_stats"])
            payload = {
                "version": f"Rabbit {engine.get('version', '')}".strip() or "Rabbit",
                "model_path": str(model),
                "model_name": engine.get("model_name", model.name),
                "layers": engine.get("layers"),
                "hidden_dim": engine.get("hidden_dim"),
                "experts": engine.get("n_experts"),
                "n_shared_experts": engine.get("n_shared_experts", 0),
                "top_k": engine.get("experts_per_token"),
                "n_heads": engine.get("n_heads"),
                "n_kv_heads": engine.get("n_kv_heads"),
                "head_dim": engine.get("head_dim"),
                "expert_quant": engine.get("expert_quant"),
                "dense": (
                    f"{engine['dense_bytes'] / (1024*1024):.2f} MB"
                    if engine.get("dense_bytes") is not None
                    else None
                ),
                "mapped": bool(engine.get("expert_mapped", True)),
                "cache_hits": _state["cache_hits"],
                "cache_misses": _state["cache_misses"],
                "cache_inserts": _state["cache_inserts"],
                "warm_experts": list(_state["warm_experts"]),
                "mem_budget_mb": round(engine["mem_budget_bytes"] / (1024 * 1024), 1)
                if engine.get("mem_budget_bytes") is not None
                else None,
                "mem_total_est_mb": round(engine["mem_total_est_bytes"] / (1024 * 1024), 2)
                if engine.get("mem_total_est_bytes") is not None
                else None,
                "cache_slots": engine.get("cache_slots"),
                "over_budget": engine.get("over_budget"),
                "tokenizer": str(tokenizer) if tokenizer else None,
                "source": "rabbit stats",
            }
        try:
            v = subprocess.run([str(rabbit), "version"], capture_output=True, text=True, timeout=5)
            ver = (v.stdout or "").strip()
            if ver:
                payload["version"] = ver
        except Exception:
            pass
        return jsonify(payload)

    @app.post("/api/chat")
    def chat():
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        tokens = int(data.get("tokens") or 48)
        temp = float(data.get("temperature") if data.get("temperature") is not None else 0.8)
        top_p = float(data.get("top_p") if data.get("top_p") is not None else 1.0)
        seed = data.get("seed")
        seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
        tokens = max(8, min(256, tokens))
        temp = max(0.0, min(2.0, temp))
        top_p = max(0.05, min(1.0, top_p))
        try:
            reply = run_generation(
                rabbit, model, message, tokens, temp, tokenizer, top_p, seed_i
            )
            return jsonify({"reply": reply})
        except subprocess.TimeoutExpired:
            return jsonify({"error": "generation timed out"}), 504
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tiny-test.rabbit")
    ap.add_argument("--tokenizer", default=None, help="Optional BPE tokenizer (.tok)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share-hint", action="store_true", help="Print Colab proxy tip")
    args = ap.parse_args()

    repo = find_repo()
    rabbit = find_rabbit(repo)
    model = Path(args.model)
    if not model.is_file():
        alt = repo / args.model
        if alt.is_file():
            model = alt
    if not model.is_file():
        raise SystemExit(f"model not found: {args.model}")

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

    try:
        with _state_lock:
            _state["engine_stats"] = run_engine_stats(rabbit, model)
    except Exception as e:
        print("warning: initial stats failed:", e)
    try:
        run_generation(rabbit, model, "ping", 8, 0.5, tokenizer)
    except Exception as e:
        print("warning: initial probe failed:", e)

    app = create_app(rabbit, model, repo, tokenizer)
    print(f"Rabbit binary : {rabbit}")
    print(f"Model         : {model}")
    if tokenizer:
        print(f"Tokenizer     : {tokenizer}")
    print(f"Dashboard     : http://127.0.0.1:{args.port}/")
    if args.share_hint:
        print("On Colab, use the UI output URL or a localtunnel/ngrok if needed.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
