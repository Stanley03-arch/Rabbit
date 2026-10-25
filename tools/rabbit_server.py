#!/usr/bin/env python3
"""
Rabbit OpenAI-compatible HTTP API (Phase C)

  pip install flask
  make && make test-model
  python tools/rabbit_server.py --model models/tiny-test.rabbit --port 8080

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
  GET  /v1/stats          (engine metrics via `rabbit stats`)

Example:
  curl http://127.0.0.1:8080/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"rabbit","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

REPO_CANDIDATES = [
    Path(__file__).resolve().parent.parent,
    Path.cwd(),
    Path("/content/Rabbit"),
]


def find_repo() -> Path:
    for p in REPO_CANDIDATES:
        if (p / "src" / "main.c").is_file():
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


def run_stats(rabbit: Path, model: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [str(rabbit), "stats", "--model", str(model)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"stats failed: {proc.stderr or out}")
    # find first JSON object
    start = out.find("{")
    end = out.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError("no JSON in stats output")
    return json.loads(out[start : end + 1])


def extract_reply(stdout: str) -> str:
    lines = stdout.splitlines()
    body: List[str] = []
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
        text = chunks[-1] if chunks else (stdout.strip()[-500:] if stdout.strip() else "")
    return text or "[no generation captured]"


def run_chat(
    rabbit: Path,
    model: Path,
    message: str,
    max_tokens: int,
    temperature: float,
) -> str:
    cmd = [
        str(rabbit),
        "chat",
        "--model",
        str(model),
        "--tokens",
        str(max_tokens),
        "--temp",
        str(temperature),
    ]
    stdin = (message.strip() + "\nquit\n").encode("utf-8", errors="replace")
    proc = subprocess.run(cmd, input=stdin, capture_output=True, timeout=180)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    return extract_reply(stdout)


def messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, list):
            # multimodal stubs — take text parts only
            bits = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    bits.append(str(c.get("text") or ""))
                elif isinstance(c, str):
                    bits.append(c)
            content = " ".join(bits)
        text = str(content).strip()
        if not text:
            continue
        if role == "system":
            parts.append(f"[system] {text}")
        elif role == "assistant":
            parts.append(f"[assistant] {text}")
        else:
            parts.append(text)
    return "\n".join(parts).strip() or "hello"


def create_app(rabbit: Path, model: Path, model_id: str) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "engine": "rabbit", "model": str(model)})

    @app.get("/v1/models")
    def list_models():
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "rabbit",
                    }
                ],
            }
        )

    @app.get("/v1/stats")
    def stats():
        try:
            data = run_stats(rabbit, model)
            data["model_path"] = str(model)
            data["model_id"] = model_id
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/v1/chat/completions")
    def chat_completions():
        body = request.get_json(force=True, silent=True) or {}
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": {"message": "messages required", "type": "invalid_request_error"}}), 400

        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 48)
        temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.8)
        max_tokens = max(1, min(512, max_tokens))
        temperature = max(0.0, min(2.0, temperature))
        req_model = body.get("model") or model_id

        prompt = messages_to_prompt(messages)
        try:
            reply = run_chat(rabbit, model, prompt, max_tokens, temperature)
        except subprocess.TimeoutExpired:
            return jsonify({"error": {"message": "generation timed out", "type": "server_error"}}), 504
        except Exception as e:
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        return jsonify(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": req_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(prompt.split())),
                    "completion_tokens": max(1, len(reply.split())),
                    "total_tokens": max(1, len(prompt.split()) + len(reply.split())),
                },
            }
        )

    return app


def main():
    ap = argparse.ArgumentParser(description="Rabbit OpenAI-compatible API")
    ap.add_argument("--model", default="models/tiny-test.rabbit")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-id", default="rabbit", help="Model id exposed in /v1/models")
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

    app = create_app(rabbit, model, args.model_id)
    print(f"Rabbit binary : {rabbit}")
    print(f"Model         : {model}")
    print(f"API           : http://127.0.0.1:{args.port}/v1/chat/completions")
    print(f"Stats         : http://127.0.0.1:{args.port}/v1/stats")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
