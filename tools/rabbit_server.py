#!/usr/bin/env python3
"""
Rabbit OpenAI-compatible HTTP API (Phase C)

  pip install flask
  make && make test-model
  python tools/rabbit_server.py --model models/tiny-test.rabbit --port 8080

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (supports "stream": true → SSE)
  GET  /v1/stats

Non-streaming:
  curl http://127.0.0.1:8080/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"rabbit","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'

Streaming (SSE):
  curl -N http://127.0.0.1:8080/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"rabbit","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stream":true}'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from flask import Flask, Response, jsonify, request, stream_with_context

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


def _is_banner_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("Rabbit ") or s.startswith("--") or s.startswith("Enter text"):
        return True
    if line.startswith("  ") or line.startswith(">"):
        return True
    if "cache hit" in s or "cache miss" in s or "hit rate" in s:
        return True
    if re.match(r"^[\-\=]{3,}$", s):
        return True
    return False


def iter_chat_tokens(
    rabbit: Path,
    model: Path,
    message: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """
    Stream tokens as the engine prints them (fputs + fflush per token).
    Falls back to chunking a full reply if live parse yields nothing.
    """
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
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        proc.stdin.write(stdin)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    # State: skip banner until we have seen the prompt echo phase, then emit.
    # Engine prints banners, then "> ", then generation pieces (often without newlines).
    buf = ""
    past_banner = False
    yielded = False
    deadline = time.time() + 180.0

    try:
        while True:
            if time.time() > deadline:
                proc.kill()
                break
            chunk = proc.stdout.read(64)
            if not chunk:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            text = chunk.decode("utf-8", errors="replace")
            buf += text

            if not past_banner:
                # Wait until after the interactive prompt marker
                if ">" in buf or "Enter text" in buf:
                    # drop everything through the last '>'
                    idx = buf.rfind(">")
                    if idx >= 0:
                        buf = buf[idx + 1 :]
                    past_banner = True
                elif len(buf) > 8000:
                    # safety: treat rest as content
                    past_banner = True
                else:
                    continue

            # Emit complete lines; keep partial trailing fragment
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if _is_banner_line(line):
                    if yielded:
                        # generation finished (blank / cache footer)
                        buf = ""
                        proc.kill()
                        break
                    continue
                piece = line
                if piece:
                    yielded = True
                    yield piece
            else:
                # no newline — engine often prints tokens without \n
                # flush safe printable run (avoid holding forever)
                if past_banner and buf and not buf.endswith(" ") and len(buf) >= 1:
                    # emit character runs that look like generation, keep last partial word optional
                    # Prefer emitting whole buffer when it has no control chars
                    if all(ord(c) >= 32 or c in "\t" for c in buf):
                        # hold short buffers briefly to coalesce tokens
                        if len(buf) >= 4 or (yielded and len(buf) >= 1):
                            # if banner-ish, skip
                            if not _is_banner_line(buf):
                                yielded = True
                                yield buf
                            buf = ""

            if proc.poll() is not None and not chunk:
                break

        # leftover
        if past_banner and buf.strip() and not _is_banner_line(buf):
            yielded = True
            yield buf.strip()
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    if not yielded:
        # Fallback: full run + word chunks
        try:
            reply = run_chat(rabbit, model, message, max_tokens, temperature)
        except Exception:
            reply = "[generation failed]"
        for word in re.findall(r"\S+\s*", reply) or [reply]:
            yield word


def messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, list):
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


def sse_pack(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def stream_completion(
    rabbit: Path,
    model: Path,
    model_id: str,
    req_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    # role chunk
    yield sse_pack(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )

    try:
        for piece in iter_chat_tokens(rabbit, model, prompt, max_tokens, temperature):
            if not piece:
                continue
            yield sse_pack(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
            )
    except Exception as e:
        yield sse_pack(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"[error: {e}]"},
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield sse_pack(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


def create_app(rabbit: Path, model: Path, model_id: str) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "engine": "rabbit", "model": str(model), "stream": True})

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
            return jsonify(
                {"error": {"message": "messages required", "type": "invalid_request_error"}}
            ), 400

        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 48)
        temperature = float(
            body.get("temperature") if body.get("temperature") is not None else 0.8
        )
        max_tokens = max(1, min(512, max_tokens))
        temperature = max(0.0, min(2.0, temperature))
        req_model = body.get("model") or model_id
        stream = bool(body.get("stream"))

        prompt = messages_to_prompt(messages)

        if stream:
            gen = stream_completion(
                rabbit, model, model_id, req_model, prompt, max_tokens, temperature
            )
            return Response(
                stream_with_context(gen),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            reply = run_chat(rabbit, model, prompt, max_tokens, temperature)
        except subprocess.TimeoutExpired:
            return jsonify(
                {"error": {"message": "generation timed out", "type": "server_error"}}
            ), 504
        except Exception as e:
            return jsonify(
                {"error": {"message": str(e), "type": "server_error"}}
            ), 500

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
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: loopback only; there is no authentication, "
        "so use 0.0.0.0 only on a trusted network)",
    )
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
    print(f"Streaming     : set \"stream\": true (SSE)")
    print(f"Stats         : http://127.0.0.1:{args.port}/v1/stats")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
