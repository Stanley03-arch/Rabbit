#!/usr/bin/env python3
"""
convert_hf_to_rabbit.py

Convert open MoE models into the Rabbit binary format.

Layouts:
  square — legacy H×H expert (lossy)
  flat   — gate/up/down ordered for SwiGLU kernel (still opaque dense)
  v2     — Phase D faithful dense + exact gate/up/down (no square crop)
           packs per-layer experts: n_expert_records = n_layers × n_experts

Expert quant:
  0 — pure float32 (exact parity vs unquantized weights)
  4 / 8 — scaled int4 / int8

Usage:
  python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit
  python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --layout v2 --quant 0
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RABBIT_MAGIC = b"RABBIT01"
HEADER_SIZE = struct.calcsize(
    "8s" + "I" * 8 + "Q" * 3 + "I" * 2 + "64s" + "128s"
)


def write_header(f, **kw) -> None:
    name_bytes = kw["name"].encode("utf-8")[:63].ljust(64, b"\0")
    reserved = b"\0" * 128
    n_records = kw.get("n_expert_records", kw["n_experts"])
    total_experts_size = n_records * kw["expert_size"]
    header = struct.pack(
        "<8sIIIIIIIIQQQII64s128s",
        RABBIT_MAGIC,
        1,
        HEADER_SIZE,
        kw["n_layers"],
        kw["hidden_dim"],
        kw["vocab_size"],
        kw["n_experts"],
        kw["experts_per_token"],
        kw["expert_dim"],
        kw["dense_size"],
        kw["expert_size"],
        total_experts_size,
        kw["dense_quant"],
        kw["expert_quant"],
        name_bytes,
        reserved,
    )
    f.write(header)


def quant_int8(arr: np.ndarray) -> Tuple[bytes, float]:
    f = arr.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 127.0 + 1e-8)
    q = np.clip(np.round(f / scale), -128, 127).astype(np.int8)
    return q.tobytes(), scale


def quant_int4(arr: np.ndarray) -> Tuple[bytes, float]:
    f = arr.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 7.0 + 1e-8)
    q = np.clip(np.round(f / scale), -8, 7).astype(np.int32)
    n = q.shape[0]
    out = bytearray((n + 1) // 2)
    for i in range(0, n, 2):
        lo = int(q[i]) + 8
        hi = int(q[i + 1]) + 8 if i + 1 < n else 0
        out[i // 2] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return bytes(out), scale


def pack_with_scale(arr: np.ndarray, quant: int) -> bytes:
    if quant == 4:
        payload, scale = quant_int4(arr)
    else:
        payload, scale = quant_int8(arr)
    return struct.pack("<f", scale) + payload


def build_demo_model(out_path: Path, hidden_dim: int = 128, n_experts: int = 16, quant: int = 4) -> None:
    n_layers = 6
    vocab_size = 256
    experts_per_token = 2
    expert_dim = hidden_dim * 2

    dense = np.random.randn(hidden_dim * 256).astype(np.float32)
    dense_blob = pack_with_scale(dense, 8)

    expert_blobs = []
    for e in range(n_experts):
        mat = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        mat += e * 0.01
        expert_blobs.append(pack_with_scale(mat, quant if quant else 8))

    expert_size = max(len(b) for b in expert_blobs)
    expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_header(
            f,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            n_experts=n_experts,
            experts_per_token=experts_per_token,
            expert_dim=expert_dim,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=8,
            expert_quant=quant if quant else 8,
            name="demo-moe",
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    print(f"[demo] {out_path}  quant={quant}  experts={n_experts}  hidden={hidden_dim}")


def load_hf_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def collect_tensors(model_dir: Path) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}

    npz = model_dir / "model.npz"
    if npz.is_file():
        print(f"[load] {npz.name}")
        data = np.load(npz)
        for key in data.files:
            tensors[key] = data[key]
        return tensors

    try:
        from safetensors import safe_open
    except ImportError:
        print("pip install safetensors numpy  (or provide model.npz)", file=sys.stderr)
        sys.exit(1)

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors or model.npz in {model_dir}")
    for path in files:
        print(f"[load] {path.name}")
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def classify_tensors(tensors: Dict[str, np.ndarray]):
    dense = {}
    experts: Dict[int, Dict[str, np.ndarray]] = {}

    expert_markers = (
        ".experts.",
        "block_sparse_moe.experts",
        "mlp.experts",
        "moe.experts",
    )

    for name, arr in tensors.items():
        lower = name.replace("/", ".")
        is_expert = any(m in lower for m in expert_markers)

        if not is_expert:
            dense[name] = arr
            continue

        parts = lower.split(".")
        idx = None
        for i, p in enumerate(parts):
            if p in ("experts", "expert") and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                break

        if idx is None:
            dense[name] = arr
            continue

        experts.setdefault(idx, {})[name] = arr

    return dense, experts


def collect_experts_per_layer(
    tensors: Dict[str, np.ndarray],
    n_layers: int,
    n_experts: int,
) -> List[List[Dict[str, np.ndarray]]]:
    per_layer: List[List[Dict[str, np.ndarray]]] = [
        [{} for _ in range(n_experts)] for _ in range(n_layers)
    ]

    expert_markers = (
        ".experts.",
        "block_sparse_moe.experts",
        "mlp.experts",
        "moe.experts",
    )

    def _parse_layer_expert(name: str) -> Optional[Tuple[int, int]]:
        lower = name.replace("/", ".").lower()
        if not any(m in lower for m in expert_markers):
            return None
        parts = lower.split(".")
        layer_idx: Optional[int] = None
        expert_idx: Optional[int] = None
        for i, p in enumerate(parts):
            if p in ("layers", "layer", "h") and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
            if p in ("experts", "expert") and i + 1 < len(parts) and parts[i + 1].isdigit():
                expert_idx = int(parts[i + 1])
        if expert_idx is None:
            return None
        return (layer_idx if layer_idx is not None else -1, expert_idx)

    shared: Dict[int, Dict[str, np.ndarray]] = {}
    saw_layered = False

    for name, arr in tensors.items():
        parsed = _parse_layer_expert(name)
        if parsed is None:
            continue
        layer_idx, expert_idx = parsed
        if expert_idx < 0 or expert_idx >= n_experts:
            continue
        if layer_idx >= 0:
            if layer_idx >= n_layers:
                continue
            per_layer[layer_idx][expert_idx][name] = arr
            saw_layered = True
        else:
            shared.setdefault(expert_idx, {})[name] = arr

    if not saw_layered and shared:
        for L in range(n_layers):
            for e in range(n_experts):
                if e in shared:
                    per_layer[L][e] = dict(shared[e])
        print(
            f"[v2] shared expert pool replicated across {n_layers} layers "
            f"({n_experts} experts → {n_layers * n_experts} records)"
        )
    else:
        filled = sum(1 for L in range(n_layers) for e in range(n_experts) if per_layer[L][e])
        print(f"[v2] per-layer experts: {filled}/{n_layers * n_experts} slots populated")

    return per_layer


def pack_dense(dense: Dict[str, np.ndarray], quant: int = 8) -> bytes:
    pieces = []
    for name in sorted(dense.keys()):
        pieces.append(pack_with_scale(dense[name], quant))
    return b"".join(pieces)


def _prefer_name(name: str) -> int:
    n = name.lower()
    if "down_proj" in n or "w2" in n:
        return 100
    if "up_proj" in n or "w3" in n or "w1" in n:
        return 50
    if "gate_proj" in n or "gate_up" in n:
        return 40
    return 10


def _order_key(name: str) -> int:
    n = name.lower()
    if "gate" in n and "proj" in n:
        return 0
    if "up" in n or "w1" in n or "w3" in n:
        return 1
    if "down" in n or "w2" in n:
        return 2
    return 3


def pack_one_expert(expert_tensors: Dict[str, np.ndarray], quant: int, layout: str,
                    hidden_dim: int) -> bytes:
    q = quant if quant else 8
    if layout == "square":
        candidates = []
        for name, arr in expert_tensors.items():
            a = np.asarray(arr)
            if a.ndim == 2:
                candidates.append((name, a))
        if candidates:
            candidates.sort(key=lambda x: (_prefer_name(x[0]), x[1].size), reverse=True)
            mat = candidates[0][1]
            flat = mat.astype(np.float32).ravel()
            need = hidden_dim * hidden_dim
            if flat.size < need:
                flat = np.pad(flat, (0, need - flat.size))
            else:
                flat = flat[:need]
            return pack_with_scale(flat.reshape(hidden_dim, hidden_dim), q)
        flat = np.concatenate([np.asarray(v).astype(np.float32).ravel() for v in expert_tensors.values()])
        need = hidden_dim * hidden_dim
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size))
        return pack_with_scale(flat[:need].reshape(hidden_dim, hidden_dim), q)

    ordered = sorted(expert_tensors.items(), key=lambda kv: _order_key(kv[0]))
    pieces = [pack_with_scale(np.asarray(v), q) for _, v in ordered]
    return b"".join(pieces)


def _find(tensors: Dict[str, np.ndarray], *substrings: str) -> Optional[np.ndarray]:
    for name, arr in tensors.items():
        n = name.lower()
        if all(s in n for s in substrings):
            return np.asarray(arr, dtype=np.float32)
    return None


def _find_layer(tensors: Dict[str, np.ndarray], layer: int, *substrings: str) -> Optional[np.ndarray]:
    markers = (f"layers.{layer}.", f"layer.{layer}.", f".{layer}.")
    for name, arr in tensors.items():
        n = name.lower()
        if not any(m in n for m in markers):
            continue
        if all(s in n for s in substrings):
            return np.asarray(arr, dtype=np.float32)
    return None


def pack_v2_dense(tensors: Dict[str, np.ndarray], n_layers: int, hidden: int,
                  vocab: int, n_experts: int) -> bytes:
    embed = _find(tensors, "embed")
    if embed is None:
        embed = _find(tensors, "wte")
    if embed is None:
        raise RuntimeError("v2 layout: missing embedding weights")
    if embed.ndim != 2:
        raise RuntimeError(f"embed shape {embed.shape}")
    if embed.shape[0] == hidden and embed.shape[1] == vocab:
        embed = embed.T
    if embed.shape != (vocab, hidden):
        e = np.zeros((vocab, hidden), dtype=np.float32)
        v = min(vocab, embed.shape[0])
        h = min(hidden, embed.shape[1])
        e[:v, :h] = embed[:v, :h]
        embed = e

    lm = _find(tensors, "lm_head")
    if lm is None:
        lm = embed.copy()
    lm = np.asarray(lm, dtype=np.float32)
    if lm.shape == (vocab, hidden):
        lm = lm.T
    if lm.shape != (hidden, vocab):
        m = np.zeros((hidden, vocab), dtype=np.float32)
        h = min(hidden, lm.shape[0] if lm.ndim == 2 else 0)
        v = min(vocab, lm.shape[1] if lm.ndim == 2 else 0)
        if lm.ndim == 2:
            m[:h, :v] = lm[:h, :v]
        lm = m

    parts: List[bytes] = [embed.astype(np.float32).tobytes(), lm.astype(np.float32).tobytes()]

    for L in range(n_layers):
        rms_attn = _find_layer(tensors, L, "input_layernorm")
        if rms_attn is None:
            rms_attn = _find_layer(tensors, L, "attention_norm")
        if rms_attn is None:
            rms_attn = np.ones(hidden, dtype=np.float32)
        rms_attn = rms_attn.reshape(-1)[:hidden]
        if rms_attn.size < hidden:
            rms_attn = np.pad(rms_attn, (0, hidden - rms_attn.size), constant_values=1.0)

        rms_ffn = _find_layer(tensors, L, "post_attention_layernorm")
        if rms_ffn is None:
            rms_ffn = _find_layer(tensors, L, "ffn_norm")
        if rms_ffn is None:
            rms_ffn = np.ones(hidden, dtype=np.float32)
        rms_ffn = rms_ffn.reshape(-1)[:hidden]
        if rms_ffn.size < hidden:
            rms_ffn = np.pad(rms_ffn, (0, hidden - rms_ffn.size), constant_values=1.0)

        q = _find_layer(tensors, L, "q_proj")
        k = _find_layer(tensors, L, "k_proj")
        v = _find_layer(tensors, L, "v_proj")
        o = _find_layer(tensors, L, "o_proj")
        if q is None or k is None or v is None:
            qkv = _find_layer(tensors, L, "qkv")
            if qkv is not None and qkv.shape[0] >= 3 * hidden:
                q = qkv[:hidden]
                k = qkv[hidden : 2 * hidden]
                v = qkv[2 * hidden : 3 * hidden]
            else:
                q = k = v = np.eye(hidden, dtype=np.float32) * 0.02
        if o is None:
            o = np.eye(hidden, dtype=np.float32) * 0.02

        def square(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=np.float32)
            out = np.zeros((hidden, hidden), dtype=np.float32)
            r = min(hidden, m.shape[0])
            c = min(hidden, m.shape[1] if m.ndim > 1 else 0)
            if m.ndim == 2:
                out[:r, :c] = m[:r, :c]
            return out

        q, k, v, o = square(q), square(k), square(v), square(o)
        w_qkv = np.concatenate([q, k, v], axis=0)

        router = _find_layer(tensors, L, "mlp.gate")
        if router is None:
            router = _find_layer(tensors, L, "block_sparse_moe.gate")
        if router is None:
            router = _find_layer(tensors, L, "router")
        if router is None:
            router = 0.05 * np.random.randn(n_experts, hidden).astype(np.float32)
        router = np.asarray(router, dtype=np.float32)
        if router.ndim == 1:
            router = router.reshape(n_experts, -1)
        if router.shape[0] == hidden and router.shape[1] == n_experts:
            router = router.T
        rr = np.zeros((n_experts, hidden), dtype=np.float32)
        r0 = min(n_experts, router.shape[0])
        c0 = min(hidden, router.shape[1])
        rr[:r0, :c0] = router[:r0, :c0]

        parts.extend(
            [
                rms_attn.astype(np.float32).tobytes(),
                rms_ffn.astype(np.float32).tobytes(),
                w_qkv.astype(np.float32).tobytes(),
                o.astype(np.float32).tobytes(),
                rr.astype(np.float32).tobytes(),
            ]
        )

    return b"".join(parts)


def pack_v2_expert(expert_tensors: Dict[str, np.ndarray], quant: int,
                   hidden: int, inter: int) -> bytes:
    """Exact gate [I,H], up [I,H], down [H,I]. quant=0 → pure f32."""
    gate = up = down = None
    for name, arr in expert_tensors.items():
        n = name.lower()
        a = np.asarray(arr, dtype=np.float32)
        if "gate" in n and "proj" in n:
            gate = a
        elif "up" in n or n.endswith("w1") or ".w1." in n or ".w3." in n:
            if "gate" not in n:
                up = a
        elif "down" in n or n.endswith("w2") or ".w2." in n:
            down = a

    def fit(a: Optional[np.ndarray], rows: int, cols: int) -> np.ndarray:
        out = np.zeros((rows, cols), dtype=np.float32)
        if a is None:
            return out
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 2:
            a = a.reshape(rows, -1) if a.size >= rows else a.reshape(-1, cols)
        r = min(rows, a.shape[0])
        c = min(cols, a.shape[1])
        out[:r, :c] = a[:r, :c]
        return out

    gate = fit(gate, inter, hidden)
    up = fit(up, inter, hidden)
    down = fit(down, hidden, inter)
    if quant == 0:
        return (
            gate.astype(np.float32).tobytes()
            + up.astype(np.float32).tobytes()
            + down.astype(np.float32).tobytes()
        )
    return pack_with_scale(gate, quant) + pack_with_scale(up, quant) + pack_with_scale(down, quant)


def convert_hf(model_dir: Path, out_path: Path, max_experts: Optional[int],
               quant: int, layout: str) -> None:
    config = load_hf_config(model_dir)
    print("[config]", {k: config.get(k) for k in (
        "model_type", "hidden_size", "num_hidden_layers", "num_local_experts",
        "num_experts_per_tok", "vocab_size", "moe_intermediate_size",
        "n_routed_experts", "num_experts",
    ) if k in config})

    tensors = collect_tensors(model_dir)
    print(f"[info] {len(tensors)} tensors")

    dense, experts = classify_tensors(tensors)
    print(f"[info] dense={len(dense)}  expert_groups={len(experts)}")

    hidden_dim = int(config.get("hidden_size") or config.get("n_embd") or 1024)
    n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 12)
    vocab_size = int(config.get("vocab_size") or 32000)
    experts_per_token = int(config.get("num_experts_per_tok") or config.get("moe_top_k") or 2)
    expert_dim = int(config.get("moe_intermediate_size") or config.get("intermediate_size") or hidden_dim * 2)

    expert_ids = sorted(experts.keys())
    if max_experts is not None and len(expert_ids) > max_experts:
        print(f"[info] limiting experts {len(expert_ids)} → {max_experts}")
        expert_ids = expert_ids[:max_experts]

    n_experts = len(expert_ids) if expert_ids else int(
        config.get("num_local_experts") or config.get("num_experts") or config.get("n_routed_experts") or 8
    )

    if layout == "v2":
        dense_blob = pack_v2_dense(tensors, n_layers, hidden_dim, vocab_size, n_experts)
        per_layer = collect_experts_per_layer(tensors, n_layers, n_experts)
        expert_blobs = []
        for L in range(n_layers):
            for e in range(n_experts):
                expert_blobs.append(
                    pack_v2_expert(
                        per_layer[L][e],
                        quant=quant,
                        hidden=hidden_dim,
                        inter=expert_dim,
                    )
                )
        if not expert_blobs:
            raise RuntimeError("v2 layout requires expert tensors")
        expert_size = max(len(b) for b in expert_blobs)
        expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]
        dense_quant = 0
        name = "micro-hf-v2" if config.get("rabbit_fixture") else model_dir.name[:63]
        n_expert_records = n_layers * n_experts
        print(
            f"[v2] packing {n_expert_records} expert records "
            f"(layers={n_layers} × experts={n_experts}, quant={quant})"
        )
    else:
        dense_blob = pack_dense(dense, quant=8)
        expert_blobs = []
        if expert_ids:
            for idx in expert_ids:
                blob = pack_one_expert(experts[idx], quant=quant, layout=layout, hidden_dim=hidden_dim)
                expert_blobs.append(blob)
            expert_size = max(len(b) for b in expert_blobs)
            expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]
        else:
            print("[warn] no experts detected — creating placeholder square experts")
            expert_size = 4 + (hidden_dim * hidden_dim if quant != 4 else hidden_dim * ((hidden_dim + 1) // 2))
            expert_blobs = [b"\0" * expert_size for _ in range(n_experts)]
        dense_quant = 8
        name = model_dir.name[:63]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_header(
            f,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            n_experts=n_experts,
            n_expert_records=(n_layers * n_experts) if layout == "v2" else n_experts,
            experts_per_token=experts_per_token,
            expert_dim=expert_dim,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=dense_quant,
            expert_quant=quant,
            name=name,
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    total = out_path.stat().st_size
    print(f"[done] {out_path}")
    print(f"       dense={len(dense_blob)/1024:.1f} KB  experts={n_experts} x {expert_size/1024:.1f} KB")
    print(f"       quant={quant}  layout={layout}  total={total/1024:.1f} KB")
    if layout == "v2":
        print(f"       forward: rabbit ref-v2 (./rabbit chat --model {out_path})")
        print(f"       expert records: {n_layers * n_experts} (per-layer)")
    else:
        print(f"\nRun:  ./rabbit chat --model {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-experts", type=int, default=None)
    ap.add_argument("--quant", type=int, choices=[0, 4, 8], default=4,
                    help="Expert quant: 0=float32 exact, 4/8=int")
    ap.add_argument(
        "--layout",
        choices=["square", "flat", "v2"],
        default="square",
        help="square=H×H; flat=ordered SwiGLU pieces; v2=faithful dense+per-layer experts",
    )
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-experts", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    if args.demo:
        build_demo_model(out, hidden_dim=args.hidden_dim, n_experts=args.n_experts, quant=args.quant)
        return
    if not args.model_dir:
        ap.error("Need --demo or --model-dir")
    convert_hf(Path(args.model_dir), out, args.max_experts, args.quant, args.layout)


if __name__ == "__main__":
    main()
