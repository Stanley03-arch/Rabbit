#!/usr/bin/env python3
"""Offline checks for dashboard atlas UI surface."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    html = (ROOT / "web" / "index.html").read_text()
    assert "Expert Atlas" in html
    assert "view-atlas" in html
    assert "tier-row" in html

    js = (ROOT / "web" / "static" / "app.js").read_text()
    assert "renderAtlas" in js
    assert "renderTiers" in js

    py = (ROOT / "tools" / "rabbit_dashboard.py").read_text()
    assert "/api/atlas" in py
    assert "tiers" in py

    assert (ROOT / "docs" / "DASHBOARD.md").is_file()
    print("test_dashboard_atlas: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
