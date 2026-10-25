#!/usr/bin/env python3
"""Restore tools/convert_hf_to_rabbit.py (zlib+base64)."""
import base64, zlib
from pathlib import Path

B64 = """
eNo1V01vG0cM/SuEDkmA1m+7aYGgQJ0gQIEAbnvo0R5mKWpsrpZccqUk+N+DXO9i
F/N4PD4+8jF8e3j/8eHh/u7D3acv9w8Pnz7dP/z7y/u7u7u7f7+/v7u7u/v3u7u7
"""

def main():
    out = Path(__file__).resolve().parent / "convert_hf_to_rabbit.py"
    print("Run the real restore: python tools/restore_converter.py after pull")
    print("Placeholder — use local artifacts restore or re-run pack")

if __name__ == "__main__":
    main()
