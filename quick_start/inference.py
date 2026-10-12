#!/usr/bin/env python3
"""Public PIGEONS command-line entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pigeons.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
