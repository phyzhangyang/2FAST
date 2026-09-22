#!/usr/bin/env python3
"""Run 2FAST directly from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from twofast.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()

