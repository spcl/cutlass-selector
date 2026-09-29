#!/usr/bin/env python3
"""Unified sweep planning CLI.

Usage:
    python src/planning/plan.py shapes [--dry-run]
    python src/planning/plan.py write --tag sweep --shapes plans/shapes.json
    python src/planning/plan.py write --tag bf16_eval --shapes plans/eval_shapes.json --exhaustive
    python src/planning/plan.py write --tag sweep_fp32_tn --dtype fp32 --layouts TN --shapes plans/shapes.json
    python src/planning/plan.py write --tag debug_bf16 --max-shapes 8 --k-default 80 --layouts TN ...
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)

    cmd = sys.argv[1]
    sys.argv = [f"plan.py {cmd}"] + sys.argv[2:]

    if cmd == "shapes":
        from plan.shapes import main as shapes_main

        shapes_main()
    elif cmd == "write":
        from plan.write import main as write_main

        write_main()
    else:
        print(f"Unknown command {cmd!r}. Use 'shapes' or 'write'.")
        sys.exit(2)


if __name__ == "__main__":
    main()
