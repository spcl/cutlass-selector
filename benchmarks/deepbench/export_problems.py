#!/usr/bin/env python3
"""Export accepted DeepBench dense GEMMs to eval/problems.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from repo_paths import DEEPBENCH_DATA, SRC

DEFAULT_CSV = DEEPBENCH_DATA / "deepbench_compatibility.csv"
DEFAULT_OUT = DEEPBENCH_DATA / "problems.json"


def export_problems(
    csv_path: Path,
    out_path: Path,
    splits: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    df = pd.read_csv(csv_path)
    acc = df[df.accepted.astype(bool)].copy()
    if splits:
        acc = acc[acc.split.isin(splits)]
    acc = acc.sort_values(["split", "M", "N", "K", "layout"]).reset_index(drop=True)
    if limit is not None:
        acc = acc.head(limit)

    seen: set[tuple[int, int, int, str]] = set()
    problems: list[dict] = []
    for row in acc.itertuples(index=False):
        key = (int(row.M), int(row.N), int(row.K), str(row.layout).upper())
        if key in seen:
            continue
        seen.add(key)
        problems.append(
            {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "layout": key[3],
                "split": str(row.split),
            }
        )

    meta = {
        "source_csv": str(csv_path.relative_to(SRC)),
        "n_rows_read": int(len(acc)),
        "n_problems": len(problems),
        "n_duplicates_dropped": int(len(acc) - len(problems)),
        "n_unique_mnk": len({(p["M"], p["N"], p["K"]) for p in problems}),
        "splits": sorted({p["split"] for p in problems}),
        "layouts": sorted({p["layout"] for p in problems}),
    }
    payload = {"meta": meta, "problems": problems}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--split",
        default=None,
        help="comma-separated DeepBench splits (default: all accepted)",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap problem count (debug)")
    args = ap.parse_args()

    splits = None
    if args.split:
        splits = [s.strip() for s in args.split.split(",") if s.strip()]

    meta = export_problems(args.csv, args.out, splits=splits, limit=args.limit)
    print(
        f"wrote {args.out} ({meta['n_problems']} problems, "
        f"{meta['n_unique_mnk']} unique M,N,K, layouts={meta['layouts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
