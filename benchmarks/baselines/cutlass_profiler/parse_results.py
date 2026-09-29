#!/usr/bin/env python3
"""Extract best TFLOPS per (dtype, layout, M, N, K) from cutlass_profiler output CSV."""
import sys

import pandas as pd

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/cutlass_reference.gemm.csv"

df = pd.read_csv(csv_path)
df = df[df["Status"].str.contains("passed", case=False, na=False)]
if df.empty:
    print("No passed results found.")
    sys.exit(1)

# Derive short layout string from A/B layout columns (e.g. "t" + "n" → "TN")
df["layout"] = df["lda"].str[0].str.upper() + df["ldb"].str[0].str.upper()
df["config"] = df["A"].str.lower().str.replace("_t", "", regex=False)

rows = []
for (config, layout, m, n, k), group in df.groupby(["config", "layout", "M", "N", "K"]):
    best = group.loc[group["GFLOPs"].idxmax()]
    rows.append({
        "config": config, "layout": layout,
        "M": int(m), "N": int(n), "K": int(k),
        "tflops": best["GFLOPs"] / 1000,
        "kernel": best["Operation"],
    })

rows.sort(key=lambda r: (r["config"], r["layout"], r["M"], r["N"], r["K"]))
print(f"{'Config':<6} {'Layout':<7} {'M':>6} {'N':>6} {'K':>6}  {'TFLOPS':>8}  Kernel")
for r in rows:
    print(f"{r['config']:<6} {r['layout']:<7} {r['M']:>6} {r['N']:>6} {r['K']:>6}  {r['tflops']:>8.1f}  {r['kernel']}")
