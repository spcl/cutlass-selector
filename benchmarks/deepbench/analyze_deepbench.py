#!/usr/bin/env python3
"""Feasibility analysis: DeepBench dense GEMMs vs paper CUTLASS selector."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from repo_paths import BENCHMARKS, DEEPBENCH_DATA, EVAL_OUT, PAPER, REPO_ROOT, SRC

DEEPBENCH = BENCHMARKS / "extern" / "deepbench"
GEMM_H = DEEPBENCH / "code" / "kernels" / "gemm_problems.h"
OUT = DEEPBENCH_DATA

sys.path.insert(0, str(SRC / "autotuner"))
from config_space import enumerate_candidates  # noqa: E402

ALIGN = 8
MIN_EVAL_DIM = 32  # catalogue excludes smaller legacy RNN / LM projection shapes
LAYOUTS = ("TN", "TT", "NN", "NT")
TMA_CHECKS = {
    "TN": ("K", "M"),
    "TT": ("K", "N", "M"),
    "NN": ("M", "K"),
    "NT": ("M", "N"),
}


def parse_gemm_header(path: Path) -> pd.DataFrame:
    text = path.read_text()
    rows = []
    current_split = None
    for line in text.splitlines():
        mset = re.search(r"(\w+_set)\s*=", line)
        if mset:
            name = mset.group(1)
            if name == "training_set":
                current_split = "train"
            elif name == "inference_server_set":
                current_split = "inference_server"
            elif name == "inference_device_set":
                current_split = "inference_device"
            else:
                current_split = name
            continue
        mt = re.search(
            r"make_tuple\((\d+),\s*(\d+),\s*(\d+),\s*(true|false),\s*(true|false)\)",
            line,
        )
        if mt and current_split:
            m, n, k = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
            a_t = mt.group(4) == "true"
            b_t = mt.group(5) == "true"
            layout = ("T" if a_t else "N") + ("T" if b_t else "N")
            rows.append(
                {
                    "split": current_split,
                    "M": m,
                    "N": n,
                    "K": k,
                    "a_transpose": a_t,
                    "b_transpose": b_t,
                    "layout": layout,
                    "original_dtype_note": "fp32/half train; int8 inference in DeepBench README",
                    "source_file": str(path.relative_to(SRC)),
                    "workload_label": "",
                }
            )
    return pd.DataFrame(rows)


def tma_ok(m: int, n: int, k: int, layout: str) -> tuple[bool, str]:
    dims = {"M": m, "N": n, "K": k}
    bad = [d for d in TMA_CHECKS[layout] if dims[d] % ALIGN != 0]
    if bad:
        return False, f"tma_align:{','.join(bad)}%8!=0"
    return True, ""


def catalog_size(layout: str) -> int:
    return len(enumerate_candidates(0, 0, 0, layout, "bf16"))


_CATALOG = {lay: catalog_size(lay) for lay in LAYOUTS}


def compat_row(row: pd.Series) -> dict:
    m, n, k, layout = int(row.M), int(row.N), int(row.K), row.layout
    ok, reason = tma_ok(m, n, k, layout)
    if layout not in LAYOUTS:
        return {
            "accepted": False,
            "num_valid_configs": 0,
            "rejection_reason": f"unsupported_layout:{layout}",
        }
    if not ok:
        return {"accepted": False, "num_valid_configs": 0, "rejection_reason": reason}
    if min(m, n, k) < 32:
        # Catalogue exists but many tiles may fail can_implement; still count as accepted
        # with note — primary gate is TMA alignment for BF16.
        reason = "tma_ok;min_dim<32_may_reduce_feasible_tiles"
    else:
        reason = ""
    return {
        "accepted": True,
        "num_valid_configs": _CATALOG[layout],
        "rejection_reason": reason,
    }


def load_overlap_sets() -> dict[str, set[tuple]]:
    out: dict[str, set] = {
        "train_shapes": set(),
        "eval68_shapes": set(),
        "eval68_shape_layout": set(),
        "broad_shapes": set(),
        "broad_shape_layout": set(),
        "broad_source": "",
    }
    feat_candidates = [
        PAPER / "features.parquet",
        SRC / "model" / "artifacts" / "paper" / "features.parquet",
    ]
    for fp in feat_candidates:
        if not fp.is_file():
            continue
        df = pd.read_parquet(fp, columns=["M", "N", "K", "layout", "split"])
        tr = df[df.split == "train"]
        out["train_shapes"] = set(zip(tr.M, tr.N, tr.K))
        ev = df[df.split == "eval"]
        out["eval68_shapes"] = set(zip(ev.M, ev.N, ev.K))
        out["eval68_shape_layout"] = set(zip(ev.M, ev.N, ev.K, ev.layout.str.upper()))
        break

    import json

    broad_candidates = [
        EVAL_OUT / "shapes.json",  # 2k held-out shapes → 8k problems
        PAPER / "eval" / "shapes.json",
    ]
    for broad_path in broad_candidates:
        if not broad_path.is_file():
            continue
        shapes = json.loads(broad_path.read_text())
        if isinstance(shapes, list) and shapes and isinstance(shapes[0], dict):
            out["broad_shapes"] = {(int(s["M"]), int(s["N"]), int(s["K"])) for s in shapes}
        elif isinstance(shapes, list):
            out["broad_shapes"] = {(int(s[0]), int(s[1]), int(s[2])) for s in shapes}
        out["broad_source"] = str(broad_path.relative_to(REPO_ROOT))
        break
    if out["broad_shapes"]:
        out["broad_shape_layout"] = {
            (m, n, k, lay) for (m, n, k) in out["broad_shapes"] for lay in LAYOUTS
        }
    return out


def summarize_shapes(df: pd.DataFrame) -> dict:
    pd.concat([df.M, df.N, df.K])
    skinny = ((df.M / df.N >= 4) | (df.N / df.M >= 4) | (df.K <= df[["M", "N"]].min(axis=1) / 4)).sum()
    any_lt32 = ((df.M < 32) | (df.N < 32) | (df.K < 32)).sum()
    sig = df.drop_duplicates(["M", "N", "K", "layout"])
    return {
        "total": len(df),
        "unique_signatures": len(sig),
        "train": int((df.split == "train").sum()),
        "inference_server": int((df.split == "inference_server").sum()),
        "inference_device": int((df.split == "inference_device").sum()),
        "layout_counts": df.layout.value_counts().to_dict(),
        "M_min": int(df.M.min()),
        "M_max": int(df.M.max()),
        "M_median": float(df.M.median()),
        "N_min": int(df.N.min()),
        "N_max": int(df.N.max()),
        "N_median": float(df.N.median()),
        "K_min": int(df.K.min()),
        "K_max": int(df.K.max()),
        "K_median": float(df.K.median()),
        "frac_div8": float(((df.M % 8 == 0) & (df.N % 8 == 0) & (df.K % 8 == 0)).mean()),
        "frac_div16": float(((df.M % 16 == 0) & (df.N % 16 == 0) & (df.K % 16 == 0)).mean()),
        "frac_div32": float(((df.M % 32 == 0) & (df.N % 32 == 0) & (df.K % 32 == 0)).mean()),
        "frac_div64": float(((df.M % 64 == 0) & (df.N % 64 == 0) & (df.K % 64 == 0)).mean()),
        "skinny_count": int(skinny),
        "any_dim_lt32": int(any_lt32),
    }


def overlap_report(df: pd.DataFrame, sets: dict[str, set]) -> dict:
    accepted = df[df.accepted]
    shape_keys = list(zip(accepted.M, accepted.N, accepted.K))
    sl_keys = list(zip(accepted.M, accepted.N, accepted.K, accepted.layout))
    near = 0
    if sets["train_shapes"]:
        for m, n, k in zip(accepted.M, accepted.N, accepted.K):
            for tm, tn, tk in sets["train_shapes"]:
                if max(abs(m - tm) / max(tm, 1), abs(n - tn) / max(tn, 1), abs(k - tk) / max(tk, 1)) < 0.05:
                    near += 1
                    break
    return {
        "exact_shape_overlap_train": sum(1 for s in shape_keys if s in sets["train_shapes"]),
        "exact_shape_overlap_eval68": sum(1 for s in shape_keys if s in sets["eval68_shapes"]),
        "exact_shape_layout_overlap_eval68": sum(1 for s in sl_keys if s in sets["eval68_shape_layout"]),
        "exact_shape_overlap_broad": sum(1 for s in shape_keys if s in sets["broad_shapes"]),
        "exact_shape_layout_overlap_broad": sum(1 for s in sl_keys if s in sets["broad_shape_layout"]),
        "near_shape_overlap_train_5pct": near,
        "accepted_problems": len(accepted),
    }


def write_analysis_md(shapes: dict, compat: pd.DataFrame, overlap: dict, sets: dict) -> None:
    acc = compat[compat.accepted]
    rej = compat[~compat.accepted]
    lines = [
        "# DeepBench external GEMM case study — feasibility analysis",
        "",
        "## Source of shapes",
        "",
        "Official DeepBench **dense GEMM** lists (executable benchmark definitions):",
        "",
        f"- `{GEMM_H.relative_to(SRC)}` — `training_set` (160), `inference_server_set` (75), `inference_device_set` (13)",
        "- Human-readable specs: `deepbench/DeepBenchKernels_train.xlsx`, `deepbench/DeepBenchKernels_inference.xlsx` (application labels; a few tuples differ from the C++ header)",
        "- README: `deepbench/README.md` (GEMM semantics, precision requirements, methodology)",
        "",
        "We use the **C++ headers** as ground truth (what DeepBench actually runs). Sparse GEMM (`sparse_gemm_problems.h`) is excluded.",
        "",
        "Layout mapping: DeepBench `(a_transpose, b_transpose)` → cuBLAS op codes → paper CUTLASS layout mnemonic `NN` / `TN` / `NT` (no `TT` in DeepBench).",
        "",
        "## Workload summary",
        "",
        f"- **Total dense GEMM entries:** {shapes['total']} ({shapes['unique_signatures']} unique M,N,K,layout)",
        f"- **Training:** {shapes['train']} | **Inference server:** {shapes['inference_server']} | **Inference device:** {shapes['inference_device']}",
        f"- **Layout distribution:** {shapes['layout_counts']}",
        f"- **M/N/K ranges:** M [{shapes['M_min']}, {shapes['M_max']}] (med {shapes['M_median']:.0f}), "
        f"N [{shapes['N_min']}, {shapes['N_max']}] (med {shapes['N_median']:.0f}), "
        f"K [{shapes['K_min']}, {shapes['K_max']}] (med {shapes['K_median']:.0f})",
        f"- **All dims %8:** {100*shapes['frac_div8']:.1f}% | %16: {100*shapes['frac_div16']:.1f}% | %32: {100*shapes['frac_div32']:.1f}% | %64: {100*shapes['frac_div64']:.1f}%",
        f"- **Skinny-ish (heuristic):** {shapes['skinny_count']} | **Any dimension < 32:** {shapes['any_dim_lt32']}",
        "",
        "Original DeepBench dtypes: **FP32 / FP16 training**; **INT8 inference** on server/device. This analysis maps problems to **BF16 CUTLASS** without changing M,N,K (dtype translation only).",
        "",
        "Unusual examples: `M=35` or `N=1` inference sequence steps; `K=500000` language-model projections; very tall/skinny RNN tiles.",
        "",
        "## CUTLASS compatibility (BF16, paper catalogue)",
        "",
        "Validity gate: SM90 **TMA alignment** (`src/eval/shapes.py`) — per-layout contiguous-dimension multiples of 8 for BF16. "
        f"If aligned, catalogue size is constant per layout (~{_CATALOG['TN']:,} valid configs; shape-independent enumeration in `config_space.py`).",
        "",
        f"- **Accepted:** {len(acc)} / {len(compat)} ({100*len(acc)/len(compat):.1f}%)",
        f"- **Rejected:** {len(rej)} ({100*len(rej)/len(compat):.1f}%)",
        "",
        "### By split",
        "",
    ]
    for split, g in compat.groupby("split"):
        a = g.accepted.sum()
        lines.append(f"- `{split}`: {a}/{len(g)} accepted ({100*a/len(g):.1f}%)")
    lines += ["", "### By layout", ""]
    for lay, g in compat.groupby("layout"):
        a = g.accepted.sum()
        lines.append(f"- `{lay}`: {a}/{len(g)} accepted ({100*a/len(g):.1f}%)")
    lines += ["", "### Dominant rejection reasons", ""]
    for reason, cnt in rej.rejection_reason.value_counts().head(10).items():
        lines.append(f"- `{reason}`: {cnt}")
    lines += [
        "",
        f"Among accepted problems, **{int((acc.rejection_reason == 'tma_ok;min_dim<32_may_reduce_feasible_tiles').sum())}** have min(M,N,K)<32; exhaustive benchmarking may still reject many tiles at `can_implement()` even though the catalogue is non-empty.",
        "",
        "## Overlap with our datasets",
        "",
    ]
    if sets["train_shapes"]:
        lines.append(f"- Training shapes in features parquet: **{len(sets['train_shapes'])}** unique (M,N,K)")
    else:
        lines.append("- Training shapes: features parquet not found locally (overlap counts may be zero)")
    if sets["broad_shapes"]:
        lines.append(
            f"- Broad eval shapes (`{sets.get('broad_source', 'artifacts/eval/out/shapes.json')}`): "
            f"**{len(sets['broad_shapes'])}** unique (M,N,K) → **{len(sets['broad_shapes']) * 4}** problems at 4 layouts"
        )
    else:
        lines.append(
            "- Broad eval shapes: `artifacts/eval/out/shapes.json` not found locally (2k held-out shapes); overlap vs 8k broad eval not computed"
        )
    lines += [
        f"- Exact (M,N,K) overlap with **training:** {overlap['exact_shape_overlap_train']} / {overlap['accepted_problems']} accepted DeepBench GEMMs",
        f"- Exact overlap with **68 exhaustive eval** shapes: {overlap['exact_shape_overlap_eval68']}",
        f"- Exact (M,N,K,layout) overlap with **68 eval groups:** {overlap['exact_shape_layout_overlap_eval68']}",
    ]
    if sets["broad_shapes"]:
        lines += [
            f"- Exact overlap with **broad eval** shapes: {overlap['exact_shape_overlap_broad']}",
            f"- Exact (M,N,K,layout) overlap with **broad eval (×4 layouts):** {overlap['exact_shape_layout_overlap_broad']}",
        ]
    lines += [
        f"- Near-neighbor (≤5% relative per dim) to a training shape: {overlap['near_shape_overlap_train_5pct']} accepted DeepBench GEMMs",
        "",
        "## Scientific usefulness",
        "",
        "1. **CUTLASS compatibility:** Mostly yes for server workloads; rejections concentrate on odd alignment (e.g. M=35, some N=1/2/4 cases) and NT layout with non-multiple-K.",
        "2. **Enough problems:** Yes for server inference (~70+ accepted); training set largely compatible. Device-only set is small (13) and partially redundant.",
        "3. **Different from our shapes:** Partially — many dimensions are legacy speech/LM sizes (1760, 5124, 7680, 24000, 500000) not on our 32-grid training sweep; overlap with train/eval is limited (see above).",
        "4. **ML workload story:** Credible as an **externally defined** DL GEMM list (speech, LM, keyword spotting), not as modern LLM coverage.",
        "5. **Age:** Benchmark is **2016-era** (Baidu, pre-Transformer dominance). Acceptable if framed narrowly.",
        "6. **Reviewer risk:** Moderate — expect \"outdated / not representative of LLMs\" unless scope is explicit.",
        "7. **Defensible wording:** *\"We additionally evaluate on dense GEMM shapes from the public DeepBench catalogue (Baidu Research, 2016), an externally defined set drawn from speech and language-model training/inference kernels. We map transpose flags to our CUTLASS layout convention and benchmark BF16 kernels without modifying problem dimensions.\"*",
        "",
        "Do **not** claim representativeness of contemporary LLM FFN/matmul workloads.",
        "",
        "## Proposed experiment (if pursued)",
        "",
        "**Protocol (frozen selector, no retraining):**",
        "",
        f"1. Use **accepted server inference** subset ({int((acc.split == 'inference_server').sum())} problems) as primary external case study; optionally add accepted training GEMMs as secondary.",
        "2. Per accepted (M,N,K,layout): enumerate full BF16 CUTLASS catalogue (`--exhaustive` plan write).",
        "3. Score with frozen paper MLP RankNet (full features); materialize nvMMH top-1 + variant fanout (existing `baseline/nvmmh/` protocol).",
        "4. Benchmark selected kernels on GH200 (same protocol as broad GEMM eval).",
        "5. Report: geo-mean speedup vs nvMMH, median speedup, win rate, coverage (accepted/total DeepBench).",
        "",
        "**Oracle / regret:** Full exhaustive measurement on ~70–200 problems × ~50k configs each is **not** practical. Treat as **throughput generalization** (like 8k broad eval), not selection-regret oracle eval.",
        "",
        "**Cost estimate:** ~70–90 accepted inference GEMMs × 1 selected kernel × benchmark ≈ similar order to a few hundred broad-eval problems (feasible); exhaustive oracle per problem is not.",
        "",
        "## Recommendation",
        "",
    ]
    n_server_acc = int((acc.split == "inference_server").sum())
    if n_server_acc >= 50 and overlap["exact_shape_overlap_eval68"] <= 5:
        rec = "**GO (narrow)** — Use DeepBench **inference_server** accepted GEMMs as a supplementary external DL shape study alongside broad GEMM eval. Frame as externally defined legacy DL kernels, BF16 remapping, throughput vs nvMMH. Skip sparse GEMM and device-only set in main paper."
    else:
        rec = "**CONDITIONAL** — Compatibility or overlap needs review before committing."
    lines.append(rec)
    lines.append("")
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))


def _min_dim_ok(m: int, n: int, k: int) -> bool:
    return min(m, n, k) >= MIN_EVAL_DIM


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = parse_gemm_header(GEMM_H)
    df = df[df.apply(lambda r: _min_dim_ok(r.M, r.N, r.K), axis=1)].reset_index(drop=True)
    df.to_csv(OUT / "deepbench_shapes.csv", index=False)

    compat_rows = []
    for _, row in df.iterrows():
        c = compat_row(row)
        compat_rows.append(
            {
                "split": row.split,
                "M": row.M,
                "N": row.N,
                "K": row.K,
                "layout": row.layout,
                **c,
            }
        )
    compat = pd.DataFrame(compat_rows)
    compat = compat[compat.accepted.astype(bool)].reset_index(drop=True)
    compat.to_csv(OUT / "deepbench_compatibility.csv", index=False)

    shapes_summary = summarize_shapes(df)
    sets = load_overlap_sets()
    compat_acc = compat.copy()
    compat_acc["accepted"] = compat_acc.accepted.astype(bool)
    overlap = overlap_report(compat_acc, sets)
    write_analysis_md(shapes_summary, compat, overlap, sets)

    print(f"Wrote {OUT}/deepbench_shapes.csv ({len(df)} rows)")
    print(f"Wrote {OUT}/deepbench_compatibility.csv")
    print(f"Wrote {OUT}/ANALYSIS.md")
    print(f"Accepted: {compat.accepted.sum()}/{len(compat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
