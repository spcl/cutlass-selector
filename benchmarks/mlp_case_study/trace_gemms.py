#!/usr/bin/env python3
"""Trace GEMM shapes executed during real MLP selector inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "model"))

from mlp_case_study.common import (  # noqa: E402
    OUT_ROOT,
    REF_GEMM,
    REF_LAYOUT,
    SELECTOR_DTYPE,
    discover_models,
    gemm_signature,
)
from propose import MetricsMLPRanker, candidates_for  # noqa: E402


class _TraceMLP(nn.Module):
    """Minimal copy of model/train_mlp.MLP for hook tracing (no sklearn import)."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_trace_module(model_dir: Path) -> nn.Module:
    """Load the inner MLP used for forward hooks (not the exported wrapper)."""
    pt = model_dir / "model_mlp.pt"
    if not pt.is_file():
        raise FileNotFoundError(f"missing {pt} — need .pt checkpoint for layer hooks")
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    model = _TraceMLP(int(ckpt["n_in"]), tuple(ckpt["hidden"]), float(ckpt["dropout"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _build_forward_input(ranker: MetricsMLPRanker, M: int, N: int, K: int, layout: str) -> torch.Tensor:
    import numpy as np
    from features import CATEGORY_LEVELS  # noqa: E402
    from propose import _encode_raw, _featurize_candidates  # noqa: E402

    cands = list(candidates_for(layout, ranker.dtype))
    feats = _featurize_candidates(cands, M, N, K, layout)
    if ranker._pt is None:
        X = _encode_raw(feats, ranker.num_cols, ranker.cat_cols)
    else:
        num = feats[ranker.num_cols].to_numpy(dtype="float32")
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        num = (num - ranker._scaler.mean_) / ranker._scaler.scale_
        cat_parts = []
        for col in ranker.cat_cols:
            vals = feats[col].astype(str)
            for lvl in CATEGORY_LEVELS[col]:
                cat_parts.append((vals == lvl).to_numpy(dtype="float32"))
        cat = np.stack(cat_parts, axis=1) if cat_parts else np.zeros((len(num), 0), dtype="float32")
        X = np.concatenate([num, cat], axis=1).astype("float32")
    return torch.from_numpy(X)


def trace_model(
    spec,
    ref_m: int,
    ref_n: int,
    ref_k: int,
    layout: str,
) -> tuple[pd.DataFrame, dict]:
    ranker = MetricsMLPRanker(spec.checkpoint, dtype=SELECTOR_DTYPE)
    mlp = _load_trace_module(spec.checkpoint)

    records: list[dict] = []
    layer_names: dict[int, str] = {}
    for name, module in mlp.named_modules():
        if isinstance(module, nn.Linear):
            layer_names[module] = name

    def hook_fn(module, inputs, _output):
        x = inputs[0]
        batch, k_dim = int(x.shape[0]), int(x.shape[1])
        n_dim = int(module.out_features)
        records.append({
            "layer": layer_names[module],
            "M": batch,
            "N": n_dim,
            "K": k_dim,
            "layout": layout,
            "dtype": "fp32",
            "accum_dtype": "fp32",
            "epilogue": "linear",
            "bench_dtype": str(SELECTOR_DTYPE),
            "bench_layout": str(layout),
            "bench_accum_dtype": "float",
        })

    handles = []
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook_fn))

    X = _build_forward_input(ranker, ref_m, ref_n, ref_k, layout)

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = mlp(X)
    forward_s = time.perf_counter() - t0
    for h in handles:
        h.remove()

    if not records:
        raise RuntimeError(f"no Linear GEMMs captured for {spec.model_id}")

    # Deduplicate identical signatures while preserving layer lists.
    grouped: dict[str, dict] = {}
    for rec in records:
        sig = gemm_signature(
            rec["M"], rec["N"], rec["K"], rec["bench_layout"],
            rec["bench_dtype"], rec["bench_accum_dtype"],
        )
        if sig not in grouped:
            grouped[sig] = {**rec, "layers": [rec["layer"]], "occurrence_count": 1}
        else:
            grouped[sig]["layers"].append(rec["layer"])
            grouped[sig]["occurrence_count"] += 1

    rows = []
    for sig, rec in grouped.items():
        rows.append({
            "model": spec.model_id,
            "checkpoint": str(spec.checkpoint),
            "gemm_id": sig,
            "layer": ";".join(rec["layers"]),
            "M": rec["M"],
            "N": rec["N"],
            "K": rec["K"],
            "layout": str(rec["bench_layout"]),
            "dtype": "bf16",
            "accum_dtype": rec["bench_accum_dtype"],
            "pytorch_dtype": rec["dtype"],
            "epilogue": rec["epilogue"],
            "occurrence_count": rec["occurrence_count"],
            "n_candidates": int(records[0]["M"]),
        })

    timing = {
        "model": spec.model_id,
        "mlp_forward_cpu_s": forward_s,
        "n_unique_gemms": len(rows),
        "n_gemm_calls_per_inference": sum(r["occurrence_count"] for r in rows),
        "n_candidates": int(records[0]["M"]),
    }
    return pd.DataFrame(rows), timing


def print_summary(df: pd.DataFrame) -> None:
    for model, sub in df.groupby("model"):
        print(f"\nModel {model}:")
        for i, row in sub.reset_index(drop=True).iterrows():
            print(
                f"  GEMM {i + 1}: {row.M}x{row.N}x{row.K} ({row.layout}, {row.dtype}), "
                f"layers={row.layer}, occurrences={row.occurrence_count}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Trace MLP internal GEMMs during inference.")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--ref-m", type=int, default=REF_GEMM[0])
    ap.add_argument("--ref-n", type=int, default=REF_GEMM[1])
    ap.add_argument("--ref-k", type=int, default=REF_GEMM[2])
    ap.add_argument("--layout", default=REF_LAYOUT)
    ap.add_argument("--model", default=None, help="single model_id (default: all discovered)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = discover_models()
    if args.model:
        models = [m for m in models if m.model_id == args.model]
        if not models:
            raise SystemExit(f"unknown model {args.model}")

    frames: list[pd.DataFrame] = []
    timings: list[dict] = []
    for spec in models:
        print(f"tracing {spec.model_id} ...", flush=True)
        df, timing = trace_model(spec, args.ref_m, args.ref_n, args.ref_k, args.layout)
        frames.append(df)
        timings.append(timing)

    out = pd.concat(frames, ignore_index=True)
    shapes_path = args.out_dir / "mlp_gemm_shapes.csv"
    out.to_csv(shapes_path, index=False)
    pd.DataFrame(timings).to_csv(args.out_dir / "trace_timing.csv", index=False)
    print_summary(out)
    print(f"\nwrote {shapes_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
