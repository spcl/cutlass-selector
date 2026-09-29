#!/usr/bin/env python3
"""Select rank-1 kernels (ours) and enumerate nvMMH variants for traced MLP GEMMs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(SRC / "autotuner"))
sys.path.insert(0, str(SRC / "baseline" / "nvmmh"))
sys.path.insert(0, str(SRC / "eval"))

from config_space import LAYOUTS, PRECISIONS  # noqa: E402
from mlp_case_study.common import OUT_ROOT, discover_models  # noqa: E402
from propose import MetricsMLPRanker, Proposal, candidates_for  # noqa: E402
from query import NvmmhInterface, layout_enum, precision_string  # noqa: E402
from translate import materialize  # noqa: E402


def _checkpoint_for_model(model_id: str) -> Path:
    for spec in discover_models():
        if spec.model_id == model_id:
            return spec.checkpoint
    raise KeyError(model_id)


def _measure_catalogue_s() -> float:
    candidates_for.cache_clear()
    t0 = time.perf_counter()
    _ = candidates_for("TN", "bf16")
    return time.perf_counter() - t0


def select_ours(row: pd.Series, ranker: MetricsMLPRanker) -> tuple[dict, dict, dict]:
    M, N, K = int(row.M), int(row.N), int(row.K)
    layout = str(row.layout)

    t0 = time.perf_counter()
    ranked = ranker.rank(M, N, K, layout)
    select_s = time.perf_counter() - t0
    if not ranked:
        raise RuntimeError(f"no ranked candidate for {row.gemm_id}")

    best = dict(ranked[0])
    score = best.pop("score", None)
    cfg = dict(best)
    out = {
        "model": row.model,
        "gemm_id": row.gemm_id,
        "M": M,
        "N": N,
        "K": K,
        "layout": layout,
        "dtype": str(row["dtype"]),
        "config_name": cfg["name"],
        "score": score,
        "tile_m": cfg["tile_m"],
        "tile_n": cfg["tile_n"],
        "tile_k": cfg["tile_k"],
        "cluster_m": cfg["cluster_m"],
        "cluster_n": cfg["cluster_n"],
        "cluster_k": cfg["cluster_k"],
        "stages": cfg["stages"],
        "kernel_schedule": cfg["kernel_schedule"],
        "epilogue_schedule": cfg["epilogue_schedule"],
        "scheduler": cfg["scheduler"],
        "raster_order": -1,
        "swizzle_size": 1,
        "splits": 1,
        "select_rank_s": select_s,
    }
    return out, cfg, {"select_rank_s": select_s}


def select_nvmmh(row: pd.Series, iface: NvmmhInterface) -> tuple[list[dict], list[dict], dict]:
    M, N, K = int(row.M), int(row.N), int(row.K)
    layout = str(row.layout)
    dtype = "bf16"
    prec_tuple = PRECISIONS[dtype]
    precision = precision_string(
        prec_tuple[1], prec_tuple[2], prec_tuple[4], prec_tuple[3]
    )
    _, layout_a, layout_b = LAYOUTS[layout]
    lay = layout_enum(layout_a, layout_b, iface.nvmmh)

    t0 = time.perf_counter()
    recs = iface.recommend(M, N, K, precision, lay, top_k=1)
    recommend_s = time.perf_counter() - t0
    if not recs:
        return [], [], {"recommend_s": recommend_s, "n_variants": 0}

    rec = recs[0]
    rows: list[dict] = []
    configs: list[dict] = []
    for vi, v in enumerate(materialize(rec, layout_a, layout_b, prec_tuple)):
        cfg = dict(v["config"])
        configs.append(cfg)
        rows.append({
            "model": row.model,
            "gemm_id": row.gemm_id,
            "M": M,
            "N": N,
            "K": K,
            "layout": layout,
            "dtype": dtype,
            "rank": rec["rank"],
            "variant": vi,
            "config_name": cfg["name"],
            "tile_m": cfg["tile_m"],
            "tile_n": cfg["tile_n"],
            "tile_k": cfg["tile_k"],
            "cluster_m": cfg["cluster_m"],
            "cluster_n": cfg["cluster_n"],
            "cluster_k": cfg["cluster_k"],
            "stages": cfg["stages"],
            "kernel_schedule": cfg["kernel_schedule"],
            "epilogue_schedule": cfg["epilogue_schedule"],
            "scheduler": cfg["scheduler"],
            "raster_order": v["raster_order"],
            "swizzle_size": v["swizzle_size"],
            "splits": v["splits"],
            "nvmmh_estimated_runtime_s": rec.get("estimated_runtime_s"),
            "recommend_s": recommend_s,
        })
    return rows, configs, {"recommend_s": recommend_s, "n_variants": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=Path, default=OUT_ROOT / "mlp_gemm_shapes.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--gpu", default="H100_SXM")
    args = ap.parse_args()

    shapes = pd.read_csv(args.shapes)
    shapes["dtype"] = "bf16"
    shapes["layout"] = shapes["layout"].astype(str).str.upper()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    catalogue_s = _measure_catalogue_s()
    print(f"catalogue enumeration: {catalogue_s:.3f}s", flush=True)

    ours_rows: list[dict] = []
    nvmmh_rows: list[dict] = []
    select_timing: list[dict] = []
    all_configs: dict[str, dict] = {}
    proposals: list[Proposal] = []

    iface = NvmmhInterface(gpu=args.gpu)
    rankers: dict[str, MetricsMLPRanker] = {}

    for _, row in shapes.iterrows():
        model_id = row.model
        if model_id not in rankers:
            rankers[model_id] = MetricsMLPRanker(
                _checkpoint_for_model(model_id), dtype=str(row["dtype"])
            )

        ours, cfg, ours_timing = select_ours(row, rankers[model_id])
        ours["select_catalogue_s"] = catalogue_s
        ours_rows.append(ours)
        all_configs[cfg["name"]] = cfg
        proposals.append(
            Proposal(
                method=f"{model_id}_ours",
                M=ours["M"],
                N=ours["N"],
                K=ours["K"],
                layout=ours["layout"].lower(),
                rank=1,
                variant=0,
                config_name=cfg["name"],
                raster_order=ours["raster_order"],
                swizzle_size=ours["swizzle_size"],
                splits=ours["splits"],
                score=ours["score"],
            )
        )

        nv_rows, nv_cfgs, nv_timing = select_nvmmh(row, iface)
        nvmmh_rows.extend(nv_rows)
        for cfg in nv_cfgs:
            all_configs[cfg["name"]] = cfg
        for nv in nv_rows:
            proposals.append(
                Proposal(
                    method=f"{model_id}_nvmmh",
                    M=nv["M"],
                    N=nv["N"],
                    K=nv["K"],
                    layout=nv["layout"].lower(),
                    rank=nv["rank"],
                    variant=nv["variant"],
                    config_name=nv["config_name"],
                    raster_order=nv["raster_order"],
                    swizzle_size=nv["swizzle_size"],
                    splits=nv["splits"],
                    score=nv.get("nvmmh_estimated_runtime_s"),
                )
            )

        select_timing.append({
            "model": model_id,
            "gemm_id": row.gemm_id,
            "ours_catalogue_s": catalogue_s,
            "ours_select_s": ours_timing["select_rank_s"],
            "nvmmh_recommend_s": nv_timing["recommend_s"],
            "nvmmh_n_variants": nv_timing["n_variants"],
        })
        print(
            f"  {model_id} {row.gemm_id}: ours={ours['config_name'][:56]} "
            f"nvmmh_variants={nv_timing['n_variants']}",
            flush=True,
        )

    iface.close()

    ours_path = args.out_dir / "our_selected_kernels.csv"
    nv_path = args.out_dir / "nvmmh_candidates.csv"
    pd.DataFrame(ours_rows).to_csv(ours_path, index=False)
    pd.DataFrame(nvmmh_rows).to_csv(nv_path, index=False)
    pd.DataFrame(select_timing).to_csv(args.out_dir / "selection_timing.csv", index=False)

    prop_path = args.out_dir / "proposals_case_study.json"
    prop_path.write_text(
        json.dumps({
            "method": "mlp_case_study",
            "dtype": "bf16",
            "gpu": args.gpu,
            "configs": all_configs,
            "proposals": [asdict(p) for p in proposals],
        })
    )
    print(f"wrote {ours_path}, {nv_path}, {prop_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
