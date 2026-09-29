#!/usr/bin/env python3
"""
propose_fusion.py — Ordered fused-kernel proposals per (M, N, K, layout, fusion_kind).

Mirrors src/eval/propose.py but uses config_space_fusion + model/features_fusion.
nvMMH remains an unfused GEMM baseline (no epilogue fusion in the library).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from repo_paths import EVAL_OUT, SRC

sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "autotuner"))
sys.path.insert(0, str(SRC / "model"))
sys.path.insert(0, str(SRC / "baseline" / "nvmmh"))

from config_space import PRECISIONS as UNFUSED_PRECISIONS  # noqa: E402
from config_space_fusion import (  # noqa: E402
    LAYOUTS as FUSION_LAYOUTS,
)
from config_space_fusion import (
    PRECISIONS as FUSION_PRECISIONS,
)
from config_space_fusion import (
    enumerate_candidates,
    stamp_fusion,
)
from feature_manifest import category_levels_for  # noqa: E402
from features_fusion import featurize  # noqa: E402

# Reuse nvMMH path from unfused propose.
from propose import (  # noqa: E402
    CUTLASS_DEFAULT_SCHEDULER_ARGS,
    Proposal,
    feature_meta_from_model_dir,
    propose_nvmmh,
    sched_map_from_nvmmh,
)

DEFAULT_SHAPES = EVAL_OUT / "shapes_fusion.json"
DEFAULT_OUT_DIR = EVAL_OUT
FUSION_DTYPES = ("fp16", "fp32", "fp8_e4m3")


def nvmmh_dtype(dtype: str) -> str:
    """Unfused nvMMH config_space has no fp16 — use bf16 as unfused baseline."""
    if dtype == "fp16":
        return "bf16"
    return dtype


@lru_cache(maxsize=None)
def candidates_for(layout: str, dtype: str, fusion: str) -> tuple[dict, ...]:
    """Valid configs for a layout and dtype, stamped with the fusion kind."""
    base = enumerate_candidates(0, 0, 0, layout, dtype)
    return tuple(stamp_fusion(dict(c), fusion, dtype) for c in base)


def load_fusion_shapes(
    path: Path, limit: int | None
) -> tuple[list[tuple[int, int, int]], list[str], list[int]]:
    """Read a fusion shapes file: (shapes, fusion kind per shape, variant per shape).

    A plain [[M, N, K], ...] list means unfused (linear) shapes.
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        shapes = [tuple(s) for s in raw]
        kinds = ["linear"] * len(shapes)
        variants = [0] * len(shapes)
    else:
        shapes = [tuple(s) for s in raw["shapes"]]
        kinds = list(raw.get("fusion_kinds", ["linear"] * len(shapes)))
        variants = list(raw.get("variants", [0] * len(shapes)))
    if len(kinds) != len(shapes):
        raise SystemExit(f"{path}: fusion_kinds length {len(kinds)} != shapes {len(shapes)}")
    if len(variants) != len(shapes):
        raise SystemExit(f"{path}: variants length {len(variants)} != shapes {len(shapes)}")
    if limit:
        shapes, kinds, variants = shapes[:limit], kinds[:limit], variants[:limit]
    return shapes, kinds, variants


def _featurize_candidates(cands: list[dict], M: int, N: int, K: int, layout: str):
    import pandas as pd
    from features_fusion import _layout_label

    df = pd.DataFrame(cands)
    df["M"], df["N"], df["K"] = M, N, K
    df["layout"] = _layout_label(df)
    return featurize(df)


def _ohe(df, cat_cols: list[str], category_levels: dict) -> np.ndarray:
    parts = []
    for col in cat_cols:
        vals = df[col].astype(str)
        for lvl in category_levels[col]:
            parts.append((vals == lvl).to_numpy(dtype="float32"))
    return np.stack(parts, axis=1)


def _encode_raw(feats, num_cols: list[str], cat_cols: list[str], category_levels: dict) -> np.ndarray:
    num = feats[num_cols].to_numpy(dtype="float32")
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([num, _ohe(feats, cat_cols, category_levels)], axis=1).astype("float32")


class FusionMLPRanker:
    """MLP ranker over fused-epilogue candidates (model_mlp.pt2, or model_mlp.pt with its scaler)."""
    def __init__(self, model_dir: Path, dtype: str):
        import torch

        self.dtype = dtype
        self.num_cols, self.cat_cols = feature_meta_from_model_dir(model_dir)
        manifest = {
            "categorical_features": self.cat_cols,
            "numeric_features": self.num_cols,
        }
        self.category_levels = category_levels_for(manifest)

        pt2 = model_dir / "model_mlp.pt2"
        if pt2.exists():
            self._model = torch.export.load(str(pt2)).module()
            self._pt = None
            self._scaler = None
        else:
            pt = model_dir / "model_mlp.pt"
            if not pt.exists():
                raise FileNotFoundError(f"no model_mlp.pt2 or model_mlp.pt in {model_dir}")
            from sklearn.preprocessing import StandardScaler

            ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            from train_mlp import MLP  # noqa: E402

            scaler = StandardScaler()
            scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
            model = MLP(int(ckpt["n_in"]), tuple(ckpt["hidden"]), float(ckpt["dropout"]))
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._model = model
            self._scaler = scaler
            self._pt = pt

    def rank(self, M: int, N: int, K: int, layout: str, fusion: str) -> list[dict]:
        """Score every valid fused candidate with the MLP; return them best first with a score key."""
        import torch

        cands = list(candidates_for(layout, self.dtype, fusion))
        if not cands:
            return []
        feats = _featurize_candidates(cands, M, N, K, layout)
        missing = [c for c in self.num_cols + self.cat_cols if c not in feats.columns]
        if missing:
            raise KeyError(f"{self.dtype}: featurizer missing {missing}")

        if self._pt is None:
            X = _encode_raw(feats, self.num_cols, self.cat_cols, self.category_levels)
            with torch.no_grad():
                scores = self._model(torch.from_numpy(X)).cpu().numpy()
        else:
            from train_mlp import build_X  # noqa: E402

            X, _ = build_X(feats, self.num_cols, self.cat_cols, self.category_levels, self._scaler)
            with torch.no_grad():
                scores = self._model(torch.from_numpy(X.astype("float32"))).cpu().numpy()

        order = np.argsort(scores)[::-1]
        return [{**cands[i], "score": float(scores[i])} for i in order]


class FusionXGBRanker:
    """XGBoost ranker over fused-epilogue candidates (model_A.ubj)."""
    def __init__(self, model_dir: Path, dtype: str):
        import pandas as pd
        import xgboost as xgb

        self.dtype = dtype
        num_cols, cat_cols = feature_meta_from_model_dir(model_dir)
        self.feat_cols = num_cols + cat_cols
        self.cat_cols = cat_cols
        manifest = {"categorical_features": cat_cols, "numeric_features": num_cols}
        self.category_levels = category_levels_for(manifest)
        ubj = model_dir / "model_A.ubj"
        if not ubj.exists():
            raise FileNotFoundError(f"no model_A.ubj in {model_dir}")
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(ubj))
        self._pd = pd

    def rank(self, M: int, N: int, K: int, layout: str, fusion: str) -> list[dict]:
        """Score every valid fused candidate with XGBoost; return them best first with a score key."""
        cands = list(candidates_for(layout, self.dtype, fusion))
        if not cands:
            return []
        feats = _featurize_candidates(cands, M, N, K, layout)
        missing = [c for c in self.feat_cols if c not in feats.columns]
        if missing:
            raise KeyError(f"missing features {missing}")
        for col in self.cat_cols:
            feats[col] = self._pd.Categorical(
                feats[col].astype(str), categories=self.category_levels[col]
            )
        scores = self._model.predict(feats[self.feat_cols])
        order = np.argsort(scores)[::-1]
        return [{**cands[i], "score": float(scores[i])} for i in order]


def propose_fusion_model(
    shapes: list[tuple[int, int, int]],
    fusion_kinds: list[str],
    method: str,
    ranker,
    sched_map: dict[tuple[int, int, int, str], tuple[int, int, int]] | None = None,
    layouts: tuple[str, ...] | None = None,
    variants: list[int] | None = None,
) -> tuple[dict, list[Proposal], dict]:
    """Fused-epilogue counterpart of propose.propose_model: the top-ranked config per problem."""
    configs: dict[str, dict] = {}
    proposals: list[Proposal] = []
    skipped: list[dict] = []
    default_sched = CUTLASS_DEFAULT_SCHEDULER_ARGS
    layout_names = layouts or tuple(FUSION_LAYOUTS.keys())
    total = len(shapes) * len(layout_names)
    done = 0
    n_sched_fallback = 0
    t0 = time.time()

    if variants is None:
        variants = [0] * len(shapes)

    for layout_name in layout_names:
        short_layout = layout_name.lower()
        for (M, N, K), fusion, variant in zip(shapes, fusion_kinds, variants):
            ranked = ranker.rank(M, N, K, layout_name, fusion)
            if not ranked:
                skipped.append({
                    "M": M, "N": N, "K": K, "layout": short_layout, "fusion": fusion,
                    "reason": "no_ranked_candidate",
                })
                continue
            cfg = dict(ranked[0])
            score = cfg.pop("score", None)
            configs[cfg["name"]] = cfg
            if sched_map is not None:
                key = (M, N, K, short_layout)
                if key in sched_map:
                    raster, swizzle, splits = sched_map[key]
                else:
                    raster, swizzle, splits = default_sched
                    n_sched_fallback += 1
                    skipped.append({
                        "M": M, "N": N, "K": K, "layout": short_layout, "fusion": fusion,
                        "reason": "nvmmh_scheduler_missing",
                    })
            else:
                raster, swizzle, splits = default_sched
            proposals.append(
                Proposal(
                    method=method,
                    M=M,
                    N=N,
                    K=K,
                    layout=short_layout,
                    rank=1,
                    variant=int(variant),
                    config_name=cfg["name"],
                    raster_order=raster,
                    swizzle_size=swizzle,
                    splits=splits,
                    score=score,
                )
            )
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                print(
                    f"  {done}/{total} problems ({rate:.1f}/s, "
                    f"eta {(total - done) / rate / 60:.1f} min)",
                    flush=True,
                )
    if n_sched_fallback:
        print(f"  warning: {n_sched_fallback} problems missing nvMMH scheduler", flush=True)
    meta = {
        "n_proposed": len(proposals),
        "n_sched_fallback": n_sched_fallback,
        "n_no_candidate": sum(1 for s in skipped if s["reason"] == "no_ranked_candidate"),
        "skipped": skipped,
    }
    return configs, proposals, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Fusion epilogue kernel proposals")
    ap.add_argument("--method", required=True)
    ap.add_argument("--backend", choices=["nvmmh", "mlp", "xgb"], default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", default="H100_SXM")
    ap.add_argument("--nvmmh-top-k", type=int, default=1)
    ap.add_argument("--scheduler-from", type=Path, default=None)
    ap.add_argument("--dtype", default="fp16", choices=FUSION_DTYPES)
    ap.add_argument("--layouts", default=None)
    args = ap.parse_args()

    layout_names = None
    if args.layouts:
        layout_names = tuple(x.strip() for x in args.layouts.split(",") if x.strip())
        bad = [x for x in layout_names if x not in FUSION_LAYOUTS]
        if bad:
            print(f"ERROR: unknown layout(s) {bad}", file=sys.stderr)
            return 1

    backend = args.backend
    if backend is None:
        if args.method == "nvmmh":
            backend = "nvmmh"
        elif args.method.startswith("mlp"):
            backend = "mlp"
        elif args.method.startswith("xgb"):
            backend = "xgb"
        else:
            print("ERROR: pass --backend", file=sys.stderr)
            return 1

    shapes, fusion_kinds, variants = load_fusion_shapes(args.shapes, args.limit)
    n_layouts = len(layout_names or FUSION_LAYOUTS)
    print(
        f"{len(shapes)} shapes x {n_layouts} layouts = {len(shapes) * n_layouts} fusion problems"
    )

    propose_meta: dict = {}
    if backend == "nvmmh":
        nv_dtype = nvmmh_dtype(args.dtype)
        if nv_dtype not in UNFUSED_PRECISIONS:
            print(f"ERROR: nvmmh dtype {nv_dtype} not in unfused PRECISIONS", file=sys.stderr)
            return 1
        configs, proposals = propose_nvmmh(
            shapes, args.gpu, args.nvmmh_top_k, nv_dtype, layout_names
        )
    else:
        if args.model_dir is None or not args.model_dir.is_dir():
            print("ERROR: --model-dir required for mlp/xgb", file=sys.stderr)
            return 1
        if args.dtype not in FUSION_PRECISIONS:
            print(f"ERROR: dtype {args.dtype} not in fusion PRECISIONS", file=sys.stderr)
            return 1
        sched_map = sched_map_from_nvmmh(args.scheduler_from) if args.scheduler_from else None
        if sched_map:
            print(f"  scheduler from {args.scheduler_from.name} ({len(sched_map)} problems)")
        ranker = FusionMLPRanker(args.model_dir, args.dtype) if backend == "mlp" else FusionXGBRanker(args.model_dir, args.dtype)
        configs, proposals, propose_meta = propose_fusion_model(
            shapes, fusion_kinds, args.method, ranker, sched_map, layout_names, variants
        )

    out = args.out or DEFAULT_OUT_DIR / f"proposals_{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "method": args.method,
                "dtype": args.dtype,
                "gpu": args.gpu if backend == "nvmmh" else None,
                "model_dir": str(args.model_dir) if args.model_dir else None,
                "fusion": True,
                "configs": configs,
                "proposals": [asdict(p) for p in proposals],
                "propose_meta": propose_meta,
            }
        )
    )
    print(f"wrote {len(proposals)} proposals, {len(configs)} kernels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
