#!/usr/bin/env python3
"""
propose.py — Turn each (M, N, K, layout) problem into an *ordered* list of candidate
kernels, one list per method. Nothing is compiled or benchmarked here; this is the
CPU-only planning step whose output run.py consumes.

Ordering is the whole point. Every proposal carries its provenance — method, rank and
variant — so the measured results can be read at any candidate budget from one table
(see src/eval/README.md).

METHODS
  nvmmh          nvMMH's rank-1 recommendation, fanned out into schedule variants
                 (src/baseline/nvmmh/translate.py). Its raster / swizzle / split_k are carried
                 as runtime benchmark args.
  mlp_* / xgb_*  the model's single highest-scoring kernel from the enumerated candidate
                 space, with nvMMH rank-1 raster / swizzle / splits when --scheduler-from
                 is passed (slurm_eval.sh does this by default).

    python src/eval/propose.py --method nvmmh --gpu H100_SXM
    python src/eval/propose.py --method mlp_full --backend mlp --model-dir artifacts/analysis/paper/mlp_mse
    python src/eval/propose.py --method xgb_structural --backend xgb \
        --model-dir artifacts/analysis/paper/xgb_mse_structural
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from repo_paths import EVAL_OUT, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "autotuner"))
sys.path.insert(0, str(SRC / "baseline" / "nvmmh"))
sys.path.insert(0, str(SRC / "model"))

from config_space import LAYOUTS, PRECISIONS, enumerate_candidates  # noqa: E402

DEFAULT_SHAPES = EVAL_OUT / "shapes.json"
DEFAULT_OUT_DIR = EVAL_OUT

CUTLASS_DEFAULT_SCHEDULER_ARGS = (-1, 1, 1)


@lru_cache(maxsize=None)
def candidates_for(layout: str, dtype: str) -> tuple[dict, ...]:
    """The candidate space is shape-independent, so enumerate it once per layout."""
    return tuple(enumerate_candidates(0, 0, 0, layout, dtype))


@dataclass(frozen=True)
class Proposal:
    """One candidate kernel for one problem, with its provenance and tile-scheduler runtime arguments."""
    method: str
    M: int
    N: int
    K: int
    layout: str
    rank: int
    variant: int
    config_name: str
    raster_order: int
    swizzle_size: int
    splits: int
    score: float | None


def feature_meta_from_model_dir(model_dir: Path) -> tuple[list[str], list[str]]:
    """Feature columns for inference — metrics.json or features.manifest.json."""
    metrics_path = model_dir / "metrics.json"
    if metrics_path.is_file():
        meta = json.loads(metrics_path.read_text())
        if "numeric_features" in meta and "categorical_features" in meta:
            return meta["numeric_features"], meta["categorical_features"]
    for name in ("features.manifest.json",):
        manifest_path = model_dir / name
        if manifest_path.is_file():
            m = json.loads(manifest_path.read_text())
            return m["numeric_features"], m["categorical_features"]
    raise KeyError(
        f"{model_dir}: missing numeric_features/categorical_features in "
        f"metrics.json or features.manifest.json"
    )


def sched_map_from_nvmmh(
    path: Path,
) -> dict[tuple[int, int, int, str], tuple[int, int, int]]:
    """Map (M, N, K, layout) to nvMMH rank-1 (raster_order, swizzle_size, splits) from a proposals file."""
    payload = json.loads(path.read_text())
    sched: dict[tuple[int, int, int, str], tuple[int, int, int]] = {}
    for p in payload["proposals"]:
        if p.get("rank") != 1:
            continue
        key = (p["M"], p["N"], p["K"], p["layout"])
        sched.setdefault(key, (p["raster_order"], p["swizzle_size"], p["splits"]))
    return sched


def load_shapes(path: Path, limit: int | None) -> list[tuple[int, int, int]]:
    """Read a shapes.json list of [M, N, K]; limit keeps the first n."""
    raw = path.read_text()
    try:
        shapes = [tuple(s) for s in json.loads(raw)]
    except json.JSONDecodeError as exc:
        preview = raw[:120].replace("\n", " ") if raw else "(empty)"
        raise SystemExit(
            f"ERROR: {path} is not valid JSON ({exc}). "
            f"Delete it and re-run shapes.py. Starts with: {preview!r}"
        ) from exc
    return shapes[:limit] if limit else shapes


Problem = tuple[int, int, int, str]  # M, N, K, layout (TN/TT/NN/NT)


def load_problems(path: Path, limit: int | None) -> list[Problem]:
    """Read explicit (M, N, K, layout) problems from a list or a {"problems": [...]} file."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        rows = payload.get("problems", payload.get("entries", []))
        meta = payload.get("meta", {})
        if meta:
            print(f"  problems meta: {meta.get('n_problems', len(rows))} from {path.name}")
    else:
        rows = payload
    problems: list[Problem] = []
    for row in rows:
        if isinstance(row, dict):
            layout = str(row["layout"]).upper()
            problems.append((int(row["M"]), int(row["N"]), int(row["K"]), layout))
        else:
            m, n, k, layout = row
            problems.append((int(m), int(n), int(k), str(layout).upper()))
    bad = [lay for _, _, _, lay in problems if lay not in LAYOUTS]
    if bad:
        raise SystemExit(f"ERROR: unknown layout(s) in {path}: {sorted(set(bad))}")
    return problems[:limit] if limit else problems


def shapes_to_problems(
    shapes: list[tuple[int, int, int]],
    layout_names: tuple[str, ...],
) -> list[Problem]:
    """Cross shapes with layouts into (M, N, K, layout) problems."""
    return [
        (M, N, K, layout_name)
        for M, N, K in shapes
        for layout_name in layout_names
    ]


# ── nvMMH ─────────────────────────────────────────────────────────────────────


def propose_nvmmh(
    problems: list[Problem],
    gpu: str,
    top_k: int,
    dtype: str = "bf16",
) -> tuple[dict, list[Proposal]]:
    """nvMMH proposals: its top_k recommendations per problem, each fanned out into schedule variants.

    Returns (configs by name, proposals). Problems nvMMH declines get no proposal.
    """
    from query import NvmmhInterface, layout_enum, precision_string
    from translate import materialize

    prec_tuple = PRECISIONS[dtype]
    iface = NvmmhInterface(gpu=gpu)
    precision = precision_string(
        prec_tuple[1], prec_tuple[2], prec_tuple[4], prec_tuple[3]
    )

    configs: dict[str, dict] = {}
    proposals: list[Proposal] = []
    n_declined = 0
    n_no_variant = 0

    for M, N, K, layout_name in problems:
        short_layout = layout_name.lower()
        _, layout_a, layout_b = LAYOUTS[layout_name]
        lay = layout_enum(layout_a, layout_b, iface.nvmmh)
        recs = iface.recommend(M, N, K, precision, lay, top_k=top_k)
        if not recs:
            n_declined += 1
            continue
        n_before = len(proposals)
        for rec in recs:
            for vi, v in enumerate(materialize(rec, layout_a, layout_b, prec_tuple)):
                cfg = v["config"]
                configs[cfg["name"]] = cfg
                proposals.append(
                    Proposal(
                        method="nvmmh",
                        M=M,
                        N=N,
                        K=K,
                        layout=short_layout,
                        rank=rec["rank"],
                        variant=vi,
                        config_name=cfg["name"],
                        raster_order=v["raster_order"],
                        swizzle_size=v["swizzle_size"],
                        splits=v["splits"],
                        score=rec["estimated_runtime_s"],
                    )
                )
        if len(proposals) == n_before:
            n_no_variant += 1
    iface.close()

    if n_no_variant:
        print(f"  {n_no_variant} recommendations yielded no valid schedule variant")
    if n_declined:
        print(f"  nvMMH declined {n_declined} problems (no recommendation)")
    return configs, proposals


# ── learned rankers ───────────────────────────────────────────────────────────


def _ohe(df, cat_cols: list[str]) -> np.ndarray:
    from features import CATEGORY_LEVELS

    parts = []
    for col in cat_cols:
        vals = df[col].astype(str)
        for lvl in CATEGORY_LEVELS[col]:
            parts.append((vals == lvl).to_numpy(dtype="float32"))
    return np.stack(parts, axis=1)


def _encode_raw(feats, num_cols: list[str], cat_cols: list[str]) -> np.ndarray:
    """Raw feature matrix for torch.export artifacts (scaler lives in the graph)."""
    import numpy as np

    num = feats[num_cols].to_numpy(dtype="float32")
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([num, _ohe(feats, cat_cols)], axis=1).astype("float32")


@lru_cache(maxsize=None)
def _base_frame(layout: str, dtype: str):
    """Candidate frame for a layout, minus the problem dims.

    Building this from ~60k dicts costs more than featurize() itself and nothing in it
    depends on (M, N, K), so it is built once per (layout, dtype) and copied per problem. Row
    order matches candidates_for(layout, dtype), which the callers index scores by.
    """
    import pandas as pd
    from features import _layout_label

    df = pd.DataFrame(list(candidates_for(layout, dtype)))
    df["layout"] = _layout_label(df)
    return df


def _featurize_candidates(cands: list[dict], M: int, N: int, K: int, layout: str, dtype: str):
    from features import featurize

    df = _base_frame(layout, dtype).copy()
    if len(df) != len(cands):
        raise RuntimeError(f"candidate/frame mismatch for {layout}: {len(cands)} vs {len(df)}")
    df["M"], df["N"], df["K"] = M, N, K
    return featurize(df)


class MetricsMLPRanker:
    """Score with model_mlp.pt2 (preferred) or model_mlp.pt + metrics.json feature columns."""

    def __init__(self, model_dir: Path, dtype: str = "bf16"):
        import numpy as np
        import torch

        self.model_dir = model_dir
        self.dtype = dtype
        self.num_cols, self.cat_cols = feature_meta_from_model_dir(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        pt2 = model_dir / "model_mlp.pt2"
        if pt2.exists():
            self._model = torch.export.load(str(pt2)).module().to(self.device)
            self._pt = None
        else:
            pt = model_dir / "model_mlp.pt"
            if not pt.exists():
                raise FileNotFoundError(
                    f"no model_mlp.pt2 or model_mlp.pt in {model_dir}"
                )
            from sklearn.preprocessing import StandardScaler

            ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            sys.path.insert(0, str(SRC / "model"))
            from train_mlp import MLP  # noqa: E402

            scaler = StandardScaler()
            scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
            model = MLP(
                int(ckpt["n_in"]), tuple(ckpt["hidden"]), float(ckpt["dropout"])
            )
            model.load_state_dict(ckpt["model_state"])
            model.eval().to(self.device)
            self._model = model
            self._scaler = scaler
            self._pt = pt

    def rank(self, M: int, N: int, K: int, layout: str) -> list[dict]:
        """Score every valid candidate for the problem with the MLP; return them best first with a score key."""
        import numpy as np
        import torch

        cands = list(candidates_for(layout, self.dtype))
        if not cands:
            return []
        feats = _featurize_candidates(cands, M, N, K, layout, self.dtype)
        missing = [c for c in self.num_cols + self.cat_cols if c not in feats.columns]
        if missing:
            raise KeyError(f"{self.model_dir.name}: featurizer missing {missing}")

        if self._pt is None:
            X = _encode_raw(feats, self.num_cols, self.cat_cols)
            with torch.no_grad():
                scores = self._model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        else:
            sys.path.insert(0, str(SRC / "model"))
            from train_mlp import build_X  # noqa: E402

            X, _ = build_X(feats, self.num_cols, self.cat_cols, scaler=self._scaler)
            with torch.no_grad():
                scores = self._model(
                    torch.from_numpy(X.astype("float32")).to(self.device)).cpu().numpy()

        order = np.argsort(scores)[::-1]
        return [{**cands[i], "score": float(scores[i])} for i in order]


class RidgeRanker:
    """Score with frozen ridge weights from baselines/linear_*/model_ridge.json."""

    def __init__(self, model_dir: Path, dtype: str = "bf16"):
        from features import CATEGORY_LEVELS
        from sklearn.preprocessing import StandardScaler

        self.dtype = dtype
        self.num_cols, self.cat_cols = feature_meta_from_model_dir(model_dir)
        ridge_path = model_dir / "model_ridge.json"
        if not ridge_path.is_file():
            raise FileNotFoundError(f"no model_ridge.json in {model_dir}")
        payload = json.loads(ridge_path.read_text())
        self.coef_ = np.asarray(payload["coef"], dtype=np.float64)
        self.intercept_ = float(payload["intercept"])
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
        scaler.n_features_in_ = len(scaler.mean_)
        self._scaler = scaler
        self._category_levels = CATEGORY_LEVELS

    def rank(self, M: int, N: int, K: int, layout: str) -> list[dict]:
        """Score every valid candidate with the ridge model; return them best first with a score key."""
        cands = list(candidates_for(layout, self.dtype))
        if not cands:
            return []
        feats = _featurize_candidates(cands, M, N, K, layout, self.dtype)
        missing = [c for c in self.num_cols + self.cat_cols if c not in feats.columns]
        if missing:
            raise KeyError(f"{self.model_dir.name}: featurizer missing {missing}")

        sys.path.insert(0, str(SRC / "model"))
        from train_mlp import build_X  # noqa: E402

        X, _ = build_X(
            feats, self.num_cols, self.cat_cols, self._category_levels, self._scaler
        )
        scores = X @ self.coef_ + self.intercept_
        order = np.argsort(scores)[::-1]
        return [{**cands[i], "score": float(scores[i])} for i in order]


class RandomRanker:
    """Uniform random kernel per problem (deterministic seed per M,N,K,layout)."""

    def __init__(self, seed: int = 42, dtype: str = "bf16"):
        import zlib

        self.seed = int(seed)
        self.dtype = dtype
        self._zlib = zlib

    def rank(self, M: int, N: int, K: int, layout: str) -> list[dict]:
        """Return every valid candidate in a random order that is fixed per (problem, seed)."""
        cands = list(candidates_for(layout, self.dtype))
        if not cands:
            return []
        key = self._zlib.crc32(f"{M},{N},{K},{layout},{self.seed}".encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(key)
        scores = rng.random(len(cands))
        order = np.argsort(scores)[::-1]
        return [{**cands[i], "score": float(scores[i])} for i in order]


class XGBRanker:
    """Score with model_A.ubj + metrics.json feature columns."""

    def __init__(self, model_dir: Path, dtype: str = "bf16"):
        import xgboost as xgb

        self.dtype = dtype
        num_cols, cat_cols = feature_meta_from_model_dir(model_dir)
        self.feat_cols = num_cols + cat_cols
        self.cat_cols = cat_cols
        ubj = model_dir / "model_A.ubj"
        if not ubj.exists():
            raise FileNotFoundError(f"no model_A.ubj in {model_dir}")
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(ubj))
        try:
            import torch

            if torch.cuda.is_available():
                self._model.set_params(device="cuda")
        except Exception:
            pass

    def rank(self, M: int, N: int, K: int, layout: str) -> list[dict]:
        """Score every valid candidate with the XGBoost model; return them best first with a score key."""
        import numpy as np
        import pandas as pd
        from features import CATEGORY_LEVELS

        cands = list(candidates_for(layout, self.dtype))
        if not cands:
            return []
        feats = _featurize_candidates(cands, M, N, K, layout, self.dtype)
        missing = [c for c in self.feat_cols if c not in feats.columns]
        if missing:
            raise KeyError(f"{self.model_dir.name}: featurizer missing {missing}")
        for col in self.cat_cols:
            feats[col] = pd.Categorical(
                feats[col].astype(str), categories=CATEGORY_LEVELS[col]
            )
        scores = self._model.predict(feats[self.feat_cols])
        order = np.argsort(scores)[::-1]
        return [{**cands[i], "score": float(scores[i])} for i in order]


def propose_model(
    problems: list[Problem],
    method: str,
    ranker,
    sched_map: dict[tuple[int, int, int, str], tuple[int, int, int]] | None = None,
) -> tuple[dict, list[Proposal], dict]:
    """Proposals of a learned ranker: its single highest-scoring config per problem.

    With sched_map, the rank-1 nvMMH scheduler arguments are copied onto each pick; problems
    without one fall back to CUTLASS defaults. Returns (configs, proposals, meta).
    """
    configs: dict[str, dict] = {}
    proposals: list[Proposal] = []
    skipped: list[dict] = []
    default_sched = CUTLASS_DEFAULT_SCHEDULER_ARGS

    total = len(problems)
    done = 0
    n_sched_fallback = 0
    t0 = time.time()

    for M, N, K, layout_name in problems:
        short_layout = layout_name.lower()
        ranked = ranker.rank(M, N, K, layout_name)
        if not ranked:
            skipped.append({
                "M": M, "N": N, "K": K, "layout": short_layout,
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
                    "M": M, "N": N, "K": K, "layout": short_layout,
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
                variant=0,
                config_name=cfg["name"],
                raster_order=raster,
                swizzle_size=swizzle,
                splits=splits,
                score=score,
            )
        )
        done += 1
        if done % 50 == 0:
            rate = done / (time.time() - t0)
            print(
                f"  {done}/{total} problems ({rate:.1f}/s, "
                f"eta {(total - done) / rate / 60:.1f} min)",
                flush=True,
            )
    if n_sched_fallback:
        print(
            f"  warning: {n_sched_fallback} problems missing nvMMH scheduler "
            f"(using CUTLASS defaults)",
            flush=True,
        )
    no_cand = sum(1 for s in skipped if s["reason"] == "no_ranked_candidate")
    if no_cand:
        print(f"  warning: {no_cand} problems had no ranked candidate", flush=True)
    meta = {
        "n_proposed": len(proposals),
        "n_sched_fallback": n_sched_fallback,
        "n_no_candidate": no_cand,
        "skipped": skipped,
    }
    return configs, proposals, meta


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Ordered candidate kernels per method")
    ap.add_argument(
        "--method",
        required=True,
        help="label stored in eval_runs, e.g. nvmmh, mlp_full, xgb_structural",
    )
    ap.add_argument(
        "--backend",
        choices=["nvmmh", "mlp", "xgb", "ridge", "random"],
        default=None,
        help="inferred from --method when omitted",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="directory with metrics.json + model artifact (mlp/xgb backends)",
    )
    ap.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    ap.add_argument(
        "--problems",
        type=Path,
        default=None,
        help="explicit (M,N,K,layout) list — skips shapes×layouts expansion",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--limit", type=int, default=None, help="cap the shape count (debug)"
    )
    ap.add_argument(
        "--gpu",
        default="H100_SXM",
        help="nvMMH SKU preset; among SM90 SXM presets the top-1 pick differs "
        "on <6%% of problems",
    )
    ap.add_argument(
        "--nvmmh-top-k",
        type=int,
        default=1,
        help="nvMMH ranks to materialize (each fans out into schedule variants)",
    )
    ap.add_argument(
        "--scheduler-from",
        type=Path,
        default=None,
        help="proposals_nvmmh.json — copy rank-1 raster/swizzle/splits onto model picks",
    )
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp8_e4m3"])
    ap.add_argument(
        "--layouts",
        default=None,
        help="comma-separated layouts to evaluate (default: all four)",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="base seed for --backend random (per-problem draw is deterministic)",
    )
    args = ap.parse_args()
    layout_names = None
    if args.layouts:
        layout_names = tuple(x.strip() for x in args.layouts.split(",") if x.strip())
        bad = [x for x in layout_names if x not in LAYOUTS]
        if bad:
            print(
                f"ERROR: unknown layout(s) {bad}; choose from {list(LAYOUTS)}",
                file=sys.stderr,
            )
            return 1

    backend = args.backend
    if backend is None:
        if args.method == "nvmmh":
            backend = "nvmmh"
        elif args.method.startswith("mlp"):
            backend = "mlp"
        elif args.method.startswith("xgb"):
            backend = "xgb"
        elif args.method.startswith("ridge"):
            backend = "ridge"
        elif args.method in ("random_pick", "random"):
            backend = "random"
        else:
            print(
                "ERROR: cannot infer --backend; pass --backend explicitly",
                file=sys.stderr,
            )
            return 1

    if args.problems:
        if args.layouts:
            print("ERROR: --layouts is incompatible with --problems", file=sys.stderr)
            return 1
        problems = load_problems(args.problems, args.limit)
        print(f"{len(problems)} explicit problems from {args.problems}")
    else:
        shapes = load_shapes(args.shapes, args.limit)
        layout_items = layout_names or tuple(LAYOUTS.keys())
        problems = shapes_to_problems(shapes, layout_items)
        print(
            f"{len(shapes)} shapes x {len(layout_items)} layouts = {len(problems)} problems"
        )

    propose_meta: dict = {}
    if backend == "nvmmh":
        configs, proposals = propose_nvmmh(
            problems,
            args.gpu,
            args.nvmmh_top_k,
            args.dtype,
        )
    else:
        if backend in ("mlp", "xgb", "ridge"):
            if args.model_dir is None:
                print(
                    f"ERROR: --backend {backend} requires --model-dir",
                    file=sys.stderr,
                )
                return 1
            if not args.model_dir.is_dir():
                print(f"ERROR: model dir {args.model_dir} not found", file=sys.stderr)
                return 1
        sched_map = (
            sched_map_from_nvmmh(args.scheduler_from) if args.scheduler_from else None
        )
        if sched_map:
            print(
                f"  scheduler from {args.scheduler_from.name} ({len(sched_map)} problems)"
            )
        if backend == "mlp":
            ranker = MetricsMLPRanker(args.model_dir, dtype=args.dtype)
        elif backend == "xgb":
            ranker = XGBRanker(args.model_dir, dtype=args.dtype)
        elif backend == "ridge":
            ranker = RidgeRanker(args.model_dir, dtype=args.dtype)
        else:
            ranker = RandomRanker(seed=args.random_seed, dtype=args.dtype)
        configs, proposals, propose_meta = propose_model(
            problems,
            args.method,
            ranker,
            sched_map,
        )
        if backend == "random":
            propose_meta["random_seed"] = args.random_seed

    out = args.out or DEFAULT_OUT_DIR / f"proposals_{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "method": args.method,
                "dtype": args.dtype,
                "gpu": args.gpu if backend == "nvmmh" else None,
                "model_dir": str(args.model_dir) if args.model_dir else None,
                "configs": configs,
                "proposals": [asdict(p) for p in proposals],
                "propose_meta": propose_meta,
            }
        )
    )

    per_problem = len(proposals) / max(len(problems), 1)
    print(
        f"wrote {len(proposals)} proposals ({per_problem:.1f} per problem), "
        f"{len(configs)} unique kernels -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
