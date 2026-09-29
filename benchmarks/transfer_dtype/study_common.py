"""Shared constants and helpers for the dtype transfer study."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from repo_paths import ANALYSIS

ANALYSIS_ROOT = ANALYSIS
STUDY_ROOT = ANALYSIS_ROOT / "paper" / "transfer_study"
PAPER_ROOT = ANALYSIS_ROOT / "paper"

FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
# Single nested-subset seed (no time for multi-seed low-fraction repeats).
STUDY_SEEDS = (0,)

DTYPE_TAGS = {
    "fp32": ("sweep_fp32_tn", "fp32"),
    "fp8_e4m3": ("sweep_fp8_tn", "fp8_e4m3"),
}

# Target-dtype TN sweep DBs on Daint (~/autotuner/autotuner_sweep_fp32_tn.db, etc.).
def sweep_db_path(dtype: str, autotuner: Path | None = None) -> Path:
    root = autotuner or (Path.home() / "autotuner")
    tag = DTYPE_TAGS[dtype][0]
    return root / f"autotuner_{tag}.db"

STRUCTURAL_NUMERIC = [
    "log2_M", "log2_N", "log2_K",
    "tile_m", "tile_n", "tile_k", "stages", "cluster_m_f", "cluster_n_f",
]
STRUCTURAL_CATEGORICAL = [
    "kernel_schedule", "epilogue_schedule", "scheduler", "sched_class", "layout",
]

PRETRAIN_MLP = {
    "full": PAPER_ROOT / "mlp_mse" / "model_mlp.pt",
    "structural": PAPER_ROOT / "mlp_mse_structural" / "model_mlp.pt",
}
PRETRAIN_XGB = {
    "full": PAPER_ROOT / "xgb_mse" / "model_A.ubj",
    "structural": PAPER_ROOT / "xgb_mse_structural" / "model_A.ubj",
}

MLP_EPOCHS_PRETRAIN = 48
MLP_EPOCHS_SCRATCH = 144
MLP_LR_PRETRAIN = 4e-5
MLP_LR_SCRATCH = 0.0001191129211393104
XGB_EXTRA_TREES = 200  # continue boosting from BF16 starter (artifacts/analysis/paper/xgb_mse)


@dataclass(frozen=True)
class RunSpec:
    dtype: str
    fraction: float
    seed: int
    family: str  # mlp | xgb
    features: str  # full | structural
    init: str  # pretrain (finetune only)

    @property
    def frac_pct(self) -> int:
        return int(round(self.fraction * 100))

    @property
    def run_id(self) -> str:
        return (
            f"{self.dtype}_f{self.frac_pct:03d}_s{self.seed}_"
            f"{self.family}_{self.features}_{self.init}"
        )

    @property
    def out_dir(self) -> Path:
        return STUDY_ROOT / "runs" / self.run_id

    @property
    def eval_method(self) -> str:
        return self.run_id


def dtype_dir(dtype: str) -> Path:
    return STUDY_ROOT / "dtype" / dtype


def features_path(dtype: str) -> Path:
    return dtype_dir(dtype) / "features_full.parquet"


def subset_path(dtype: str, seed: int, fraction: float) -> Path:
    pct = int(round(fraction * 100))
    return STUDY_ROOT / "subsets" / dtype / f"seed{seed}" / f"f{pct:03d}.json"


def seeds_for_fraction(fraction: float) -> tuple[int, ...]:
    return STUDY_SEEDS


def iter_run_specs() -> list[RunSpec]:
    """6 fractions × (2 MLP + 2 XGB finetune) = 24 runs per dtype."""
    specs: list[RunSpec] = []
    for dtype in DTYPE_TAGS:
        for fraction in FRACTIONS:
            for seed in seeds_for_fraction(fraction):
                for features in ("full", "structural"):
                    specs.append(RunSpec(dtype, fraction, seed, "mlp", features, "pretrain"))
                    specs.append(RunSpec(dtype, fraction, seed, "xgb", features, "pretrain"))
    return specs


def load_shapes(path: Path) -> list[tuple[int, int, int]]:
    data = json.loads(path.read_text())
    return [tuple(x) for x in data["shapes"]]


def save_shapes(path: Path, shapes: list[tuple[int, int, int]], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "shapes": [list(s) for s in shapes],
        **meta,
    }, indent=2) + "\n")


def nested_subset(shapes: list[tuple[int, int, int]], fraction: float, seed: int) -> list[tuple[int, int, int]]:
    shapes = sorted(shapes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(shapes))
    n = max(1, int(round(len(shapes) * fraction)))
    return [shapes[i] for i in order[:n]]


def scaler_policy_path() -> Path:
    return STUDY_ROOT / "scaler_policy.json"


def load_scaler_policy() -> str:
    path = scaler_policy_path()
    if not path.is_file():
        return "fit"
    return json.loads(path.read_text())["policy"]


EVAL_ROOT = STUDY_ROOT / "eval"
RESULTS_ROOT = STUDY_ROOT / "results"
PLOTS_ROOT = STUDY_ROOT / "plots"

# TN-only broad eval shape count (one problem per shape).
EVAL_N_SHAPES = 1000
EVAL_SHAPE_SEED = 0

CURVE_LABELS: dict[str, str] = {
    "mlp_pretrain_full": "MLP finetune hardware-aware",
    "mlp_pretrain_structural": "MLP finetune structural",
    "xgb_full": "XGBoost hardware-aware",
    "xgb_structural": "XGBoost structural",
}

CURVE_ORDER = list(CURVE_LABELS.keys())


def eval_shapes_path(dtype: str) -> Path:
    return EVAL_ROOT / dtype / "shapes.json"


def eval_db_home_path(dtype: str) -> Path:
    """Durable eval DB on $HOME (nvMMH baseline + legacy merged rows)."""
    return EVAL_ROOT / f"autotuner_transfer_{dtype}.db"


def eval_db_shard_path(spec: RunSpec) -> Path:
    """Per-run shard DB so parallel Slurm array tasks do not share one SQLite file."""
    return EVAL_ROOT / "shards" / spec.dtype / f"{spec.run_id}.db"


def eval_db_live_path(dtype: str) -> Path:
    """Live DB for the current job (node-local when CKS_TRANSFER_EVAL_DB is set)."""
    override = os.environ.get("CKS_TRANSFER_EVAL_DB")
    if override:
        return Path(override)
    return eval_db_home_path(dtype)


def eval_db_path(dtype: str) -> Path:
    return eval_db_home_path(dtype)


def eval_prep_dir(dtype: str) -> Path:
    return EVAL_ROOT / dtype


def proposals_nvmmh_path(dtype: str) -> Path:
    return eval_prep_dir(dtype) / "proposals_nvmmh.json"


def proposals_model_path(spec: RunSpec) -> Path:
    return eval_prep_dir(spec.dtype) / f"proposals_{spec.run_id}.json"


def curve_key(spec: RunSpec) -> str | None:
    if spec.family == "mlp":
        return f"mlp_{spec.init}_{spec.features}"
    if spec.family == "xgb":
        return f"xgb_{spec.features}"
    return None


def run_manifest_path() -> Path:
    return STUDY_ROOT / "run_manifest.json"


def write_run_manifest() -> Path:
    specs = iter_run_specs()
    payload = {
        "n_runs": len(specs),
        "runs": [
            {
                "run_id": s.run_id,
                "dtype": s.dtype,
                "fraction": s.fraction,
                "seed": s.seed,
                "family": s.family,
                "features": s.features,
                "init": s.init,
                "curve_key": curve_key(s),
                "out_dir": str(s.out_dir),
            }
            for s in specs
        ],
    }
    run_manifest_path().parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path().write_text(json.dumps(payload, indent=2) + "\n")
    return run_manifest_path()
