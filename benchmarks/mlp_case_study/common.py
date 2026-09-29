"""Shared helpers for the MLP inference GEMM case study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from repo_paths import CAPACITY, PAPER

OUT_ROOT = PAPER / "mlp_case_study"
BUILD_ROOT = OUT_ROOT / "build"
DB_PATH = OUT_ROOT / "case_study.db"

# Representative CUTLASS problem used to drive real selector inference (batch size
# is shape-independent; internal MLP GEMMs depend only on n_candidates).
REF_GEMM = (4096, 4096, 4096)
REF_LAYOUT = "TN"
SELECTOR_DTYPE = "bf16"

CAPACITY_WIDTHS = (
    "16x8",
    "32x16",
    "64x32",
    "128x64",
    "256x128x64",
    "1024x1024x512x256",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint: Path
    feature_set: str
    capacity: str | None = None
    seed: int | None = None

    @property
    def metrics_path(self) -> Path:
        return self.checkpoint / "metrics.json"

    def load_metrics(self) -> dict:
        return json.loads(self.metrics_path.read_text())


def discover_models(seed: int = 42) -> list[ModelSpec]:
    models: list[ModelSpec] = [
        ModelSpec(
            "paper_mlp_mse_full",
            PAPER / "mlp_mse",
            "full",
        ),
        ModelSpec(
            "paper_mlp_mse_structural",
            PAPER / "mlp_mse_structural",
            "structural",
        ),
    ]
    for width in CAPACITY_WIDTHS:
        for feat in ("full", "structural"):
            ckpt = f"{CAPACITY}/mlp_mse_{width}_{feat}_s{seed}"
            if not ckpt.is_dir():
                continue
            models.append(
                ModelSpec(
                    f"capacity_{width}_{feat}_s{seed}",
                    ckpt,
                    feat,
                    capacity=width,
                    seed=seed,
                )
            )
    return models


def gemm_signature(M: int, N: int, K: int, layout: str, dtype: str, accum_dtype: str) -> str:
    return f"{M}x{N}x{K}:{layout}:{dtype}:{accum_dtype}"
