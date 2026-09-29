#!/usr/bin/env python3
"""Discover MLP/XGB checkpoints from the capacity sweep for DeepBench eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from repo_paths import CAPACITY

DEFAULT_CAPACITY_DIR = CAPACITY

# Matches benchmarks/eval/plot_capacity_results.py ordering.
MLP_WIDTHS = ("16x8", "32x16", "64x32", "128x64", "256x128x64", "1024x1024x512x256")
XGB_DEPTHS = ("d2", "d3", "d4", "d6", "d8", "d11", "d14")

# Largest MLP width is the same architecture as artifacts/analysis/paper/mlp_mse{,_structural}.
PAPER_MLP_WIDTH = "1024x1024x512x256"


@dataclass(frozen=True)
class CapacityModel:
    method_id: str
    family: str  # mlp | xgb
    capacity: str
    feature_set: str  # full | structural
    seed: int
    model_dir: Path

    @property
    def backend(self) -> str:
        return self.family


def _mlp_dir(capacity_dir: Path, width: str, feature_set: str, seed: int) -> Path:
    return capacity_dir / f"mlp_mse_{width}_{feature_set}_s{seed}"


def _xgb_dir(capacity_dir: Path, depth: str, feature_set: str, seed: int) -> Path:
    return capacity_dir / f"xgb_mse_{depth}_{feature_set}_s{seed}"


def _valid_checkpoint(model_dir: Path, family: str) -> bool:
    metrics = model_dir / "metrics.json"
    if not metrics.is_file():
        return False
    if family == "mlp":
        return (model_dir / "model_mlp.pt2").is_file() or (model_dir / "model_mlp.pt").is_file()
    if family == "xgb":
        return (model_dir / "model_A.ubj").is_file()
    return False


def discover_capacity_models(
    capacity_dir: Path = DEFAULT_CAPACITY_DIR,
    *,
    seed: int = 42,
    skip_paper_mlp_width: bool = True,
) -> list[CapacityModel]:
    """Return capacity-sweep checkpoints present on disk for one seed."""
    capacity_dir = capacity_dir.resolve()
    models: list[CapacityModel] = []

    for width in MLP_WIDTHS:
        if skip_paper_mlp_width and width == PAPER_MLP_WIDTH:
            continue
        for feature_set in ("full", "structural"):
            model_dir = _mlp_dir(capacity_dir, width, feature_set, seed)
            if not _valid_checkpoint(model_dir, "mlp"):
                continue
            models.append(
                CapacityModel(
                    method_id=f"mlp_cap_{width}_{feature_set}",
                    family="mlp",
                    capacity=width,
                    feature_set=feature_set,
                    seed=seed,
                    model_dir=model_dir,
                )
            )

    for depth in XGB_DEPTHS:
        for feature_set in ("full", "structural"):
            model_dir = _xgb_dir(capacity_dir, depth, feature_set, seed)
            if not _valid_checkpoint(model_dir, "xgb"):
                continue
            models.append(
                CapacityModel(
                    method_id=f"xgb_cap_{depth}_{feature_set}",
                    family="xgb",
                    capacity=depth,
                    feature_set=feature_set,
                    seed=seed,
                    model_dir=model_dir,
                )
            )

    return models


def write_manifest(models: list[CapacityModel], out_path: Path) -> None:
    payload = {
        "models": [
            {
                "method_id": m.method_id,
                "family": m.family,
                "capacity": m.capacity,
                "feature_set": m.feature_set,
                "seed": m.seed,
                "model_dir": str(m.model_dir),
            }
            for m in models
        ]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
