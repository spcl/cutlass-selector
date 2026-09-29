#!/usr/bin/env python3
"""Method registry for DeepBench eval (paper baselines + optional capacity sweep)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from capacity_models import CapacityModel, discover_capacity_models

from repo_paths import CAPACITY, PAPER

DEFAULT_PAPER_DIR = PAPER
DEFAULT_CAPACITY_DIR = CAPACITY


@dataclass(frozen=True)
class EvalMethod:
    method_id: str
    backend: str
    model_dir: Path | None = None
    random_seed: int | None = None

    @property
    def needs_model_dir(self) -> bool:
        return self.backend in ("mlp", "xgb", "ridge")


PAPER_BASELINES: tuple[EvalMethod, ...] = (
    EvalMethod("mlp_full", "mlp", DEFAULT_PAPER_DIR / "mlp_mse"),
    EvalMethod("mlp_structural", "mlp", DEFAULT_PAPER_DIR / "mlp_mse_structural"),
    EvalMethod("xgb_full", "xgb", DEFAULT_PAPER_DIR / "xgb_mse"),
    EvalMethod("xgb_structural", "xgb", DEFAULT_PAPER_DIR / "xgb_mse_structural"),
    EvalMethod("ridge_full", "ridge", DEFAULT_PAPER_DIR / "baselines" / "linear_full"),
    EvalMethod(
        "ridge_structural",
        "ridge",
        DEFAULT_PAPER_DIR / "baselines" / "linear_structural",
    ),
    EvalMethod("random_pick", "random", random_seed=42),
)

DEFAULT_METHOD_IDS = ("nvmmh",) + tuple(m.method_id for m in PAPER_BASELINES)


def capacity_to_eval_method(model: CapacityModel) -> EvalMethod:
    return EvalMethod(
        method_id=model.method_id,
        backend=model.family,
        model_dir=model.model_dir,
    )


def resolve_methods(
    method_ids: list[str] | None = None,
    *,
    include_capacity: bool = False,
    capacity_dir: Path = DEFAULT_CAPACITY_DIR,
    paper_dir: Path = DEFAULT_PAPER_DIR,
    capacity_seed: int = 42,
    random_seed: int = 42,
) -> list[EvalMethod]:
    """Expand method id list; `default` means paper baselines (+ nvmmh handled separately)."""
    paper_map = {m.method_id: m for m in _paper_methods(paper_dir, random_seed)}
    capacity_models = discover_capacity_models(
        capacity_dir, seed=capacity_seed, skip_paper_mlp_width=True
    )
    capacity_map = {m.method_id: capacity_to_eval_method(m) for m in capacity_models}

    if method_ids is None or method_ids == ["default"]:
        ids = [m.method_id for m in PAPER_BASELINES]
    else:
        ids = method_ids

    out: list[EvalMethod] = []
    for mid in ids:
        if mid == "nvmmh":
            continue
        if mid in paper_map:
            out.append(paper_map[mid])
        elif mid in capacity_map:
            out.append(capacity_map[mid])
        else:
            raise KeyError(f"unknown method id: {mid}")

    if include_capacity:
        seen = {m.method_id for m in out}
        for model in capacity_models:
            em = capacity_map[model.method_id]
            if em.method_id not in seen:
                out.append(em)
                seen.add(em.method_id)

    return out


def _paper_methods(paper_dir: Path, random_seed: int) -> list[EvalMethod]:
    return (
        EvalMethod("mlp_full", "mlp", paper_dir / "mlp_mse"),
        EvalMethod("mlp_structural", "mlp", paper_dir / "mlp_mse_structural"),
        EvalMethod("xgb_full", "xgb", paper_dir / "xgb_mse"),
        EvalMethod("xgb_structural", "xgb", paper_dir / "xgb_mse_structural"),
        EvalMethod("ridge_full", "ridge", paper_dir / "baselines" / "linear_full"),
        EvalMethod(
            "ridge_structural",
            "ridge",
            paper_dir / "baselines" / "linear_structural",
        ),
        EvalMethod("random_pick", "random", random_seed=random_seed),
    )
