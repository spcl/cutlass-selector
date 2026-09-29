#!/usr/bin/env python3
"""Generate Eval2 zero-shot fusion problems: held-out shapes × unseen fusion kinds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC / "eval"))
from shapes import write_json_atomic  # noqa: E402

sys.path.insert(0, str(SRC / "autotuner"))
from config_space_fusion import ZERO_SHOT_FUSION_KINDS  # noqa: E402

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.gen_eval_shapes import (  # noqa: E402
    align_for_dtype,
    gen_shapes_tn,
    sweep_benched_shapes,
)
from transfer_fusion.study_common import (  # noqa: E402
    DTYPE_TAGS,
    EVAL2_N_SHAPES,
    EVAL2_SHAPE_SEED,
    STUDY_ROOT,
    eval_shapes_path,
    load_shapes,
    sweep_db_path,
)


def expand_shape_grid(
    shapes: list[tuple[int, int, int]],
    fusion_kinds: tuple[str, ...],
) -> tuple[list[tuple[int, int, int]], list[str], list[int]]:
    out_shapes: list[tuple[int, int, int]] = []
    out_kinds: list[str] = []
    out_variants: list[int] = []
    for shape in shapes:
        for vi, kind in enumerate(fusion_kinds):
            out_shapes.append(shape)
            out_kinds.append(kind)
            out_variants.append(vi)
    return out_shapes, out_kinds, out_variants


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval2 zero-shot fusion shape grid")
    ap.add_argument("--autotuner", type=Path, default=Path.home() / "autotuner")
    ap.add_argument("--n-shapes", type=int, default=EVAL2_N_SHAPES)
    ap.add_argument("--seed", type=int, default=EVAL2_SHAPE_SEED)
    ap.add_argument(
        "--fusion-kinds",
        nargs="+",
        default=list(ZERO_SHOT_FUSION_KINDS),
        help="zero-shot fusion kinds (not in training sweep)",
    )
    ap.add_argument("--dtype", choices=list(DTYPE_TAGS) + ["all"], default="all")
    args = ap.parse_args()

    fusion_kinds = tuple(args.fusion_kinds)
    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]

    for dtype in dtypes:
        master = STUDY_ROOT / "dtype" / dtype / "training_shapes.json"
        if not master.is_file():
            raise SystemExit(f"missing {master} — run make_subsets.py first")
        train_shapes = set(load_shapes(master))
        db = sweep_db_path(dtype, args.autotuner)
        if not db.is_file():
            raise SystemExit(f"missing {db}")
        exclude = train_shapes | sweep_benched_shapes(db)
        align = align_for_dtype(dtype)
        base_shapes = gen_shapes_tn(args.n_shapes, args.seed, exclude, align)

        bad = [s for s in base_shapes if s[0] % align or s[2] % align]
        if bad:
            raise SystemExit(f"{dtype}: {len(bad)} TN shapes violate align={align}")
        leaked = [s for s in base_shapes if s in exclude]
        if leaked:
            raise SystemExit(f"{dtype}: {len(leaked)} shapes overlap training/holdout")

        shapes, kinds, variants = expand_shape_grid(base_shapes, fusion_kinds)
        out = eval_shapes_path(dtype, suite="eval2")
        payload = {
            "protocol": "zero_shot_fusion_grid",
            "suite": "eval2",
            "shapes": [list(s) for s in shapes],
            "fusion_kinds": kinds,
            "variants": variants,
            "zero_shot_kinds": list(fusion_kinds),
            "train_kinds": ["linear", "relu", "bias", "bias_relu", "bias_gelu"],
            "layout": "TN",
            "dtype": dtype,
            "n_shapes": len(base_shapes),
            "n_problems": len(shapes),
            "n_kinds": len(fusion_kinds),
            "seed": args.seed,
            "align": align,
        }
        write_json_atomic(out, payload)
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(payload, indent=2) + "\n")

        nvmmh_out = out.parent / "shapes_nvmmh.json"
        nvmmh_payload = {
            "protocol": "zero_shot_nvmmh_unique_shapes",
            "shapes": [list(s) for s in base_shapes],
            "layout": "TN",
            "dtype": dtype,
            "n": len(base_shapes),
            "seed": args.seed,
            "align": align,
        }
        write_json_atomic(nvmmh_out, nvmmh_payload)

        print(
            f"{dtype}: wrote {len(base_shapes)} unique shapes × {len(fusion_kinds)} kinds "
            f"= {len(shapes)} problems -> {out}"
        )
        print(f"  zero-shot kinds: {fusion_kinds}")
        print(f"  nvmmh unique shapes -> {nvmmh_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
