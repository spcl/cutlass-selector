# Fusion transfer — Eval2 (zero-shot epilogue transfer)

Eval1 measures **in-vocabulary** transfer: held-out TN shapes with fusion kinds from the
training sweep (`linear`, `relu`, `bias`, `bias_relu`, `bias_gelu`).

Eval2 measures **zero-shot epilogue transfer**: same finetuned checkpoints, but eval problems
use fusion kinds that were **never in the training sweep**:

| Eval2 kind | CUTLASS op | vs training |
|---|---|---|
| `silu` | `LinCombPerRowBiasEltAct<SiLu>` | new activation |
| `bias_silu` | same + bias | new activation + smem carveout |
| `tanh` | `LinCombPerRowBiasEltAct<Tanh>` | new activation |
| `bias_tanh` | same + bias | new activation + smem carveout |

The model has **no `fusion_kind` categorical** — transfer relies entirely on numeric features
(`smem_total`, bytes, tile geometry) computed from each candidate's stamped fusion op.

---

## Problem grid

```
200 held-out TN shapes  ×  4 zero-shot kinds  =  800 problems per dtype
```

- Shapes: seed `0xF0510C`, disjoint from training + fusion sweep (same rules as Eval1).
- nvMMH baseline: benchmarked once per **unique** shape (`shapes_nvmmh.json`, 200 problems).
- Model: rank-1 fused kernel per (shape, kind); `variant` column disambiguates kinds at same shape.

Outputs: `transfer_fusion_study/eval2/` (DBs, shapes) and `transfer_fusion_study/results_eval2/`.

---

## Commands (Daint)

**Prerequisite:** the fine-tuned fusion checkpoints under `transfer_fusion_study/runs/`
(from [STUDY.md](STUDY.md)). Run all commands from the repository root.

### Eval1 (in-vocabulary)

```bash
# Prep: shapes + nvMMH proposals + nvMMH bench (~1–2 h per dtype)
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp16.sh
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp32.sh
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp8.sh

# Model eval (24 runs per dtype, parallel)
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp16.sh
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp32.sh
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp8.sh

# Summarize + plots (after all array tasks finish)
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval_fp16.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval_fp32.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval_fp8.sh
```

Results: `artifacts/analysis/paper/transfer_fusion_study/results/SUMMARY.md`

### Eval2 (zero-shot)

Uses the **same** finetuned checkpoints — no retrain.

```bash
# Prep: zero-shot shape grid + nvMMH (~1–2 h per dtype)
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval2_fp16.sh
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval2_fp32.sh
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval2_fp8.sh

# Model eval (24 runs per dtype)
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval2_array_fp16.sh
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval2_array_fp32.sh
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval2_array_fp8.sh

# Summarize (per fusion kind breakdown)
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval2_fp16.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval2_fp32.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval2_fp8.sh
```

Results: `artifacts/analysis/paper/transfer_fusion_study/results_eval2/SUMMARY.md`

### Local sanity (optional)

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/benchmarks
python benchmarks/transfer_fusion/gen_eval2_shapes.py --dtype fp16
python benchmarks/transfer_fusion/summarize_eval2.py --dtype fp16
```

---

## Metrics

| Metric | Eval1 | Eval2 |
|---|---|---|
| Headline | geomean speedup vs nvMMH (1000 problems) | same (800 problems) |
| Stratification | by dtype / fraction / model | **+ per zero-shot `fusion_kind`** |
| Baseline | unfused nvMMH | same (unique shapes only) |

**Hypothesis:** hardware-aware model (full features) ≥ structural on zero-shot kinds where smem
carveout matters (`bias_silu`, `bias_tanh`).

---

## Implementation

| Script | Role |
|---|---|
| `gen_eval2_shapes.py` | 200 shapes × 4 zero-shot kinds → `eval2/<dtype>/shapes.json` |
| `study_eval2_job.sh` | Sets `STUDY_EVAL_SUITE=eval2`, sources shared eval driver |
| `summarize_eval2.py` | Per-run × per-kind speedup table |
| `config_space_fusion.py` | `ZERO_SHOT_FUSION_KINDS` + CUTLASS C++ types |

Eval1 and Eval2 share `run_eval_grid.py` / `eval_one.py`; suite selected via `STUDY_EVAL_SUITE`.
