# Reproducing the paper on CSCS Alps

These are the Slurm jobs and analysis scripts that produced the paper's data, models and
results on the GH200 nodes of CSCS Alps (Daint and Clariden). They drive the library in
`src/`, which the [top-level README](../README.md) explains.

## Before you start

```bash
git clone --recursive https://github.com/spcl/cutlass-selector.git
cd cutlass-selector
uenv run pytorch/v2.9.1:v2 -- python -m venv --system-site-packages .venv
uenv run pytorch/v2.9.1:v2 -- .venv/bin/pip install -r requirements.txt nvidia-matmul-heuristics
mkdir -p ~/autotuner
export PYTHONPATH=$PWD/src:$PWD/benchmarks   # for commands you run directly
```

- Every job has `#SBATCH --account=YOUR_ACCOUNT`; replace it with your project account.
- Submit every job from the repository root: Slurm copies only the batch script, and the
  jobs find everything else through `$SLURM_SUBMIT_DIR`.
- Jobs are resumable. Each restores its registry from `~/autotuner/autotuner_<tag>.db`,
  works on a copy in node-local `/dev/shm`, checkpoints to `$SCRATCH` every 10 minutes
  and writes the result back on exit, so a timed-out job can simply be resubmitted.

The jobs assume Alps: the `pytorch/v2.9.1:v2` uenv, the `vsc` / `vsch` container
environments, `~/autotuner/` for registries and `$SCRATCH` for build trees.

## Main results

**1. Measure.** BF16 training data (4 nodes) and the exhaustively measured oracle:

```bash
sbatch benchmarks/data_collection/run_bf16_final.sh        # tag bf16_final
sbatch benchmarks/data_collection/run_bf16_eval.sh         # tag bf16_eval: 17 shapes x 4 layouts x every config
sbatch benchmarks/data_collection/run_bf16_eval_finish.sh  # finishes the oracle on one node if needed
sbatch benchmarks/data_collection/slurm_ncu.sh             # Nsight Compute counters (optional)
```

**2. Train.**

```bash
python src/model/features.py \
    --db ~/autotuner/autotuner_bf16_final.db --train-tags bf16_final \
    --eval-db ~/autotuner/autotuner_bf16_eval.db --eval-tag bf16_eval \
    --eval-min-configs 8000 --out artifacts/analysis/paper/features.parquet
sbatch benchmarks/training/train_sweep.sh      # every model family x objective
CKS_FEATURE_SETS="full structural" sbatch benchmarks/training/train_sweep.sh   # + structural ablation
sbatch benchmarks/training/capacity_sweep.sh   # model size vs feature set
```

**3. Evaluate.**

```bash
sbatch benchmarks/eval/slurm_eval.sh           # held-out GEMMs: nvMMH, cuBLASLt and the four selectors
sbatch benchmarks/eval/run_capacity_eval.sh    # every capacity-sweep model on the oracle groups
```

Progress: `python src/autotuner/query_db.py --db ~/autotuner/autotuner_<tag>.db summary`.

## Studies

| Study | Folder | Start with |
|---|---|---|
| Cross-precision transfer (FP32, FP8) | `transfer_dtype/` | [`STUDY.md`](transfer_dtype/STUDY.md) |
| Epilogue-fusion transfer | `transfer_fusion/` | [`STUDY.md`](transfer_fusion/STUDY.md), [`STUDY_EVAL2.md`](transfer_fusion/STUDY_EVAL2.md) |
| DeepBench GEMMs | `deepbench/` | [`README.md`](deepbench/README.md) |
| GEMMs of MLP inference | `mlp_case_study/` | `slurm_case_study.sh` header |
| Linear, analytical and random selectors | `baselines/learned_and_analytical/` | `slurm_baselines_*.sh` headers |
| nvMatmulHeuristics regret by lookup | `baselines/nvmmh/` | `nvmmh_eval.py --help` |
| Measurement-budget experiment | `sampling/` | `run_sampling_experiment.sh` header |
| Hyperparameter search | `training/` | `tune_mlp.sh`, `tune_xgboost.sh` |

The transfer studies need their own sweeps first: `data_collection/plan_fp32_fp8.sh` with
`run_fp32.sh` / `run_fp8.sh`, and `data_collection/plan_fusion.sh` with `run_fusion.sh`.

### Reference baselines

```bash
sbatch benchmarks/baselines/cublas_lt/cublas_benchmark.sh          # cuBLASLt, best of 8 algorithms
sbatch benchmarks/baselines/cutlass_profiler/cutlass_benchmark.sh  # cutlass_profiler with its own heuristics
```

Both measure the same 19 reference shapes. The cuBLASLt job writes
`artifacts/analysis/paper/baselines/cublas_reference.csv`; the CUTLASS job builds
`cutlass_profiler` against `src/extern/cutlass` and summarises with `parse_results.py`.

## Reference

| Path | Contents |
|---|---|
| `~/autotuner/autotuner_<tag>.db` | registry per tag |
| `$SCRATCH/autotuner_build_<tag>*/` | compiled kernels |
| `artifacts/analysis/paper/` | features, trained models, figures |
| `artifacts/analysis/capacity/` | capacity-sweep models |
| `artifacts/eval/out/run_<job>/` | evaluation runs: shapes, proposals, report |

| Variable | Default | Meaning |
|---|---|---|
| `CKS_TAG` | per job | plan tag |
| `DB` | `~/autotuner/autotuner_<tag>.db` | registry |
| `CKS_GPUS` | `4` | GPUs per node |
| `CKS_COMPILE_JOBS` | `64` | concurrent `nvcc` processes |
| `CKS_NCU_TOP_K` | `20` | kernels profiled per shape |
| `CKS_CKPT_INTERVAL` | `600` | seconds between checkpoints (`0` disables) |
| `CKS_ROOT` | `$SLURM_SUBMIT_DIR` | repository root |
| `PYTHON` | `.venv/bin/python` | interpreter used by the jobs |

`common/slurm_common.sh` holds the shared job logic and the remaining knobs
(`CKS_WAVE_EFF`, `CKS_UENV`, `CKS_SRUN_ENV`).
