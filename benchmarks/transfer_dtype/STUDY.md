# Dtype transfer study

## Submit (copy-paste)

Run all commands from the repository root.

### Training

**Prep** (~1–2 h, CPU-heavy; no GPU training):

```bash
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_dtype/slurm_study_train_fp32.sh
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_dtype/slurm_study_train_fp8.sh
```

Wait for both prep jobs to finish, then **train** (24 parallel tasks per dtype, one run each):

```bash
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_dtype/slurm_study_train_array_fp32.sh
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_dtype/slurm_study_train_array_fp8.sh
```

**Alternative — one sequential job per dtype** (~8–12 h, prep + all 24 finetune runs):

```bash
sbatch --time=18:00:00 benchmarks/transfer_dtype/slurm_study_train_fp32.sh
sbatch --time=18:00:00 benchmarks/transfer_dtype/slurm_study_train_fp8.sh
```

### Evaluation (after all training checkpoints exist)

Eval uses **1000 held-out TN shapes per dtype** (1000 problems each — not 4-layout broad GEMM).
Override: `EVAL_N_SHAPES=500` (etc.) on prep.

**Prep** (eval shapes + nvMMH proposals + nvMMH bench, ~1–2 h):

```bash
STUDY_EVAL_STEPS=prep sbatch --time=01:00:00 benchmarks/transfer_dtype/slurm_study_eval_fp32.sh
STUDY_EVAL_STEPS=prep sbatch --time=01:00:00 benchmarks/transfer_dtype/slurm_study_eval_fp8.sh
```

**Bench** (24 parallel tasks per dtype):

```bash
sbatch --time=02:00:00 --array=0-23%6 benchmarks/transfer_dtype/slurm_study_eval_array_fp32.sh
sbatch --time=02:00:00 --array=0-23%6 benchmarks/transfer_dtype/slurm_study_eval_array_fp8.sh
```

**Summarize** (after all eval array tasks finish):

```bash
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_dtype/slurm_study_eval_fp32.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_dtype/slurm_study_eval_fp8.sh
```

Local plots (after summarize):

```bash
# from the repository root
python benchmarks/transfer_dtype/plot_study.py --also-quick
# paper PDFs → artifacts/analysis/paper/figures/transfer_*.pdf
```

`--array=0-23%6` runs at most **6 tasks at once** (24 finetune runs per dtype). Omit `%6` for full parallelism,
or use one sequential job (`sbatch --time=18:00:00 .../slurm_study_train_fp32.sh`) for a single GPU slot.

Monitor: `squeue -u $USER`

---

## Design

| Axis | Values |
|------|--------|
| Target dtypes | FP32 TN, FP8 E4M3 TN |
| Sweep DBs | `~/autotuner/autotuner_sweep_fp32_tn.db`, `~/autotuner/autotuner_sweep_fp8_tn.db` |
| Training fractions | 1%, 5%, 10%, 25%, 50%, 100% of **593 base shapes** |
| Subset seed | 0 (nested subsets) |
| MLP | finetune from `artifacts/analysis/paper/mlp_mse{,_structural}` → **2 per fraction** |
| XGB | warm-start from `artifacts/analysis/paper/xgb_mse{,_structural}` (+200 trees) → **2 per fraction** |

**24 finetune runs per dtype** (48 total). Starters: `artifacts/analysis/paper/mlp_mse{,_structural}/model_mlp.pt`,
`artifacts/analysis/paper/xgb_mse{,_structural}/model_A.ubj`.

Uses `.venv/bin/python`, `#SBATCH --environment=vsch`, no `SCRATCH`.

Outputs: `artifacts/analysis/paper/transfer_study/runs/<run_id>/` (train),
`transfer_study/eval/` (bench; nvMMH in `autotuner_transfer_<dtype>.db`, each model
run in `eval/shards/<dtype>/<run_id>.db`), `transfer_study/results/` + `plots/` (summarize).

Prep benches nvMMH on node-local `/dev/shm` and checkpoints to `$HOME` (SQLite WAL is
unreliable on the home filesystem).
