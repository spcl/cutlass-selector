# DeepBench external GEMM evaluation

Feasibility analysis: `benchmarks/deepbench/analyze_deepbench.py` writes `datasets/deepbench/ANALYSIS.md`.

## Problem set

`deepbench_compatibility.csv` lists **153 accepted** BF16-compatible rows with
\(\min(M,N,K)\ge 32\) (batch-1 RNN and \(K{=}500{,}000\) vocabulary projections
are omitted from the catalogue). Export dedupes to **151 unique
`(M,N,K,layout)`** problems (107 train / 39 inference_server / 5 inference_device).

Export locally:

```bash
python benchmarks/deepbench/export_problems.py
# -> datasets/deepbench/problems.json
```

## Cluster run

Copy frozen model artifacts to `artifacts/analysis/paper/` on Daint (`mlp_mse`, `xgb_mse`, etc.).

```bash
# full run: paper baselines (default)
#   nvmmh, mlp_full/structural, xgb_full/structural, ridge_full/structural, random_pick
sbatch benchmarks/deepbench/slurm_deepbench.sh

# paper baselines + capacity-sweep MLP/XGB (seed 42, artifacts/analysis/capacity/)
sbatch benchmarks/deepbench/slurm_deepbench_capacity.sh

# subset only
EVAL_METHODS=mlp_full,xgb_full sbatch benchmarks/deepbench/slurm_deepbench.sh

# local proposal prep (CPU only)
python benchmarks/deepbench/propose_methods.py \
  --prep-dir artifacts/eval/out/deepbench_local \
  --problems datasets/deepbench/problems.json \
  --include-capacity

# resume compile/bench
SKIP_PREP=1 EVAL_PREP_DIR=artifacts/eval/out/deepbench_<job_id> PHASE=compile,bench sbatch benchmarks/deepbench/slurm_deepbench.sh
```

Outputs land in `artifacts/eval/out/deepbench_<job_id>/` (`problems.json`, `proposals_*.json`, `report.csv`).
DB: `~/autotuner/autotuner_deepbench.db`.

## Local prep (CPU only)

```bash
python benchmarks/deepbench/export_problems.py
python src/eval/propose.py --method nvmmh --gpu H100_SXM \
  --problems datasets/deepbench/problems.json \
  --out artifacts/eval/out/deepbench_local/proposals_nvmmh.json
```
