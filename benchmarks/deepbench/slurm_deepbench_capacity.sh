#!/bin/bash
#SBATCH --job-name=deepbench-cap
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G

#
# DeepBench eval with paper baselines + capacity-sweep MLP/XGB checkpoints.
# Capacity models live under artifacts/analysis/capacity/ (mlp_mse_<width>_* and xgb_mse_d*_*).
#
#   sbatch benchmarks/deepbench/slurm_deepbench_capacity.sh
#
# Capacity checkpoints only (no paper mlp_full / xgb_full reruns):
#   DEEPBENCH_CAPACITY_ONLY=1 sbatch benchmarks/deepbench/slurm_deepbench_capacity.sh
#
# Resume compile/bench:
#   SKIP_PREP=1 EVAL_PREP_DIR=artifacts/eval/out/deepbench_cap_<job_id> PHASE=compile,bench \
#     sbatch benchmarks/deepbench/slurm_deepbench_capacity.sh

export DEEPBENCH_INCLUDE_CAPACITY=1
export CKS_TAG="${CKS_TAG:-deepbench_cap}"
export DB="${DB:-$HOME/autotuner/autotuner_deepbench_cap.db}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
exec bash "${REPO_ROOT}/benchmarks/deepbench/slurm_deepbench.sh"
