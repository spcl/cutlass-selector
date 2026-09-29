#!/bin/bash
#SBATCH --job-name=mlp-case-bench
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# GPU bench: compile + measure our rank-1 kernels and all 8 nvMMH variants
# per traced MLP GEMM shape. Requires proposals from prep.
#
#   sbatch benchmarks/mlp_case_study/slurm_case_study_bench.sh
#
# Re-bench pending rows only:
#   CASE_PHASE=bench sbatch benchmarks/mlp_case_study/slurm_case_study_bench.sh

export CASE_STEPS=bench
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/mlp_case_study/study_case_job.sh"
