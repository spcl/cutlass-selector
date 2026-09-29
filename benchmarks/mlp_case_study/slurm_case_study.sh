#!/bin/bash
#SBATCH --job-name=mlp-case-study
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# End-to-end MLP inference GEMM case study (trace → select → bench → analyze).
#
# Scale: ~52 unique MLP-internal GEMMs × (1 ours + 8 nvMMH) ≈ 500 proposal rows
# (heavy compile dedup across models). Sequential workflow (recommended):
#   sbatch benchmarks/mlp_case_study/slurm_case_study_prep.sh      # 30m CPU
#   sbatch benchmarks/mlp_case_study/slurm_case_study_bench.sh     # 3h GPU
#   sbatch benchmarks/mlp_case_study/slurm_case_study_summarize.sh  # 10m CPU
#
# Or all-in-one:
#   sbatch benchmarks/mlp_case_study/slurm_case_study.sh
#
# Resume bench only (keep DB + build cache):
#   CASE_STEPS=bench CASE_PHASE=compile,bench sbatch benchmarks/mlp_case_study/slurm_case_study_bench.sh
#
# Fresh bench DB:
#   CASE_FRESH_DB=1 CASE_STEPS=bench sbatch benchmarks/mlp_case_study/slurm_case_study_bench.sh

export CASE_STEPS="${CASE_STEPS:-all}"
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/mlp_case_study/study_case_job.sh"
