#!/bin/bash
#SBATCH --job-name=mlp-case-sum
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# CPU-only: weighted GEMM time + time-to-solution tables → SUMMARY.md
#
#   sbatch benchmarks/mlp_case_study/slurm_case_study_summarize.sh

export CASE_STEPS=summarize
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/mlp_case_study/study_case_job.sh"
