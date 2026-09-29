#!/bin/bash
#SBATCH --job-name=bl-anly
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Job 1/3 — training-only feature correlations. Parallel with job 2.
#
#   (from the repository root)
#   sbatch benchmarks/baselines/learned_and_analytical/slurm_baselines_1_analysis.sh

export BASELINE_STEPS=analysis
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/baselines/learned_and_analytical/study_baseline_job.sh"
