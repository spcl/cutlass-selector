#!/bin/bash
#SBATCH --job-name=bl-ridge
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Job 2/3 — ridge linear (full + structural). Parallel with job 1.
#
#   (from the repository root)
#   sbatch benchmarks/baselines/learned_and_analytical/slurm_baselines_2_train.sh

export BASELINE_STEPS=train
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/baselines/learned_and_analytical/study_baseline_job.sh"
