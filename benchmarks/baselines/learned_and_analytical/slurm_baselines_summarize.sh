#!/bin/bash
# Legacy: summarize only (needs jobs 2+3 analytical). Prefer slurm_baselines_3_finish.sh.
#SBATCH --job-name=bl-sum
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

export BASELINE_STEPS=summarize
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CKS_ROOT/benchmarks/baselines/learned_and_analytical/study_baseline_job.sh"
