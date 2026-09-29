#!/bin/bash
# Legacy: analytical only (needs job 1). Prefer slurm_baselines_3_finish.sh.
#SBATCH --job-name=bl-analytic
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

export BASELINE_STEPS=analytical
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CKS_ROOT/benchmarks/baselines/learned_and_analytical/study_baseline_job.sh"
