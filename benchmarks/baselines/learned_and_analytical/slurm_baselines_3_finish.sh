#!/bin/bash
#SBATCH --job-name=bl-fin
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
#
# Job 3/3 — analytical scores + comparison table/figures.
# Submit after jobs 1 and 2 complete (replace JOB1 JOB2 with their Slurm ids):
#
#   (from the repository root)
#   sbatch --dependency=afterok:JOB1:JOB2 benchmarks/baselines/learned_and_analytical/slurm_baselines_3_finish.sh

export BASELINE_STEPS=finish
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/baselines/learned_and_analytical/study_baseline_job.sh"
