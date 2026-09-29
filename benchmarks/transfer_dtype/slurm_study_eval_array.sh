#!/bin/bash
#SBATCH --job-name=transfer-study-eval-a
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --array=0-71
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Prefer dtype-specific arrays (0-35 each):
#   slurm_study_eval_array_fp32.sh
#   slurm_study_eval_array_fp8.sh

export STUDY_DTYPE="${STUDY_DTYPE:-all}"
export STUDY_EVAL_STEPS=eval
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_dtype/study_eval_job.sh"
