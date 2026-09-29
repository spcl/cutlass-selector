#!/bin/bash
#SBATCH --job-name=transfer-study-array
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --array=0-71
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Both dtypes in one array (0-71). Prefer dtype-specific arrays:
#   slurm_study_train_array_fp32.sh  (0-35)
#   slurm_study_train_array_fp8.sh   (0-35)

export STUDY_DTYPE="${STUDY_DTYPE:-all}"
export STUDY_STEPS=train
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_dtype/study_train_job.sh"
