#!/bin/bash
#SBATCH --job-name=fusion-study-fp8-a
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --array=0-23%6
#SBATCH --time=05:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# One FP8 training run per array task (run prep first):
#   STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_train_fp8.sh
#   sbatch --time=05:00:00 --array=0-35%6 benchmarks/transfer_fusion/slurm_study_train_array_fp8.sh

export STUDY_DTYPE=fp8_e4m3
export STUDY_STEPS=train
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_fusion/study_train_job.sh"
