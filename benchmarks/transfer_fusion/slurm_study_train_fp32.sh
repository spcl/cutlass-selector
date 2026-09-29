#!/bin/bash
#SBATCH --job-name=fusion-study-fp32
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=18:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# FP32 fusion transfer study — prep + 36 training runs.
# DB: ~/autotuner/autotuner_sweep_fusion_fp32.db
#
#   STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_train_fp32.sh
#   sbatch --time=18:00:00 benchmarks/transfer_fusion/slurm_study_train_fp32.sh

export STUDY_DTYPE=fp32
export STUDY_STEPS="${STUDY_STEPS:-all}"
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_fusion/study_train_job.sh"
