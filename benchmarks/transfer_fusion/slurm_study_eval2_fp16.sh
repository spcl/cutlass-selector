#!/bin/bash
#SBATCH --job-name=fusion-eval2-fp16
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Eval2: zero-shot fusion kinds (silu, bias_silu, tanh, bias_tanh).
#
#   STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval2_fp16.sh
#   STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval2_fp16.sh

export STUDY_DTYPE=fp16
export STUDY_EVAL_STEPS="${STUDY_EVAL_STEPS:-all}"
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_fusion/study_eval2_job.sh"
