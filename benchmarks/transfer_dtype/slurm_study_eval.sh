#!/bin/bash
#SBATCH --job-name=transfer-study-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Both dtypes sequential. For parallel eval:
#   STUDY_EVAL_STEPS=prep sbatch --time=01:00:00 benchmarks/transfer_dtype/slurm_study_eval_fp32.sh
#   sbatch --time=06:00:00 benchmarks/transfer_dtype/slurm_study_eval_array_fp32.sh

export STUDY_DTYPE="${STUDY_DTYPE:-all}"
export STUDY_EVAL_STEPS="${STUDY_EVAL_STEPS:-all}"
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_dtype/study_eval_job.sh"
