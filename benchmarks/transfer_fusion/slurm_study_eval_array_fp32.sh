#!/bin/bash
#SBATCH --job-name=fusion-eval-fp32-a
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --array=0-23%6
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
#   STUDY_EVAL_STEPS=prep sbatch --time=01:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp32.sh
#   sbatch --time=02:00:00 --array=0-35%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp32.sh

export STUDY_DTYPE=fp32
export STUDY_EVAL_STEPS=eval
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_fusion/study_eval_job.sh"
