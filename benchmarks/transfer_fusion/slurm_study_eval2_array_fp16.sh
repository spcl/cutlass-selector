#!/bin/bash
#SBATCH --job-name=fusion-eval2-fp16-a
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --array=0-23%6
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

export STUDY_DTYPE=fp16
export STUDY_EVAL_STEPS=eval
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_fusion/study_eval2_job.sh"
