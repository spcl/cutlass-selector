#!/bin/bash
#SBATCH --job-name=transfer-ft-fp32
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Fine-tune BF16 mlp_mse + xgb_mse on the FP32 TN sweep only.
#
#   sbatch benchmarks/transfer_dtype/slurm_finetune_fp32.sh

export TRANSFER_DTYPE=fp32
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_dtype/finetune_job.sh"
