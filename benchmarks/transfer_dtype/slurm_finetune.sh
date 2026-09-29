#!/bin/bash
#SBATCH --job-name=transfer-ft
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Fine-tune BF16 mlp_mse + xgb_mse on FP32 + FP8 TN sweeps (features + train, no in-job eval).
# Test later: broad GEMM eval vs nvMMH (eval dtype wiring TBD).
#
# Prerequisites:
#   ~/autotuner/autotuner_sweep_fp32_tn.db  (tag sweep_fp32_tn)
#   ~/autotuner/autotuner_sweep_fp8_tn.db   (tag sweep_fp8_tn)
#   artifacts/analysis/paper/mlp_mse/model_mlp.pt
#   artifacts/analysis/paper/xgb_mse/model_A.ubj
#
# Submit:
#   sbatch benchmarks/transfer_dtype/slurm_finetune.sh
#
# One dtype only:
#   sbatch benchmarks/transfer_dtype/slurm_finetune_fp32.sh
#   sbatch benchmarks/transfer_dtype/slurm_finetune_fp8.sh
#
# Overrides:
#   FINETUNE_EPOCHS=72 FINETUNE_LR=4e-5 sbatch benchmarks/transfer_dtype/slurm_finetune_fp32.sh

export TRANSFER_DTYPE="${TRANSFER_DTYPE:-all}"
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/transfer_dtype/finetune_job.sh"
