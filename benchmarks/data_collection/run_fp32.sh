#!/bin/bash
#SBATCH --job-name=autotuner-fp32
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# FP32×FP32→FP32 (accum FP32), TN layout only, 593 shapes.
# Plan: benchmarks/data_collection/plan_fp32_fp8.sh  (writes ~/autotuner/autotuner_sweep_fp32_tn.db)
# Submit: sbatch benchmarks/data_collection/run_fp32.sh
# Or: DB=$HOME/autotuner/autotuner_sweep_fp32_tn.db sbatch benchmarks/data_collection/run_fp32.sh

: "${CKS_TAG:=sweep_fp32_tn}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi
export CKS_TAG

slurm_cd_repo
slurm_require_cutlass
slurm_init_db_paths
slurm_db_restore
slurm_db_sync_loop
slurm_print_paths

slurm_run_scheduler \
    --tag          "$CKS_TAG" \
    --phase        compile,eval \
    --build-dir    "$CKS_BUILD_DIR" \
    --db-path      "$CKS_LOCAL_DB" \
    --num-gpus     "$CKS_GPUS" \
    --compile-jobs "$CKS_COMPILE_JOBS"
