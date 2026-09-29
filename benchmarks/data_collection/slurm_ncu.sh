#!/bin/bash
#SBATCH --job-name=autotuner-ncu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# NCU profiling pass (run after compile+eval finished).
# Submit: sbatch benchmarks/data_collection/slurm_ncu.sh

: "${CKS_TAG:=sweep}"
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
    --phase        ncu \
    --build-dir    "$CKS_BUILD_DIR" \
    --db-path      "$CKS_LOCAL_DB" \
    --num-gpus     "$CKS_GPUS" \
    --ncu-top-k    "$CKS_NCU_TOP_K"
