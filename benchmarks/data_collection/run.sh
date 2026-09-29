#!/bin/bash
#SBATCH --job-name=autotuner-bf16
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G

#
# Compile + benchmark the planned sweep (resume-safe).
# Plan first: benchmarks/data_collection/plan_sweep.sh
# Submit:  sbatch benchmarks/data_collection/run.sh
# Parallel: use matching DB= on plan and sbatch (default: ~/autotuner/autotuner_${CKS_TAG}.db)
#   DB=$HOME/autotuner/autotuner_sweep.db sbatch benchmarks/data_collection/run.sh
# Override tag: CKS_TAG=myrun sbatch benchmarks/data_collection/run.sh

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
    --phase        compile,eval \
    --build-dir    "$CKS_BUILD_DIR" \
    --db-path      "$CKS_LOCAL_DB" \
    --num-gpus     "$CKS_GPUS" \
    --compile-jobs "$CKS_COMPILE_JOBS"
