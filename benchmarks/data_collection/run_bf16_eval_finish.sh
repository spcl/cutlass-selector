#!/bin/bash
#SBATCH --job-name=bf16-eval-fin
#SBATCH --account=YOUR_ACCOUNT
# Do not use #SBATCH --environment here — srun needs it (see slurm_srun_prefix).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288 # Do not change this, will cause OOM
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Finish BF16 exhaustive eval on one node: eval-only, no recompile.
# Reads compiled .so from all 4-node shard trees on $SCRATCH and resumes from
# the durable home DB.
#
# Prereq: prior 4-node compile job left builds at
#   $SCRATCH/autotuner_build_bf16_eval_{0,1,2,3}
# and progress in
#   $HOME/autotuner/autotuner_bf16_eval.db
#
# Submit:
#   sbatch benchmarks/data_collection/run_bf16_eval_finish.sh
#
# Resume after preemption:
#   CKS_SKIP_MERGE=1 sbatch benchmarks/data_collection/run_bf16_eval_finish.sh
#
# Override:
#   DB=$HOME/autotuner/autotuner_bf16_eval.db sbatch benchmarks/data_collection/run_bf16_eval_finish.sh

: "${CKS_TAG:=bf16_eval}"
: "${CKS_NUM_SHARD_BUILDS:=4}"
: "${CKS_SKIP_MERGE:=0}"
: "${CKS_GPUS:=4}"
: "${CKS_COMPILE_JOBS:=0}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

export CKS_TAG
export CKS_NUM_SHARDS=1
export CKS_GPUS
export CKS_COMPILE_JOBS
export DB="${DB:-$HOME/autotuner/autotuner_${CKS_TAG}.db}"

slurm_cd_repo
slurm_require_cutlass
slurm_init_head_paths

scratch="${SCRATCH:-/iopsstor/scratch/cscs/${USER:-unknown}}"
MERGED_BUILD="${CKS_BUILD_DIR:-$scratch/autotuner_build_${CKS_TAG}_merged}"

if [ "$CKS_SKIP_MERGE" != "1" ]; then
    slurm_merge_shard_build_dirs "$CKS_TAG" "$CKS_NUM_SHARD_BUILDS" "$MERGED_BUILD"
else
    export CKS_BUILD_DIR="$MERGED_BUILD"
    echo "=== build merge skipped (CKS_SKIP_MERGE=1); using $MERGED_BUILD ==="
fi

slurm_print_paths
echo "phase     : eval-only (compiled kernels from ${CKS_NUM_SHARD_BUILDS} shard trees)"
echo

mkdir -p "$(dirname "$DB")"

# shellcheck disable=SC2046
if ! srun $(slurm_srun_prefix) -n1 -N1 --gpus-per-task="$CKS_GPUS" --cpus-per-task="${SLURM_CPUS_PER_TASK:-288}" \
    bash -c '
        source "'"$CKS_ROOT"'/benchmarks/common/slurm_common.sh"
        export CKS_TAG="'"$CKS_TAG"'"
        export CKS_NUM_SHARDS=1
        export CKS_GPUS="'"$CKS_GPUS"'"
        export CKS_COMPILE_JOBS="'"$CKS_COMPILE_JOBS"'"
        export CKS_NO_DB_TRAP=1
        export CKS_CKPT_INTERVAL=600
        export DB="'"$DB"'"
        slurm_cd_repo
        slurm_srun_worker_preamble
        export CKS_BUILD_DIR="'"$MERGED_BUILD"'"
        slurm_db_restore
        slurm_db_sync_loop
        slurm_run_scheduler \
            --tag "'"$CKS_TAG"'" \
            --phase eval \
            --build-dir "$CKS_BUILD_DIR" \
            --db-path "$CKS_LOCAL_DB" \
            --num-gpus "'"$CKS_GPUS"'" \
            --compile-jobs "'"$CKS_COMPILE_JOBS"'"
        slurm_db_checkpoint_final
    '; then
    echo "ERROR: eval finish failed (see .err). Home DB: $DB" >&2
    exit 1
fi

echo "Done. DB: $DB  tag: $CKS_TAG  build catalog: $MERGED_BUILD"
