#!/bin/bash
#SBATCH --job-name=bf16-final
#SBATCH --account=YOUR_ACCOUNT
# Do not use #SBATCH --environment here — nested srun needs it (see slurm_srun_prefix).
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288 # Do not change this, will cause OOM
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# BF16 final collection: plan + parallel compile/eval on 4 nodes (one 16 h job).
#
# Each node: live DB on /dev/shm, compiled .so on $SCRATCH (same layout as run.sh).
#   shard i owns eval pairs where hash(name:M:N:K) % 4 == i, and compiles only
#   configs needed for those pairs. Each node backs up to ~/autotuner/shards/…;
#   the head node merges shard DBs into ~/autotuner/autotuner_bf16_final.db at end.
#
# Submit:  sbatch benchmarks/data_collection/run_bf16_final.sh
# Resume:  CKS_SKIP_PLAN=1 sbatch benchmarks/data_collection/run_bf16_final.sh

: "${CKS_TAG:=bf16_final}"
: "${CKS_NUM_NODES:=4}"
: "${CKS_WAVE_EFF:=new_we}"
: "${CKS_SKIP_PLAN:=0}"
: "${CKS_COMPILE_JOBS:=64}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

export CKS_TAG
export CKS_NUM_SHARDS="$CKS_NUM_NODES"
export CKS_GPUS=4
export CKS_COMPILE_JOBS
export DB="${DB:-$HOME/autotuner/autotuner_${CKS_TAG}.db}"

slurm_cd_repo
slurm_require_cutlass
unset CKS_BUILD_DIR CKS_LOCAL_DB
slurm_init_head_paths
slurm_print_paths

PYTHON="$(slurm_python)"
SHAPES="${SHAPES:-$PLANS_DIR/shapes.json}"

mkdir -p "$(dirname "$DB")"

if [ "$CKS_SKIP_PLAN" != "1" ]; then
    echo "=== plan write tag=$CKS_TAG wave_eff=$CKS_WAVE_EFF DB=$DB ==="
    slurm_srun_vsc "$SRC/planning/plan.py" write \
        --tag "$CKS_TAG" \
        --shapes "$SHAPES" \
        --wave-eff "$CKS_WAVE_EFF" \
        --db "$DB"
else
    echo "=== plan skipped (CKS_SKIP_PLAN=1) ==="
    if compgen -G "${CKS_SHARD_DIR}/shard_*.db" > /dev/null; then
        echo "=== merging partial shard backups into $DB ==="
        slurm_merge_shards
    fi
fi

echo "=== compile+eval ($CKS_NUM_NODES nodes × $CKS_GPUS GPUs, node-local) ==="

# shellcheck disable=SC2046
if ! srun $(slurm_srun_prefix) \
    --nodes="$CKS_NUM_NODES" --ntasks="$CKS_NUM_NODES" --gpus-per-task="$CKS_GPUS" \
    bash -c '
        source "'"$CKS_ROOT"'/benchmarks/common/slurm_common.sh"
        export CKS_TAG="'"$CKS_TAG"'"
        export CKS_NUM_SHARDS="'"$CKS_NUM_NODES"'"
        export CKS_GPUS="'"$CKS_GPUS"'"
        export CKS_COMPILE_JOBS="'"$CKS_COMPILE_JOBS"'"
        export CKS_NO_DB_TRAP=1
        export CKS_CKPT_INTERVAL=600
        export DB="'"$DB"'"
        slurm_cd_repo
        slurm_srun_worker_preamble
        slurm_db_restore
        slurm_db_sync_loop
        slurm_run_scheduler \
            --tag "'"$CKS_TAG"'" \
            --phase compile,eval \
            --build-dir "$CKS_BUILD_DIR" \
            --db-path "$CKS_LOCAL_DB" \
            --num-gpus "'"$CKS_GPUS"'" \
            --compile-jobs "'"$CKS_COMPILE_JOBS"'" \
            --shard-index "${SLURM_PROCID}" \
            --num-shards "'"$CKS_NUM_NODES"'"
        slurm_shard_backup
    '; then
    echo "ERROR: srun failed (see .err). Partial shards in ${CKS_SHARD_DIR}." >&2
    slurm_merge_shards || true
    exit 1
fi

echo "=== merging $CKS_NUM_NODES shard DBs → $DB ==="
slurm_merge_shards
echo "Done. DB: $DB  tag: $CKS_TAG  shards: $CKS_SHARD_DIR"
