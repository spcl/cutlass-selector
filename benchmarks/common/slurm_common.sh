#!/bin/bash
# Shared helpers for CSCS Alps Slurm jobs (GH200).

_SLURM_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults (override via env) ───────────────────────────────────────────────
# CKS_TAG is set by each job script before sourcing this file.
CKS_UENV="${CKS_UENV:-pytorch/v2.9.1:v2}"
CKS_GPUS="${CKS_GPUS:-4}"
CKS_COMPILE_JOBS="${CKS_COMPILE_JOBS:-64}"
CKS_NCU_TOP_K="${CKS_NCU_TOP_K:-20}"
CKS_WAVE_EFF="${CKS_WAVE_EFF:-new_we}"   # proxy plan: old_we | new_we
CKS_CKPT_INTERVAL="${CKS_CKPT_INTERVAL:-600}"  # 0 = disable background checkpoint
CKS_DB_SHARED="${CKS_DB_SHARED:-0}"           # 1 = legacy shared scratch DB (slow; avoid)
CKS_NUM_SHARDS="${CKS_NUM_SHARDS:-1}"
CKS_SRUN_ENV="${CKS_SRUN_ENV:-vsc}"           # CSCS: pass --environment to srun, not sbatch (see slurm_srun_prefix)
CKS_SRUN_NETWORK="${CKS_SRUN_NETWORK:-disable_rdzv_get}"  # 0 = omit --network=…

slurm_repo_root() {
    local hint
    if [ -n "${CKS_ROOT:-}" ]; then
        hint="$CKS_ROOT"
    elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        hint="$SLURM_SUBMIT_DIR"
    else
        hint="$_SLURM_COMMON_DIR"
    fi
    # shellcheck source=repo_paths.sh
    source "$_SLURM_COMMON_DIR/repo_paths.sh"
    repo_paths_find_root "$hint"
}

slurm_cd_repo() {
    # shellcheck source=repo_paths.sh
    source "$_SLURM_COMMON_DIR/repo_paths.sh"
    repo_paths_export "${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$_SLURM_COMMON_DIR}}"
    cd "$REPO_ROOT"
    export PYTHONUNBUFFERED=1
}

slurm_python() {
    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"
    elif [ -x "$CKS_ROOT/.venv/bin/python" ]; then
        echo "$CKS_ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        echo python3
    else
        echo python
    fi
}

slurm_init_head_paths() {
    local tag="${CKS_TAG:-default}"
    local job="${SLURM_JOB_ID:-local}"
    export CKS_HOME_DB="${DB:-$HOME/autotuner/autotuner_${tag}.db}"
    export CKS_SHARD_DIR="$HOME/autotuner/shards/${tag}/job_${job}"
    mkdir -p "$(dirname "$CKS_HOME_DB")" "$CKS_SHARD_DIR"
}

# Durable DB in $HOME; live DB on /dev/shm; build + ckpt on $SCRATCH (same as run.sh).
slurm_init_db_paths() {
    local scratch="${SCRATCH:-/iopsstor/scratch/cscs/${USER:-unknown}}"
    local tag="${CKS_TAG:-default}"
    local shard="${SLURM_PROCID:-${SLURM_NODEID:-0}}"
    local job="${SLURM_JOB_ID:-local}"

    export CKS_HOME_DB="${DB:-${CKS_HOME_DB:-$HOME/autotuner/autotuner_${tag}.db}}"
    export CKS_SHARD_DIR="${CKS_SHARD_DIR:-$HOME/autotuner/shards/${tag}/job_${job}}"
    export CKS_CKPT_DIR="${CKS_CKPT_DIR:-$scratch/autotuner_ckpt_${tag}}"

    if [ "${CKS_DB_SHARED:-0}" = "1" ]; then
        echo "WARNING: CKS_DB_SHARED=1 — shared scratch DB (slow). Use node-local shards instead." >&2
        export CKS_BUILD_DIR="${CKS_BUILD_DIR:-$scratch/autotuner_build_${tag}}"
        export CKS_LOCAL_DB="${CKS_SHARED_DB:-$scratch/autotuner_db_${tag}/autotuner.db}"
        mkdir -p "$(dirname "$CKS_LOCAL_DB")" "$CKS_CKPT_DIR" "$(dirname "$CKS_HOME_DB")" "$CKS_BUILD_DIR"
        return 0
    fi

    if [ "${CKS_NUM_SHARDS:-1}" -gt 1 ] 2>/dev/null; then
        # Always derive from SLURM_PROCID — do not use ${VAR:-} here; the sbatch
        # head node calls slurm_init_db_paths before srun with PROCID unset (→ 0)
        # and srun inherits that export, which would pin every task to shard _0.
        export CKS_LOCAL_DB="/dev/shm/autotuner_${tag}_${job}_${shard}.db"
        export CKS_BUILD_DIR="$scratch/autotuner_build_${tag}_${shard}"
    else
        export CKS_LOCAL_DB="${CKS_LOCAL_DB:-/dev/shm/autotuner_${tag}_${job}.db}"
        export CKS_BUILD_DIR="${CKS_BUILD_DIR:-$scratch/autotuner_build_${tag}}"
    fi
    mkdir -p "$(dirname "$CKS_LOCAL_DB")" "$CKS_BUILD_DIR" "$CKS_CKPT_DIR" \
        "$(dirname "$CKS_HOME_DB")" "$CKS_SHARD_DIR"
}

# CSCS Alps: do NOT use #SBATCH --environment=… when the batch script also calls srun.
# Nested srun needs --environment on srun itself (otherwise switch_hpe_slingshot / libjson-c errors).
# See https://docs.cscs.ch/software/alps-extended-images/
slurm_srun_worker_preamble() {
    unset CKS_BUILD_DIR CKS_LOCAL_DB
    echo "=== srun task procid=${SLURM_PROCID:-?} node=${SLURMD_NODENAME:-?} ==="
    slurm_init_db_paths
    echo "build_dir=$CKS_BUILD_DIR"
    echo "local_db=$CKS_LOCAL_DB"
}

slurm_srun_prefix() {
    local -a args=(--environment="${CKS_SRUN_ENV:-vsc}")
    if [ "${CKS_SRUN_NETWORK:-disable_rdzv_get}" != "0" ]; then
        args+=(--network="${CKS_SRUN_NETWORK:-disable_rdzv_get}")
    fi
    echo "${args[@]}"
}

# Batch-head helpers (plan.py, merge_shards.py) must not use login-node python3 —
# on CSCS it is 3.6 and lacks `from __future__ import annotations`. Workers already
# get vsc via slurm_srun_prefix; this runs one CPU task on the allocation.
slurm_srun_vsc() {
    # shellcheck disable=SC2046
    srun $(slurm_srun_prefix) -n1 -N1 --gpus-per-task=0 --cpus-per-task=4 \
        bash -c '
            source "'"$BENCHMARKS"'/common/slurm_common.sh"
            slurm_cd_repo
            exec "$(slurm_python)" "$@"
        ' _ "$@"
}

slurm_db_restore() {
    if [ "${CKS_DB_SHARED:-0}" = "1" ]; then
        if [ ! -f "$CKS_LOCAL_DB" ] && [ -f "$CKS_HOME_DB" ]; then
            echo "Initializing shared DB from $CKS_HOME_DB"
            cp "$CKS_HOME_DB" "$CKS_LOCAL_DB"
        fi
        return 0
    fi
    if [ -f "$CKS_HOME_DB" ]; then
        echo "Restoring node-local DB from $CKS_HOME_DB → $CKS_LOCAL_DB"
        cp "$CKS_HOME_DB" "$CKS_LOCAL_DB"
    fi
}

# Online backup API — tolerates concurrent writers better than wal_checkpoint + cp.
slurm_db_checkpoint_copy() {
    local src="$1" dst="$2"
    local py
    py="$(slurm_python 2>/dev/null || echo python)"
    "$py" - "$src" "$dst" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
with sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=120) as src_conn:
    src_conn.execute("PRAGMA busy_timeout=120000")
    with sqlite3.connect(dst, timeout=120) as dst_conn:
        dst_conn.execute("PRAGMA busy_timeout=120000")
        src_conn.backup(dst_conn)
PY
}

slurm_shard_id() {
    echo "${SLURM_PROCID:-${SLURM_NODEID:-0}}"
}

slurm_shard_backup() {
    [ -f "$CKS_LOCAL_DB" ] || return 0
    local shard size dest tmp
    shard="$(slurm_shard_id)"
    size=$(stat -c%s "$CKS_LOCAL_DB" 2>/dev/null || stat -f%z "$CKS_LOCAL_DB" 2>/dev/null || echo 0)
    [ "$size" -gt 0 ] || return 0
    mkdir -p "$CKS_SHARD_DIR"
    dest="$CKS_SHARD_DIR/shard_${shard}.db"
    tmp="$dest.tmp.$$"
    rm -f "$tmp"
    echo "Shard ${shard} backup (${size} bytes) → $dest"
    if slurm_db_checkpoint_copy "$CKS_LOCAL_DB" "$tmp"; then
        mv "$tmp" "$dest"
        rm -f "$dest-wal" "$dest-shm"
    else
        rm -f "$tmp"
        echo "Shard backup failed." >&2
        return 1
    fi
}

# Union per-node scratch build trees (batch_*.so) into one catalog for eval resume.
# Default layout: $SCRATCH/autotuner_build_${tag}_0 .. _{num_shards-1}.
slurm_merge_shard_build_dirs() {
    local tag="${1:-${CKS_TAG:-bf16_eval}}"
    local num_shards="${2:-4}"
    local scratch="${SCRATCH:-/iopsstor/scratch/cscs/${USER:-unknown}}"
    local dest="${3:-$scratch/autotuner_build_${tag}_merged}"
    local i src f base n_src n_new=0 n_total=0

    mkdir -p "$dest"
    echo "=== merging shard build dirs → $dest ==="
    for i in $(seq 0 "$((num_shards - 1))"); do
        src="$scratch/autotuner_build_${tag}_${i}"
        if [ ! -d "$src" ]; then
            echo "WARNING: missing $src" >&2
            continue
        fi
        n_src=0
        shopt -s nullglob
        for f in "$src"/batch_*.so; do
            n_src=$((n_src + 1))
            base="$(basename "$f")"
            if [ -e "$dest/$base" ]; then
                continue
            fi
            if ln "$f" "$dest/$base" 2>/dev/null; then
                n_new=$((n_new + 1))
            else
                cp -n "$f" "$dest/$base"
                n_new=$((n_new + 1))
            fi
        done
        shopt -u nullglob
        echo "  shard $i: $n_src .so in $src"
    done
    n_total=$(find "$dest" -maxdepth 1 -name 'batch_*.so' 2>/dev/null | wc -l)
    echo "merged catalog: $n_total batch_*.so ($n_new newly linked)"
    export CKS_BUILD_DIR="$dest"
}

slurm_merge_shards() {
    local tag job shard_dir dest
    tag="${CKS_TAG:-default}"
    job="${SLURM_JOB_ID:-local}"
    shard_dir="${CKS_SHARD_DIR:-$HOME/autotuner/shards/${tag}/job_${job}}"
    dest="${CKS_HOME_DB:-$HOME/autotuner/autotuner_${tag}.db}"

    if ! compgen -G "$shard_dir/shard_*.db" > /dev/null; then
        echo "No shard backups in $shard_dir — skip merge."
        return 0
    fi

    echo "Merging shard DBs from $shard_dir → $dest"
    slurm_srun_vsc "$SRC/autotuner/merge_shards.py" \
        --work-dir "/dev/shm/merge_${tag}_${job}" \
        --dest "$dest" \
        --shards "$shard_dir"/shard_*.db
}

slurm_db_checkpoint() {
    if [ "${CKS_NUM_SHARDS:-1}" -gt 1 ] 2>/dev/null; then
        slurm_shard_backup
        return 0
    fi
    [ -f "$CKS_LOCAL_DB" ] || return 0
    local size tmp
    size=$(stat -c%s "$CKS_LOCAL_DB" 2>/dev/null || stat -f%z "$CKS_LOCAL_DB" 2>/dev/null || echo 0)
    [ "$size" -gt 0 ] || return 0
    echo "Checkpointing DB (${size} bytes) ..."
    tmp="$CKS_CKPT_DIR/autotuner.db.tmp.$$"
    rm -f "$tmp"
    if slurm_db_checkpoint_copy "$CKS_LOCAL_DB" "$tmp"; then
        mv "$tmp" "$CKS_CKPT_DIR/autotuner.db"
        rm -f "$CKS_CKPT_DIR/autotuner.db-wal" "$CKS_CKPT_DIR/autotuner.db-shm"
    else
        rm -f "$tmp"
        echo "Checkpoint failed." >&2
    fi
}

slurm_db_checkpoint_final() {
    kill "${CKS_SYNC_PID:-}" 2>/dev/null || true
    if [ "${CKS_NUM_SHARDS:-1}" -gt 1 ] 2>/dev/null && [ "${CKS_DB_SHARED:-0}" != "1" ]; then
        slurm_shard_backup
        return 0
    fi
    [ -f "$CKS_LOCAL_DB" ] || return 0
    if [ "${CKS_DB_SHARED:-0}" = "1" ]; then
        local size htmp
        size=$(stat -c%s "$CKS_LOCAL_DB" 2>/dev/null || stat -f%z "$CKS_LOCAL_DB" 2>/dev/null || echo 0)
        [ "$size" -gt 0 ] || return 0
        echo "Shared DB checkpoint (${size} bytes) → $CKS_HOME_DB"
        htmp="$(dirname "$CKS_HOME_DB")/autotuner.db.tmp.$$"
        rm -f "$htmp"
        slurm_db_checkpoint_copy "$CKS_LOCAL_DB" "$htmp" && mv "$htmp" "$CKS_HOME_DB"
        return 0
    fi
    local size tmp htmp
    size=$(stat -c%s "$CKS_LOCAL_DB" 2>/dev/null || stat -f%z "$CKS_LOCAL_DB" 2>/dev/null || echo 0)
    [ "$size" -gt 0 ] || return 0
    echo "Final DB checkpoint (${size} bytes) ..."
    tmp="$CKS_CKPT_DIR/autotuner.db.tmp.$$"
    rm -f "$tmp"
    slurm_db_checkpoint_copy "$CKS_LOCAL_DB" "$tmp" && mv "$tmp" "$CKS_CKPT_DIR/autotuner.db"
    rm -f "$CKS_CKPT_DIR/autotuner.db-wal" "$CKS_CKPT_DIR/autotuner.db-shm"
    htmp="$(dirname "$CKS_HOME_DB")/autotuner.db.tmp.$$"
    rm -f "$htmp"
    cp "$CKS_CKPT_DIR/autotuner.db" "$htmp" && mv "$htmp" "$CKS_HOME_DB"
    echo "Durable copy: $CKS_HOME_DB"
}

slurm_db_checkpoint_home() {
    slurm_db_checkpoint_final
}

slurm_db_sync_loop() {
    if [ "${CKS_CKPT_INTERVAL:-600}" -le 0 ] 2>/dev/null; then
        echo "Background DB checkpoint disabled (CKS_CKPT_INTERVAL=0)"
        return 0
    fi
    ( while true; do
        sleep "$CKS_CKPT_INTERVAL"
        slurm_db_checkpoint
    done ) &
    CKS_SYNC_PID=$!
    export CKS_SYNC_PID
}

# Run a Python driver with SIGTERM grace period (--signal=B:TERM@…).
slurm_run_scheduler() {
    local python pid
    python="$(slurm_python)"
    ulimit -c 0
    "$python" -u "$SRC/autotuner/scheduler.py" "$@" &
    pid=$!
    if [ "${CKS_NO_DB_TRAP:-0}" != "1" ]; then
        trap slurm_db_checkpoint_final EXIT
    fi
    trap '
        kill -TERM '"$pid"' 2>/dev/null || true
        sleep 5
        pkill -KILL -P '"$pid"' 2>/dev/null || true
        kill -KILL '"$pid"' 2>/dev/null || true
        wait '"$pid"' 2>/dev/null || true
        if [ "${CKS_NO_DB_TRAP:-0}" != "1" ]; then
            slurm_db_checkpoint_final
        elif [ "${CKS_NUM_SHARDS:-1}" -gt 1 ] 2>/dev/null; then
            slurm_shard_backup
        fi
        trap - EXIT
        exit
    ' SIGTERM
    wait "$pid"
}

slurm_require_cutlass() {
    if [ ! -d "$CUTLASS_DIR/include" ]; then
        echo "ERROR: cutlass missing at $CUTLASS_DIR. Run: git submodule update --init src/extern/cutlass" >&2
        exit 1
    fi
}

slurm_print_paths() {
    echo "repo      : $CKS_ROOT"
    echo "tag       : $CKS_TAG"
    echo "wave eff  : $CKS_WAVE_EFF  (plan write --wave-eff; does not affect eval resume)"
    echo "shards    : ${CKS_NUM_SHARDS:-1}  db_shared=${CKS_DB_SHARED:-0}"
    if [ -n "${CKS_BUILD_DIR:-}" ]; then
        echo "build dir : $CKS_BUILD_DIR"
    fi
    if [ -n "${CKS_LOCAL_DB:-}" ]; then
        echo "live DB   : $CKS_LOCAL_DB"
    fi
    echo "home DB   : ${CKS_HOME_DB:-${DB:-unset}}"
    echo "shard dir : ${CKS_SHARD_DIR:-unset}"
    echo "python    : $(slurm_python) ($($(slurm_python) --version 2>&1))"
}
