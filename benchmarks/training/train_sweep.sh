#!/bin/bash
#SBATCH --job-name=train-sweep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Trains the objective sweep: every (model family, objective)
# pair, optionally for both feature sets, then reduces the per-run outputs to
# training_history_{xgb,mlp}.csv + summary_training.md.
#
# Prerequisite: artifacts/analysis/paper/features.parquet (+ .manifest.json), built by
#   python src/model/features.py \
#       --db <train.db> --train-tags bf16_final \
#       --eval-db <eval.db> --eval-tag bf16_eval \
#       --eval-min-configs 8000 --out artifacts/analysis/paper/features.parquet
# Either build it on the cluster or scp it up; it is gitignored, so a fresh clone
# will not have it.
#
# Submit:
#   sbatch benchmarks/training/train_sweep.sh                       # 6 hardware-aware runs
#   CKS_FEATURE_SETS="full structural" sbatch benchmarks/training/train_sweep.sh   # + ablation
#   CKS_RUNS="mlp:lambdarank xgb:ndcg" sbatch benchmarks/training/train_sweep.sh   # subset
#
# Each run is single-device and writes its own directory, so a run that dies takes
# nothing else with it; re-submitting with CKS_RUNS redoes only what is missing.

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

slurm_cd_repo
PYTHON="$(slurm_python)"
export PYTHONPATH="$SRC/model:${PYTHONPATH:-}"

ARTIFACTS="${CKS_ARTIFACTS:-$PAPER_DIR}"
FEATURES="$ARTIFACTS/features.parquet"
SHM="/dev/shm/train_sweep_${SLURM_JOB_ID:-local}"

# family:objective pairs. XGBoost runs on CPU, the MLP on one GPU each.
CKS_RUNS="${CKS_RUNS:-xgb:ndcg xgb:pairwise xgb:mse mlp:lambdarank mlp:ranknet mlp:mse}"
CKS_FEATURE_SETS="${CKS_FEATURE_SETS:-full}"
CKS_HISTORY_EVERY="${CKS_HISTORY_EVERY:-1}"
CKS_VAL_FRAC="${CKS_VAL_FRAC:-0.15}"
CKS_VAL_SEED="${CKS_VAL_SEED:-42}"
CKS_TRAIN_SEED="${CKS_TRAIN_SEED:-42}"

[ -f "$FEATURES" ] || { echo "ERROR: missing $FEATURES — run src/model/features.py first (see header)." >&2; exit 1; }
[ -f "${FEATURES%.parquet}.manifest.json" ] || { echo "ERROR: missing feature manifest next to $FEATURES" >&2; exit 1; }

# The parquet is read repeatedly by every run; keep it off the shared filesystem.
mkdir -p "$SHM" "$ARTIFACTS"
cp "$FEATURES" "$SHM/features.parquet"
cp "${FEATURES%.parquet}.manifest.json" "$SHM/features.manifest.json"

# Build the full run list (family:objective:feature_set).
JOBS=()
for FS in $CKS_FEATURE_SETS; do
    for RUN in $CKS_RUNS; do
        JOBS+=("${RUN}:${FS}")
    done
done

NSLOTS="${CKS_GPUS}"
[ "${#JOBS[@]}" -lt "$NSLOTS" ] && NSLOTS="${#JOBS[@]}"
CORES="${SLURM_CPUS_PER_TASK:-$(nproc)}"
XGB_THREADS=$(( CORES / NSLOTS )); [ "$XGB_THREADS" -lt 1 ] && XGB_THREADS=1

echo "repo        : $CKS_ROOT"
echo "artifacts   : $ARTIFACTS"
echo "features    : $SHM/features.parquet"
echo "runs        : ${JOBS[*]}"
echo "slots       : $NSLOTS (xgb threads/slot: $XGB_THREADS)"
echo "python      : $PYTHON"

# One shell function per slot: takes every JOBS entry with its index, runs them in
# sequence. Round-robin rather than a queue, so a slow run cannot starve a slot.
run_slot() {
    local slot="$1" i job family objective fs outdir log rc
    for (( i = slot; i < ${#JOBS[@]}; i += NSLOTS )); do
        job="${JOBS[$i]}"
        IFS=':' read -r family objective fs <<< "$job"
        local suffix=""
        [ "$fs" != "full" ] && suffix="_${fs}"
        outdir="$ARTIFACTS/${family}_${objective}${suffix}"
        log="$SHM/${family}_${objective}${suffix}.log"
        echo "[slot $slot] START $family/$objective/$fs -> $outdir"
        rc=0
        if [ "$family" = "xgb" ]; then
            OMP_NUM_THREADS="$XGB_THREADS" CUDA_VISIBLE_DEVICES="" \
            "$PYTHON" -u "$SRC/model/train_xgb.py" \
                --features "$SHM/features.parquet" \
                --loss "$objective" \
                --feature-set "$fs" \
                --val-frac "$CKS_VAL_FRAC" \
                --val-seed "$CKS_VAL_SEED" \
                --history-every "$CKS_HISTORY_EVERY" \
                --seed "$CKS_TRAIN_SEED" \
                --outdir "$outdir" > "$log" 2>&1 || rc=$?
        else
            CUDA_VISIBLE_DEVICES="$slot" \
            "$PYTHON" -u "$SRC/model/train_mlp.py" \
                --features "$SHM/features.parquet" \
                --loss "$objective" \
                --feature-set "$fs" \
                --val-frac "$CKS_VAL_FRAC" \
                --val-seed "$CKS_VAL_SEED" \
                --history-every "$CKS_HISTORY_EVERY" \
                --seed "$CKS_TRAIN_SEED" \
                --outdir "$outdir" > "$log" 2>&1 || rc=$?
        fi
        cp "$log" "$outdir/" 2>/dev/null || true
        if [ "$rc" -ne 0 ]; then
            echo "[slot $slot] FAILED $family/$objective/$fs (rc=$rc) — see $outdir/$(basename "$log")"
            tail -20 "$log" || true
        else
            echo "[slot $slot] DONE  $family/$objective/$fs"
        fi
    done
}

collect() {
    echo
    echo "Collecting training histories ..."
    "$PYTHON" -u "$SRC/model/collect_history.py" --artifacts "$ARTIFACTS" || true
    cp "$SHM"/*.log "$ARTIFACTS/" 2>/dev/null || true
    rm -rf "$SHM"
}

PIDS=()
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null || true; sleep 10; kill -KILL "${PIDS[@]}" 2>/dev/null || true; wait "${PIDS[@]}" 2>/dev/null || true; collect; trap - EXIT; exit' SIGTERM

for (( slot = 0; slot < NSLOTS; slot++ )); do
    run_slot "$slot" &
    PIDS+=($!)
done

wait "${PIDS[@]}"
collect
echo "Done. Artifacts in $ARTIFACTS"
