#!/bin/bash
#SBATCH --job-name=capacity-sweep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Feature-vs-capacity study: does the hardware-aware advantage shrink as the model
# gets big enough to synthesize the derived features itself?
#
#   H1 (MLP)     the full-vs-structural gap decays to zero with capacity
#   H2 (XGBoost) the gap persists at model sizes where the MLP's has closed,
#                because axis-aligned splits cannot build ceil(M/tm)-style terms
#   refuted if   XGBoost's gap closes at comparable model size
#
# Every run trains MSE on the same data, split and validation seed; only capacity,
# feature set and training seed vary. Scored on the validation split only
# (--no-eval), so the incomplete eval sweep cannot affect it.
#
# Runs both under Slurm and on a single workstation:
#   sbatch benchmarks/training/capacity_sweep.sh            # cluster: GPUs for MLP, cores for XGB
#   bash   benchmarks/training/capacity_sweep.sh            # local: one MLP stream + one XGB stream
#
# Resumable: a run whose metrics.json exists is skipped, so re-invoking after a
# crash, an OOM kill or a walltime kill continues where it stopped.
#
# Options:
#   --widths "16x8 ..."     MLP hidden sizes         --depths "2 3 ..."   XGB max_depth
#   --seeds "42 43 44"      training seeds           --feature-sets "..." full/structural
#   --artifacts PATH        output root              --dry-run            list runs only

set -uo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi
set +u; slurm_cd_repo; set -u
PYTHON="$(slurm_python)"
export PYTHONPATH="$SRC/model:${PYTHONPATH:-}"

WIDTHS="${CKS_WIDTHS:-16x8 32x16 64x32 128x64 256x128x64 1024x1024x512x256}"
DEPTHS="${CKS_DEPTHS:-2 3 4 6 8 11 14}"
SEEDS="${CKS_SEEDS:-42 43 44}"
FEATURE_SETS="${CKS_FEATURE_SETS:-full structural}"
ARTIFACTS="${CKS_ARTIFACTS:-$CAPACITY_DIR}"
EPOCHS="${CKS_EPOCHS:-144}"
N_EST="${CKS_N_ESTIMATORS:-916}"
MLP_HIST="${CKS_MLP_HISTORY_EVERY:-4}"
XGB_HIST="${CKS_XGB_HISTORY_EVERY:-25}"
TM_ROWS="${CKS_TRAIN_METRIC_ROWS:-400000}"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --widths)        WIDTHS="$2";       shift 2 ;;
        --depths)        DEPTHS="$2";       shift 2 ;;
        --seeds)         SEEDS="$2";        shift 2 ;;
        --feature-sets)  FEATURE_SETS="$2"; shift 2 ;;
        --artifacts)     ARTIFACTS="$2";    shift 2 ;;
        --epochs)        EPOCHS="$2";       shift 2 ;;
        --n-estimators)  N_EST="$2";        shift 2 ;;
        --dry-run)       DRY_RUN=1;         shift   ;;
        -h|--help)       sed -n '14,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

FEATURES="${CKS_FEATURES:-$PAPER_DIR/features.parquet}"
[ -f "$FEATURES" ] || { echo "ERROR: missing $FEATURES" >&2; exit 1; }
[ -f "${FEATURES%.parquet}.manifest.json" ] || { echo "ERROR: missing manifest for $FEATURES" >&2; exit 1; }
mkdir -p "$ARTIFACTS"

# Build the two run lists. MLP needs a GPU, XGBoost needs cores; keeping them in
# separate streams lets the two overlap instead of serializing.
MLP_JOBS=(); XGB_JOBS=()
for SEED in $SEEDS; do
  for FS in $FEATURE_SETS; do
    for W in $WIDTHS; do MLP_JOBS+=("$W:$FS:$SEED"); done
    for D in $DEPTHS; do XGB_JOBS+=("$D:$FS:$SEED"); done
  done
done

if [ -n "${SLURM_JOB_ID:-}" ]; then
    MLP_STREAMS="${CKS_GPUS:-4}"; CORES="${SLURM_CPUS_PER_TASK:-$(nproc)}"; XGB_STREAMS=2
else
    # One workstation: a single GPU and ~8GB per process. One of each runs
    # concurrently; a second XGBoost stream would not fit beside them.
    MLP_STREAMS=1; CORES="$(nproc)"; XGB_STREAMS=1
fi
XGB_THREADS=$(( CORES / XGB_STREAMS )); [ "$XGB_THREADS" -lt 1 ] && XGB_THREADS=1

echo "artifacts : $ARTIFACTS"
echo "features  : $FEATURES"
echo "widths    : $WIDTHS"
echo "depths    : $DEPTHS"
echo "seeds     : $SEEDS   feature sets: $FEATURE_SETS"
echo "runs      : ${#MLP_JOBS[@]} MLP + ${#XGB_JOBS[@]} XGB = $(( ${#MLP_JOBS[@]} + ${#XGB_JOBS[@]} ))"
echo "streams   : $MLP_STREAMS MLP / $XGB_STREAMS XGB (${XGB_THREADS} threads each)"

if [ "$DRY_RUN" = "1" ]; then
    for j in "${MLP_JOBS[@]}"; do IFS=':' read -r w fs sd <<< "$j"; echo "  mlp  w=$w fs=$fs seed=$sd"; done
    for j in "${XGB_JOBS[@]}"; do IFS=':' read -r d fs sd <<< "$j"; echo "  xgb  d=$d fs=$fs seed=$sd"; done
    exit 0
fi

run_one() {
    local kind="$1" cap="$2" fs="$3" seed="$4" slot="$5" outdir log rc=0
    outdir="$ARTIFACTS/${kind}_mse_${cap//x/x}_${fs}_s${seed}"
    [ "$kind" = "xgb" ] && outdir="$ARTIFACTS/xgb_mse_d${cap}_${fs}_s${seed}"
    if [ -f "$outdir/metrics.json" ]; then echo "  SKIP $(basename "$outdir")"; return 0; fi
    log="$outdir.log"; mkdir -p "$outdir"
    if [ "$kind" = "mlp" ]; then
        CUDA_VISIBLE_DEVICES="$slot" "$PYTHON" -u "$SRC/model/train_mlp.py" \
            --features "$FEATURES" --loss mse --feature-set "$fs" \
            --hidden ${cap//x/ } --epochs "$EPOCHS" --history-every "$MLP_HIST" \
            --val-frac 0.15 --val-seed 42 --seed "$seed" --no-eval \
            --outdir "$outdir" > "$log" 2>&1 || rc=$?
    else
        OMP_NUM_THREADS="$XGB_THREADS" CUDA_VISIBLE_DEVICES="" "$PYTHON" -u "$SRC/model/train_xgb.py" \
            --features "$FEATURES" --loss mse --feature-set "$fs" \
            --max-depth "$cap" --n-estimators "$N_EST" --history-every "$XGB_HIST" \
            --train-metric-rows "$TM_ROWS" \
            --val-frac 0.15 --val-seed 42 --seed "$seed" --no-eval \
            --outdir "$outdir" > "$log" 2>&1 || rc=$?
    fi
    cp "$log" "$outdir/" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then
        echo "  FAILED $(basename "$outdir") rc=$rc"; tail -15 "$log" || true
    else
        echo "  done $(basename "$outdir")  $(date +%H:%M:%S)"
    fi
}

stream() {
    local kind="$1" slot="$2" nstreams="$3"; shift 3
    local jobs=("$@") i
    for (( i = slot; i < ${#jobs[@]}; i += nstreams )); do
        IFS=':' read -r cap fs seed <<< "${jobs[$i]}"
        run_one "$kind" "$cap" "$fs" "$seed" "$slot"
    done
}

collect() { "$PYTHON" -u "$SRC/model/collect_history.py" --artifacts "$ARTIFACTS" || true; }

PIDS=()
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null || true; sleep 5; kill -KILL "${PIDS[@]}" 2>/dev/null || true; collect; exit' SIGTERM SIGINT

for (( s = 0; s < MLP_STREAMS; s++ )); do stream mlp "$s" "$MLP_STREAMS" "${MLP_JOBS[@]}" & PIDS+=($!); done
for (( s = 0; s < XGB_STREAMS; s++ )); do stream xgb "$s" "$XGB_STREAMS" "${XGB_JOBS[@]}" & PIDS+=($!); done
wait "${PIDS[@]}"
collect
echo "Done. Artifacts in $ARTIFACTS"
