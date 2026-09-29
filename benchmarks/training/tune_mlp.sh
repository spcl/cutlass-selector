#!/bin/bash
#SBATCH --job-name=tune-mlp
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Optuna hyperparameter search for the MLP (4 GPU workers, shared study).
# Prerequisite: artifacts/analysis/paper/features.parquet
# Submit: sbatch benchmarks/training/tune_mlp.sh

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

slurm_cd_repo
PYTHON="$(slurm_python)"
export PYTHONPATH="$SRC/model:${PYTHONPATH:-}"

FEATURES="$PAPER_DIR/features.parquet"
ARTIFACTS="$PAPER_DIR"
SHM="/dev/shm/tune_mlp_${SLURM_JOB_ID}"
OPTUNA_DB="$SHM/optuna.db"
CHECKPOINT="$ARTIFACTS/optuna_mlp.db"
STORAGE="sqlite:///$OPTUNA_DB"
STUDY="tune_mlp"
TRIALS_PER_GPU=75

[ -f "$FEATURES" ] || { echo "Missing $FEATURES — run src/model/features.py first." >&2; exit 1; }

mkdir -p "$SHM" "$ARTIFACTS"
cp "$FEATURES" "$SHM/features.parquet"
cp "${FEATURES%.parquet}.manifest.json" "$SHM/features.manifest.json"

if [ -f "$CHECKPOINT" ]; then
    echo "Resuming Optuna study from $CHECKPOINT"
    sqlite3 "$CHECKPOINT" ".backup $OPTUNA_DB"
fi

"$PYTHON" - <<PY
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.create_study(study_name="$STUDY", storage="$STORAGE", direction="minimize", load_if_exists=True)
PY

cleanup() {
    cp "$SHM"/worker_gpu*.log "$ARTIFACTS/" 2>/dev/null || true
    [ -f "$OPTUNA_DB" ] && sqlite3 "$OPTUNA_DB" ".backup $CHECKPOINT" || true
    "$PYTHON" - <<'PY' || true
import json, optuna, os
_HIDDEN = [(128,64),(256,128,64),(256,256,256),(512,256,128),(512,512,256),
           (1024,512,256),(256,128,64,32),(512,256,128,64),(1024,512,256,128),(1024,1024,512,256)]
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name=os.environ["STUDY"], storage=os.environ["STORAGE"])
best = study.best_trial
raw = dict(best.params)
loss = raw.pop("loss", None)
mlp = {"loss": loss, "lr": raw.pop("lr"), "hidden": list(_HIDDEN[raw.pop("hidden_idx")]),
       "dropout": raw.pop("dropout"), "weight_decay": raw.pop("weight_decay"), "epochs": raw.pop("epochs")}
if loss == "mse":
    mlp["batch_size"] = raw.pop("batch_size", None)
else:
    mlp.update({k: raw.pop(k, None) for k in ("groups_per_batch","grad_clip","max_pairs_per_group")})
    mlp["gain"] = raw.pop("gain", "linear")
out = os.path.join(os.environ["ARTIFACTS"], "tune_results_mlp.json")
json.dump({"best_cv_regret_mean": best.value, "best_trial": best.number, "mlp_params": mlp,
           "all_trials": [{"number": t.number, "value": t.value, "params": dict(t.params)}
                          for t in study.trials if t.value is not None]}, open(out, "w"), indent=2)
print(f"Wrote {out}  (trial #{best.number}, regret={best.value:.4f})")
PY
    rm -rf "$SHM"
}
export STUDY STORAGE ARTIFACTS
trap cleanup EXIT
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null; sleep 10; kill -KILL "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; cleanup; trap - EXIT; exit' SIGTERM

PIDS=()
for GPU in $(seq 0 $((CKS_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u "$SRC/model/tune_mlp.py" \
        --features "$SHM/features.parquet" \
        --outdir "$SHM" \
        --trials "$TRIALS_PER_GPU" \
        --cv 3 --train-seeds 0 1 \
        --sampler-seed $((42 + GPU)) \
        --device cuda \
        --storage "$STORAGE" \
        --study-name "$STUDY" \
        --suffix "gpu${GPU}" \
        > "$SHM/worker_gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done

tail -f "$SHM"/worker_gpu*.log &
TAIL=$!
wait "${PIDS[@]}"
kill "$TAIL" 2>/dev/null || true
