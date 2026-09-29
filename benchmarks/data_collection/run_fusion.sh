#!/bin/bash
#SBATCH --job-name=autotuner-fusion
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
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#
# Fusion fine-tune compile+eval (scheduler_fusion + hopper_template_fusion).
#
# Plan first:
#   benchmarks/data_collection/plan_fusion.sh fp16
# Submit (one job per dtype, can run in parallel):
#   CKS_TAG=sweep_fusion_fp16 DB=$HOME/autotuner/autotuner_sweep_fusion_fp16.db sbatch benchmarks/data_collection/run_fusion.sh
#   CKS_TAG=sweep_fusion_fp32 DB=$HOME/autotuner/autotuner_sweep_fusion_fp32.db sbatch benchmarks/data_collection/run_fusion.sh
#   CKS_TAG=sweep_fusion_fp8  DB=$HOME/autotuner/autotuner_sweep_fusion_fp8.db  sbatch benchmarks/data_collection/run_fusion.sh

: "${CKS_TAG:=sweep_fusion_fp16}"
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

# Use fusion scheduler (not the unfused src/autotuner/scheduler.py).
python="$(slurm_python)"
ulimit -c 0
"$python" -u "$SRC/autotuner/scheduler_fusion.py" \
    --tag          "$CKS_TAG" \
    --phase        compile,eval \
    --build-dir    "$CKS_BUILD_DIR" \
    --db-path      "$CKS_LOCAL_DB" \
    --num-gpus     "$CKS_GPUS" \
    --compile-jobs "$CKS_COMPILE_JOBS" \
    --template     hopper_template_fusion.cu.j2 &
pid=$!
trap slurm_db_checkpoint_final EXIT
trap '
    kill -TERM '"$pid"' 2>/dev/null || true
    sleep 5
    pkill -KILL -P '"$pid"' 2>/dev/null || true
    kill -KILL '"$pid"' 2>/dev/null || true
    wait '"$pid"' 2>/dev/null || true
    slurm_db_checkpoint_final
    trap - EXIT
    exit
' SIGTERM
wait "$pid"
