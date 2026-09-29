#!/bin/bash
#SBATCH --job-name=cutlass_baseline
#SBATCH --output=cutlass_bench_%j.out
#SBATCH --error=cutlass_bench_%j.err
#SBATCH --nodes=1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=01:00:00

REPO="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO/benchmarks/baselines/cutlass_profiler" || exit 1

uenv run --view=default pytorch/v2.9.1:v2 -- bash -c "
    source $REPO/.venv/bin/activate
    bash build.sh
    bash run_benchmark.sh
"
