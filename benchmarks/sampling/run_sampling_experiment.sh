#!/bin/bash
#SBATCH --job-name=sampling-exp
#SBATCH --account=YOUR_ACCOUNT
# Do not use #SBATCH --environment here — srun needs it (see slurm_srun_prefix).
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Offline sampling-coverage experiment (benchmarks/sampling/sampling_experiment.py)
#
# Submit:
#   sbatch benchmarks/sampling/run_sampling_experiment.sh
#
# Override:
#   DB=~/autotuner_bf16_eval.db OUTDIR=artifacts/analysis/sampling sbatch ...

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

slurm_cd_repo

DB="${DB:-$HOME/autotuner/autotuner_bf16_eval.db}"
OUTDIR="${OUTDIR:-$ARTIFACTS/analysis/sampling}"
RESAMPLES="${RESAMPLES:-100}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-32}}"
MIN_COVERAGE="${MIN_COVERAGE:-0.85}"

[ -f "$DB" ] || { echo "Missing DB: $DB" >&2; exit 1; }

echo "=== sampling experiment ==="
echo "DB          : $DB"
echo "OUTDIR      : $OUTDIR"
echo "RESAMPLES   : $RESAMPLES"
echo "JOBS        : $JOBS"
echo "MIN_COVERAGE: $MIN_COVERAGE"
echo

# shellcheck disable=SC2046
srun $(slurm_srun_prefix) -n1 -N1 --cpus-per-task="$JOBS" \
    bash -c '
        source "'"$CKS_ROOT"'/benchmarks/common/slurm_common.sh"
        slurm_cd_repo
        exec "$(slurm_python)" -u "'"$CKS_ROOT"'/benchmarks/sampling/sampling_experiment.py" \
            --db "'"$DB"'" \
            --outdir "'"$OUTDIR"'" \
            --resamples "'"$RESAMPLES"'" \
            --jobs "'"$JOBS"'" \
            --min-coverage "'"$MIN_COVERAGE"'"
    '

echo
echo "Done. Results in $OUTDIR"
