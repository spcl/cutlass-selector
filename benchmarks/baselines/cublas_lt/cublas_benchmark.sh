#!/bin/bash
#SBATCH --job-name=cublas_benchmark
#SBATCH --output=cublas_bench_%j.out
#SBATCH --error=cublas_bench_%j.err
#SBATCH --nodes=1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=00:30:00
#
# cuBLASLt reference benchmark — builds src/baseline/cublas_lt, then runs it.
# Submit from the repository root:  sbatch benchmarks/baselines/cublas_lt/cublas_benchmark.sh

REPO="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1

echo "Compiling cublas_profiler..."
make -C src/baseline/cublas_lt
echo "Build done."

bash benchmarks/baselines/cublas_lt/run_benchmark.sh
echo "All done."
