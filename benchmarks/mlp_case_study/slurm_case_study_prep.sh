#!/bin/bash
#SBATCH --job-name=mlp-case-prep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# CPU-only: trace internal MLP GEMMs + rank-1 / nvMMH candidate selection.
#
#   sbatch benchmarks/mlp_case_study/slurm_case_study_prep.sh

export CASE_STEPS=prep
export CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$CKS_ROOT/benchmarks/mlp_case_study/study_case_job.sh"
