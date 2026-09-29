# Dtype transfer experiments

See **[STUDY.md](STUDY.md)** for the full submit cheat sheet with `--time` on every `sbatch`.

```bash
# from the repository root
# train prep
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_dtype/slurm_study_train_fp32.sh
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_dtype/slurm_study_train_fp8.sh

# train (after prep)
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_dtype/slurm_study_train_array_fp32.sh
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_dtype/slurm_study_train_array_fp8.sh
```

DBs: `~/autotuner/autotuner_sweep_fp32_tn.db`, `~/autotuner/autotuner_sweep_fp8_tn.db`

Legacy single-fraction finetune: **[EXPERIMENT.md](EXPERIMENT.md)**
