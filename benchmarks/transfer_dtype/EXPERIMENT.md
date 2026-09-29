# Dtype transfer fine-tune experiment

## Question

Do **BF16-pretrained** `mlp_mse` and `xgb_mse`, fine-tuned on a target dtype sweep, beat
**nvMMH** on the **broad GEMM eval**?

No scratch baseline. No in-domain oracle during finetune. Test = broad GEMM (phase 2).

## Slurm (phase 1)

```bash
sbatch benchmarks/transfer_dtype/slurm_finetune.sh           # fp32 + fp8
sbatch benchmarks/transfer_dtype/slurm_finetune_fp32.sh      # fp32 only
sbatch benchmarks/transfer_dtype/slurm_finetune_fp8.sh       # fp8 only
```

| Item | Value |
|------|--------|
| Source | `artifacts/analysis/paper/mlp_mse/model_mlp.pt` |
| Train DBs | `~/autotuner/autotuner_sweep_fp32_tn.db`, `~/autotuner/autotuner_sweep_fp8_tn.db` |
| Tags | `sweep_fp32_tn`, `sweep_fp8_tn` |
| MLP finetune | 48 epochs, lr `4e-5`, scaler `fit`, MSE, `--skip-eval` |
| XGB finetune | `+200` trees on top of BF16 `model_A.ubj`, MSE, `--skip-eval` |
| Artifacts | `.../mlp_mse/model_mlp.pt2`, `.../xgb_mse/model_A.ubj` |

Env overrides: `FINETUNE_EPOCHS`, `FINETUNE_LR`, `DB`, `SOURCE_CKPT`.

## Phase 2 — broad GEMM eval

The fine-tuned models are evaluated against nvMMH as part of the full study; see the
Evaluation section of [STUDY.md](STUDY.md).
