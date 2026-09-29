# Epilogue-fusion transfer study

Replicates the dtype-transfer pipeline for **fused TN GEMMs** (5 fusion kinds per shape from
`benchmarks/data_collection/plan_fusion.sh`). Training data comes from the fusion fine-tune sweeps; features use
`src/model/features_fusion.py` (fusion-aware smem / bytes; same categoricals as unfused).

## Submit (copy-paste)

Run all commands from the repository root.

### Training

**Prep** (~1–2 h, CPU-heavy; run once per dtype or all three in parallel):

```bash
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_train_fp16.sh
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_train_fp32.sh
STUDY_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_train_fp8.sh
```

**Train** (24 finetune runs per dtype, parallel):

```bash
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_train_array_fp16.sh
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_train_array_fp32.sh
sbatch --time=05:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_train_array_fp8.sh
```

**Alternative — one sequential job per dtype** (prep + all 24 runs):

```bash
sbatch --time=18:00:00 benchmarks/transfer_fusion/slurm_study_train_fp16.sh
sbatch --time=18:00:00 benchmarks/transfer_fusion/slurm_study_train_fp32.sh
sbatch --time=18:00:00 benchmarks/transfer_fusion/slurm_study_train_fp8.sh
```

### Evaluation (after training)

**Prep** (eval shapes + nvMMH proposals + nvMMH bench, ~1–2 h per dtype):

```bash
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp16.sh
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp32.sh
STUDY_EVAL_STEPS=prep sbatch --time=02:00:00 benchmarks/transfer_fusion/slurm_study_eval_fp8.sh
```

**Bench** (24 model runs per dtype):

```bash
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp16.sh
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp32.sh
sbatch --time=04:00:00 --array=0-23%6 benchmarks/transfer_fusion/slurm_study_eval_array_fp8.sh
```

**Summarize** (after all array tasks finish — run per dtype or all three):

```bash
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval_fp16.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval_fp32.sh
STUDY_EVAL_STEPS=summarize sbatch --time=00:30:00 benchmarks/transfer_fusion/slurm_study_eval_fp8.sh
```

Local plots:

```bash
python benchmarks/transfer_fusion/plot_study.py --also-quick
```

## Prep on Daint (manual, optional)

```bash
# from the repository root
source .venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/benchmarks

# 1) Feature parquets (one per target dtype; no oracle eval in fusion DBs)
python benchmarks/transfer_fusion/build_study_features.py

# 2) Nested subsets (593 base shapes × fractions × seed 0)
python benchmarks/transfer_fusion/make_subsets.py

# 3) SMEM / feature audit (featurizer vs config_space_fusion)
python benchmarks/transfer_fusion/verify_fusion_features.py

# 4) Run manifest (72 runs = 3 dtypes × 24 finetune)
python -c "from transfer_fusion.study_common import write_run_manifest; print(write_run_manifest())"
```

Outputs live under `artifacts/analysis/paper/transfer_fusion_study/`.

## Design

| Axis | Values |
|------|--------|
| Target dtypes | FP16, FP32, FP8 E4M3 |
| Sweep DBs | `~/autotuner/autotuner_sweep_fusion_fp16.db`, `_fp32.db`, `_fp8.db` |
| Runs tags | `sweep_fusion_fp16`, `sweep_fusion_fp32`, `sweep_fusion_fp8` |
| Training fractions | 1%, 5%, 10%, 25%, 50%, 100% of **593 TN base shapes** |
| Subset seed | **0** (nested subsets, same as dtype study) |
| MLP | finetune from unfused BF16 × hardware-aware/structural → **2 per fraction** |
| XGB | warm-start from unfused BF16 `xgb_mse{,_structural}` (+200 trees) → **2 per fraction** |

**72 runs total** (24 finetune per dtype). Starters: same unfused BF16 checkpoints as the dtype study
(`mlp_mse`, `xgb_mse`, structural variants).

### Structural feature set

Same tile/cluster/schedule/layout columns as the dtype study. Each shape has one fusion kind
(fixed by `FUSION_SEED` in `plan_fusion.sh`, shared across dtypes); fusion affects
**numeric** features (smem carveout, bytes) but is **not** a model categorical.

### Feature mapping vs unfused TN study

| Feature family | Change for fusion |
|----------------|-------------------|
| Operand bytes / peak TC | Per target dtype (fp16/fp32/fp8) |
| `problem_arith_intensity`, `restream_factor` | Dtype-scaled A/B/C bytes |
| `smem_total` | Mainloop + TMA epilogue staging + **fusion visitor carveout** (bias kinds) |

## Shared memory (the tricky part)

Training features use an **analytic** smem model (not NCU at inference):

1. **Mainloop** — `stages × (tile_m×tile_k×bytes_a + tile_n×tile_k×bytes_b)`
2. **Epilogue TMA staging** — EpilogueTile subtiles + StagesC/D + ReuseSmem
   (`features_fusion.py` mirrors `config_space_fusion.estimate_epilogue_smem_bytes`)
3. **Fusion visitor** — for `bias`, `bias_relu`, `bias_gelu`:
   `smem_fusion = ceil128(tile_n × bytes_c)` (CTA-N bias vector; see CUTLASS
   `ScaledLinCombPerRowBiasEltAct` visitor storage)

`config_space_fusion.estimate_smem_total()` is the reference for Rule-7 validity in the
autotuner; `verify_fusion_features.py` checks **featurizer == config_space** on random sweep rows.

**CUTLASS caveat:** the visitor carveout is conservative (128 B aligned N-vector). If NCU
`smem_dynamic_bytes` on fused kernels systematically disagrees, tighten against the generated
`hopper_template_fusion.cu.j2` / CUTLASS `SharedStorage` layout before trusting
`blocks_per_sm` at low fractions.

## Notes

nvMMH baseline is **unfused** GEMM (bf16 proxy for fp16 fusion dtype).
