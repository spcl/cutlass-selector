# Evaluation

Measures the trained selector against nvMatmulHeuristics and cuBLASLt on held-out GEMM
problems, on an SM90 GPU. Everything is a *measured* number: no predicted runtimes are
compared, and no method is scored on a kernel that was not actually built and run.

## Pipeline

```bash
OUT=artifacts/eval/out

# 1. held-out problems (CPU); --holdout is required, see below
python src/eval/shapes.py --holdout build_cache/autotuner.db:sweep --out $OUT/shapes.json

# 2. candidate kernels per problem (CPU)
python src/eval/propose.py --method nvmmh --backend nvmmh --gpu H100_SXM \
    --shapes $OUT/shapes.json --out $OUT/proposals_nvmmh.json
python src/eval/propose.py --method mlp_full --backend mlp \
    --model-dir artifacts/analysis/paper/mlp_mse \
    --scheduler-from $OUT/proposals_nvmmh.json \
    --shapes $OUT/shapes.json --out $OUT/proposals_mlp_full.json

# 3. register + build (CPU; cross-compiles for sm_90a anywhere)
python src/eval/run.py --phase plan,compile --proposals $OUT/proposals_*.json

# 4. measure (SM90 required)
python src/eval/run.py --phase bench
make -C src/baseline/cublas_lt && python src/eval/cublas.py --shapes $OUT/shapes.json

# 5. reduce (CPU); benchmarks/eval/plot.py draws the figures
python src/eval/report.py
```

`--backend` is one of `nvmmh`, `mlp`, `xgb`, `ridge` or `random`; `--method` is the label
the results are stored under. `benchmarks/eval/slurm_eval.sh` runs all steps on the cluster
for nvMMH and the four trained selectors.

Phases are resumable: `plan` is idempotent, `compile` skips built kernels, `bench` picks up
only `pending` rows, and `cublas.py` skips problems already stored.

## What is being compared

Each series is **best measured GFLOP/s among the first `b` candidates the method proposed**
(stored in the legacy ``mean_tflops`` DB columns),
where `b` is the number of kernels compiled and benchmarked for that problem. The methods
differ only in what orders their candidates:

| series | ordering | notes |
|---|---|---|
| `ours@b` | the model's rank | tile scheduler from nvMMH's rank-1 pick (`--scheduler-from`) |
| `cublas@b` | cuBLASLt's heuristic rank | `cublas@1` = algo 0 = its own top pick, no search |
| `nvmmh@b` | see below | |
| `nvmmh@8` | — | best of the **8 schedule variants of nvMMH's single rank-1 recommendation** |

`nvmmh@8` is the headline nvMMH number. The 8 variants
share nvMMH's tile, cluster, rasterization, swizzle and split-k, and differ only in the
mainloop/epilogue/tile-scheduler combination, which nvMMH does not emit — mirroring CUTLASS's
own `get_valid_schedules` workflow. It is **not** a search over the config space.

For `b < 8`, `nvmmh@b` is the **expectation over a uniformly random variant order**, computed
exactly over all `C(8, b)` subsets. The model ranks its candidates and cuBLASLt ranks its
algos, so "the first `b`" is a property of those methods; nvMMH expresses no schedule
preference, so taking the first `b` in list order would report an artifact of how
`src/baseline/nvmmh/translate.py` happens to be written.

**A failed candidate scores 0 GFLOP/s**, not "missing": it consumed budget and produced no
usable kernel. Every series is therefore a plain max and monotone in `b`. Problems where a
series is 0 are excluded from *ratio* statistics (reported as `-n unusable`) but still counted
in the coverage table, which is where compile failures and CUTLASS rejections surface.

Aggregates are geometric means of per-problem ratios with bootstrap 95% CIs — never a mean of
GFLOP/s across shapes.

## Measurement

Both harnesses use the same protocol. `src/baseline/cublas_lt/cublas_profiler` takes
`--warmup/--rounds/--iters-per-round` and `src/eval/cublas.py` passes the live values from
`src/autotuner/profile_worker.py`, so the two cannot drift apart. Both rotate through
`ceil(3·L2/(A+B))` operand copies (capped at 64) so timed iterations read cold operands, and
both report mean/std over rounds. CUTLASS kernels are measured by the autotuner's own
`benchmark()` and `_BufferPool` — the same code path that produced the training data.

## Held-out problems

`shapes.py` requires at least one `--holdout DB:TAG` (repeatable): pointing it at the wrong
database or tag would silently produce an evaluation set overlapping the training data. For
each source it excludes everything planned under those tags *and* everything ever benchmarked
in that DB, and asserts zero overlap before writing. Two strata, 50/50:

- `aligned32` — every dimension a multiple of 32, matching the training sweep's grid.
- `ragged8` — multiples of 8 but not 32, off the training distribution.

The sweep's 593 shapes are 564 fully 32-aligned with `K` 32-aligned in all of them, so
`ragged8` is a genuine generalization test.

## Known limitations

- **Small shapes are measured L2-resident.** The 64-copy cap means that when `A+B` is below
  about 3 MB, the whole rotation set still fits in L2 and operands never go cold. This affects
  *both* harnesses identically (the cap is `profile_worker`'s), so the comparison stays fair,
  but small-shape numbers are not cold-memory measurements. Deliberately not "fixed" here:
  diverging from the autotuner's constant would break comparability with the training data.
- **nvMMH has no GH200 SKU.** We use `H100_SXM`. Among the SM90 SXM presets the top-1
  recommendation is unchanged on ~95% of problems (`H100_NVL` 4.0%, `H200_SXM` 5.5% differ),
  while `H100_PCIE` (62.5%) and `A100_SXM_80GB` (98%) differ substantially — so the choice
  within the Hopper SXM family is immaterial. Worth re-checking if the cluster ships a newer
  nvMMH with a GH200 entry.
- **The model does not predict rasterization, swizzle or split-k.** nvMMH's recommendations
  run with its own `cta_order`, `swizzle_factor` and `split_k`. With `--scheduler-from`, the
  model's picks run with the same three values nvMMH chose for that problem, so both methods
  differ only in the kernel configuration; without it they run on CUTLASS defaults.
- **The schedule fanout is frozen deliberately.** `SCHEDULE_VARIANTS` in
  `src/baseline/nvmmh/translate.py` is a hand-maintained list, not derived from the config space
  at import, so editing the search space cannot silently change what nvMMH was given or
  invalidate a completed run. Re-freeze it on purpose and re-run the baseline when the space
  changes.
- **A checkpoint with unknown features is refused.** `propose.py` reads the feature list from
  the model's `metrics.json` (or `features.manifest.json`) and raises if the featurizer cannot
  produce one of them, rather than scoring a model on inputs it was not trained on.

## Artifacts

Everything generated lands in `artifacts/eval/out/` (gitignored): `shapes.json`, `proposals_*.json`,
`eval.db` (`configs`, `eval_runs`, `cublas_runs`), `report.csv`, and the figures. The shape set
is reproducible from `--seed`, so it is not tracked.
