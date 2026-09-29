# Hardware-Aware Features for CUTLASS Kernel Selection

A single CUTLASS GEMM can be implemented in tens of thousands of ways. This repository
picks a good one **without compiling or running any of them**: every candidate
configuration is described by statically computable estimates of the hardware
behaviour it induces (wave quantization, shared-memory and register footprint,
arithmetic intensity, L2 reuse, …), and a learning-to-rank model orders the candidates
of each problem.

It contains the full pipeline used for the paper, for NVIDIA Hopper (SM90) GPUs:

```
 config space ──► compile + benchmark ──► features ──► train ranker ──► evaluate
 src/autotuner        src/autotuner       src/model      src/model       src/eval
 src/planning         (SQLite registry)
```

| Folder | Contents |
|---|---|
| `src/autotuner` | CUTLASS SM90 config space, kernel template, compile / benchmark / Nsight Compute scheduler, registry |
| `src/planning` | training shape grid and measurement plans |
| `src/model` | feature extraction, MLP and XGBoost rankers, evaluation harness |
| `src/eval` | held-out evaluation against nvMatmulHeuristics and cuBLASLt |
| `src/baseline` | nvMatmulHeuristics wrapper, cuBLASLt benchmark |
| `benchmarks/` | the cluster jobs and studies behind the paper ([README](benchmarks/README.md)) |

## Quick start

Requirements: Linux, CUDA 12.3+ with `nvcc`, Python 3.10+. Benchmarking needs an SM90
GPU (H100 / GH200); everything else, including compiling kernels, runs without one.

```bash
git clone --recursive https://github.com/spcl/cutlass-selector.git
cd cutlass-selector
python -m venv .venv && source .venv/bin/activate
pip install torch                      # the build matching your CUDA, see pytorch.org
pip install -r requirements.txt
pip install nvidia-matmul-heuristics   # optional: the nvMatmulHeuristics baseline
export PYTHONPATH=$PWD/src             # all commands run from the repository root

python -m pytest src/model/tests
```

Tested with PyTorch 2.9.1 (the paper's runs) and with Python 3.14, PyTorch 2.10–2.14 and
XGBoost 3.3–3.4. Generated files go to `build_cache/`, `artifacts/` and `datasets/`, all
gitignored. Every script documents its options with `--help`.

## 1. Collect data

Plan which `(config, M, N, K)` pairs to measure, then compile and benchmark them. All
state lives in a SQLite registry (`build_cache/autotuner.db`), so every step can be
interrupted and resumed.

```bash
python src/planning/plan.py shapes                           # training shape grid
python src/planning/plan.py write --tag sweep --shapes datasets/plans/shapes.json

python src/autotuner/scheduler.py --tag sweep --phase compile --compile-jobs 16
python src/autotuner/scheduler.py --tag sweep --phase eval --num-gpus 1
python src/autotuner/scheduler.py --tag sweep --phase ncu --num-gpus 1   # optional

python src/autotuner/query_db.py summary                     # progress
python src/autotuner/query_db.py best 4096 4096 4096         # fastest kernels for a shape
```

`plan.py write` takes `--dtype {bf16,fp32,fp8_e4m3}`, `--layouts` and `--exhaustive` (every
valid config per problem, for an evaluation oracle). The config space and its validity
rules are in `src/autotuner/config_space.py`: 242,971 valid BF16 configurations across the
four layouts.

## 2. Build features and train

```bash
python src/model/features.py --db build_cache/autotuner.db --train-tags sweep \
    --eval-db oracle.db --eval-tag oracle --out artifacts/analysis/paper/features.parquet

python src/model/train_mlp.py --features artifacts/analysis/paper/features.parquet --loss mse
python src/model/train_xgb.py --features artifacts/analysis/paper/features.parquet --loss mse
```

Groups tagged `--eval-tag` were measured exhaustively and form the evaluation split. The
trainers report selection quality on it (regret against the measured best, top-1,
within 5%), not prediction error. `--feature-set structural` drops the hardware-derived
features; `--val-frac`, `--hidden` / `--max-depth` and `--init-checkpoint` / `--init-model`
control validation, capacity and fine-tuning.

## 3. Evaluate

Benchmark the kernel each method picks on problems that were not used for training.
[`src/eval/README.md`](src/eval/README.md) describes the methodology.

```bash
OUT=artifacts/eval/out
python src/eval/shapes.py --holdout build_cache/autotuner.db:sweep --out $OUT/shapes.json
python src/eval/propose.py --method nvmmh --shapes $OUT/shapes.json --out $OUT/proposals_nvmmh.json
python src/eval/propose.py --method mlp_full --model-dir artifacts/analysis/paper/mlp_mse \
    --scheduler-from $OUT/proposals_nvmmh.json --shapes $OUT/shapes.json --out $OUT/proposals_mlp_full.json
python src/eval/run.py --phase plan,compile,bench --proposals $OUT/proposals_*.json
make -C src/baseline/cublas_lt && python src/eval/cublas.py
python src/eval/report.py
```

## Reproducing the paper

[`benchmarks/README.md`](benchmarks/README.md) lists the Slurm jobs, in order, that
produced the paper's data, models and results on CSCS Alps, together with the
cross-precision, epilogue-fusion, DeepBench and baseline studies.

## Citation

Using our repository in your work? Please reference us using the provided citation:

```bibtex
@misc{chandran2026hardware,
  title={Hardware-Aware Features for CUTLASS Kernel Selection},
  author={Shriram Chandran and Dominic Rinderer and Yakup Budanaz and Alexandru Calotoiu and Marcin Copik and Torsten Hoefler},
  year={2026},
  month = {Sep},
  eprint={2609.35587},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.35587},
}
```

## License

See [LICENSE](LICENSE).
