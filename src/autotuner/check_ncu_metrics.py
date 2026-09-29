"""
check_ncu_metrics.py — verify which NCU_METRICS are valid on this device.

Runs ncu with all metrics from ncu_worker.NCU_METRICS on a trivial torch.matmul,
then reports which ones returned a value and which were missing/zero.

Usage:
    python src/autotuner/check_ncu_metrics.py [--ncu-bin ncu]
"""
import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autotuner"))
from ncu_worker import NCU_METRICS

TARGET = """
import torch
a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
b = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
torch.matmul(a, b)
torch.cuda.synchronize()
"""

def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-bin", default="ncu")
    args = parser.parse_args()
    # fmt: on

    metrics_arg = ",".join(NCU_METRICS.values())
    cmd = [
        args.ncu_bin,
        "--csv",
        "--metrics", metrics_arg,
        "--clock-control", "none",
        sys.executable, "-c", TARGET,
    ]

    print(f"Running ncu with {len(NCU_METRICS)} metrics on a trivial matmul...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print("ncu failed:")
        print(result.stderr[-2000:])
        sys.exit(1)

    # Parse CSV — ncu emits one row per metric
    lines = [line for line in result.stdout.splitlines() if not line.startswith("==")]
    if not lines:
        print("No output from ncu.")
        sys.exit(1)

    found: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        ncu_id = row.get("Metric Name", "").strip('"').strip()
        val    = row.get("Metric Value", "").strip('"').strip().replace(",", "")
        if ncu_id:
            found[ncu_id] = val

    {v: k for k, v in NCU_METRICS.items()}

    ok, na, missing = [], [], []
    for db_col, ncu_id in NCU_METRICS.items():
        if ncu_id not in found:
            missing.append((db_col, ncu_id))
        elif found[ncu_id].lower() in ("", "n/a"):
            na.append((db_col, ncu_id))
        else:
            ok.append((db_col, ncu_id, found[ncu_id]))

    print(f"\n{'='*60}")
    print(f"  Has value : {len(ok)}/{len(NCU_METRICS)}")
    print(f"  N/A       : {len(na)}/{len(NCU_METRICS)}  (metric valid, not used by this kernel)")
    print(f"  Missing   : {len(missing)}/{len(NCU_METRICS)}  (metric name rejected by ncu)")
    print(f"{'='*60}")

    if ok:
        print("\nMetrics with values:")
        for db_col, ncu_id, val in ok:
            print(f"  ✓  {db_col:<30} = {val}")

    if na:
        print("\nN/A for this kernel (expect real values on CUTLASS kernels):")
        for db_col, ncu_id in na:
            print(f"  ~  {db_col:<30}  ({ncu_id})")

    if missing:
        print("\nRejected by ncu — fix these metric names:")
        for db_col, ncu_id in missing:
            print(f"  ✗  {db_col:<30}  ({ncu_id})")

if __name__ == "__main__":
    main()
