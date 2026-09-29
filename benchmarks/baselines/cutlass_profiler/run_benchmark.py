#!/usr/bin/env python3
"""Run cutlass_profiler for each (shape, dtype, layout) using compiled heuristic kernels.

Reads build/testlist.csv to enumerate all (shape, dtype, layout) combinations the
heuristics selected. For each group, invokes the profiler with a dtype-based wildcard
filter (e.g. cutlass3x_sm90_tensorop_gemm_bf16*). The compiled library already contains
only the heuristic-selected kernels, so the wildcard runs exactly those.

Usage:
    python3 run_benchmark.py [--testlist PATH] [--profiler PATH] [--output PATH]
                              [--warmup N] [--iters N]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

_LAYOUT = {"t": "row", "n": "column", "row": "row", "column": "column"}

# Dtype-based kernel wildcard — matches all compiled heuristic kernels for that dtype
_KERNEL_PATTERN = {
    "bf16": "cutlass3x_sm90_tensorop_gemm_bf16*",
    "e4m3": "cutlass3x_sm90_tensorop_gemm_e4m3*",
}


def parse_testlist(path: Path) -> dict:
    """Return {(m,n,k,dtype_a,la,lb): ref_row} with one entry per unique shape+layout."""
    groups = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            key = (
                int(row["m"]), int(row["n"]), int(row["k"]),
                row["dtype_a"], row["layout_a"], row["layout_b"],
            )
            if key not in groups:
                groups[key] = row  # keep first row as representative
    return groups


def _c_arg_from_op_name(op_name: str, layout_d: str) -> str:
    """Extract C dtype from compiled kernel name (position after A,B,acc in name).
    e.g. ...gemm_bf16_bf16_f32_void_bf16_... -> 'void'
         ...gemm_e4m3_e4m3_f32_e4m3_e4m3_... -> 'e4m3:column'
    """
    parts = op_name.split("_")
    c_dtype = parts[7]  # cutlass3x_sm90_tensorop_gemm_{A}_{B}_{acc}_{C}_...
    if c_dtype == "void":
        return "void"
    return f"{c_dtype}:{_LAYOUT[layout_d]}"


def build_profiler_cmd(profiler: str, row: dict, shape_key, outfile: str,
                       warmup: int, iters: int) -> list[str]:
    m, n, k, dtype_a, la, lb = shape_key
    la_full = _LAYOUT[la]
    lb_full = _LAYOUT[lb]
    ld_full = _LAYOUT[row["layout_d"]]
    c_arg = _c_arg_from_op_name(row["operation_name"], row["layout_d"])

    return [
        profiler,
        f"--m={m}", f"--n={n}", f"--k={k}",
        f"--A={dtype_a}:{la_full}",
        f"--B={row['dtype_b']}:{lb_full}",
        f"--C={c_arg}",
        f"--D={row['dtype_d']}:{ld_full}",
        f"--accumulator-type={row['dtype_acc']}",
        f"--kernels={_KERNEL_PATTERN[dtype_a]}",
        f"--warmup-iterations={warmup}",
        f"--profiling-iterations={iters}",
        "--verification-enabled=false",
        "--dist=uniform,min:-1,max:1,scale:-1",
        "--append",
        f"--output={outfile}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testlist", default="build/testlist.csv")
    parser.add_argument("--profiler", default="build/tools/profiler/cutlass_profiler")
    parser.add_argument("--output", default="results/cutlass_reference.csv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    testlist_path = script_dir / args.testlist
    profiler_path = script_dir / args.profiler
    out_path = script_dir / args.output

    if not testlist_path.exists():
        sys.exit(f"Testlist not found: {testlist_path} — run build.sh first")
    if not profiler_path.exists():
        sys.exit(f"Profiler not found: {profiler_path} — run build.sh first")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gemm_out = out_path.parent / (out_path.stem + ".gemm.csv")
    gemm_out.unlink(missing_ok=True)

    groups = parse_testlist(testlist_path)
    total = len(groups)
    print(f"Loaded {total} unique (shape, layout) combinations from testlist")

    for idx, (key, row) in enumerate(sorted(groups.items()), 1):
        m, n, k, dtype_a, la, lb = key
        la_short = "T" if _LAYOUT[la] == "row" else "N"
        lb_short = "T" if _LAYOUT[lb] == "row" else "N"
        tag = f"{dtype_a.upper()} {la_short}{lb_short}  {m}x{n}x{k}"
        print(f"[{idx}/{total}] {tag}", flush=True)
        cmd = build_profiler_cmd(
            str(profiler_path), row, key,
            str(out_path.parent / out_path.stem),
            args.warmup, args.iters,
        )
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"  WARNING: profiler exited {result.returncode}", flush=True)

    if gemm_out.exists():
        print(f"\nDone. Results in {gemm_out}")
        subprocess.run([sys.executable, str(script_dir / "parse_results.py"), str(gemm_out)])
    else:
        print(f"\nDone. No output file found — check {out_path.parent}/")


if __name__ == "__main__":
    main()
