
import csv
from pathlib import Path

import numpy as np


ROOT = Path("results/sweep_full_v1")
INPUT = ROOT / "inference_benchmark.csv"
OUTPUT = ROOT / "cost_summary.csv"

BATCH_SIZES = {1, 32, 128, 1024}


def main():
    with open(INPUT, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 24:
        raise ValueError(
            f"Expected 24 benchmark rows, found {len(rows)}"
        )

    required = {
        "optimized_dense_latency_ms",
        "sparse_latency_ms",
        "dense_parameter_bytes",
        "sparse_array_bytes",
        "dense_weight_multiplications",
        "sparse_weight_multiplications",
    }

    for row in rows:
        if not required.issubset(row):
            raise ValueError("Missing benchmark columns")

    checkpoints = {}

    for row in rows:
        name = row["checkpoint"]
        batch = int(row["batch_size"])

        if batch not in BATCH_SIZES:
            raise ValueError("Unexpected batch size")

        checkpoints.setdefault(name, {})

        if batch in checkpoints[name]:
            raise ValueError("Duplicate benchmark configuration")

        checkpoints[name][batch] = row

    if len(checkpoints) != 6:
        raise ValueError("Expected six checkpoints")

    results = []

    for name, batches in sorted(checkpoints.items()):
        if set(batches) != BATCH_SIZES:
            raise ValueError(
                f"Missing batch sizes for {name}"
            )

        for batch in sorted(batches):
            row = batches[batch]

            dense_ms = float(
                row["optimized_dense_latency_ms"]
            )
            sparse_ms = float(
                row["sparse_latency_ms"]
            )

            dense_bytes = int(
                row["dense_parameter_bytes"]
            )
            sparse_bytes = int(
                row["sparse_array_bytes"]
            )

            dense_ops = int(
                row["dense_weight_multiplications"]
            )
            sparse_ops = int(
                row["sparse_weight_multiplications"]
            )

            if not all(np.isfinite([dense_ms, sparse_ms])):
                raise ValueError("Non-finite timings")

            if min(dense_ms, sparse_ms, dense_bytes, dense_ops) <= 0:
                raise ValueError("Invalid positive measurement")

            if sparse_bytes < 0 or sparse_ops < 0:
                raise ValueError("Invalid sparse cost")

            results.append({
                "checkpoint": name,
                "batch_size": batch,
                "achieved_sparsity_pct": (
                    float(row["achieved_sparsity"]) * 100
                ),
                "arithmetic_reduction_pct": (
                    (1 - sparse_ops / dense_ops) * 100
                ),
                "array_storage_reduction_pct": (
                    (1 - sparse_bytes / dense_bytes) * 100
                ),
                "optimized_dense_ms": dense_ms,
                "sparse_ms": sparse_ms,
                "speedup": dense_ms / sparse_ms,
            })

    with open(
        OUTPUT, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nCOST SUMMARY: BATCH SIZE 1")
    print("-" * 100)

    for row in results:
        if row["batch_size"] != 1:
            continue

        print(
            f"{row['checkpoint']:30s} | "
            f"sparsity={row['achieved_sparsity_pct']:6.2f}% | "
            f"arithmetic={row['arithmetic_reduction_pct']:6.2f}% | "
            f"storage={row['array_storage_reduction_pct']:7.2f}% | "
            f"speedup={row['speedup']:.2f}x"
        )

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
