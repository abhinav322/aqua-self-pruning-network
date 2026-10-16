
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SEEDS = [42, 123, 2026]
SPARSITIES = [0.0, 0.50, 0.75, 0.90, 0.95]
METHODS = ["magnitude", "taylor"]

OUTPUT_ROOT = Path(
    os.environ.get(
        "AQUA_SWEEP_ROOT",
        "results/sweep_v1"
    )
)


def build_experiments(smoke=False):
    if smoke:
        return [
            (42, "magnitude", 0.0),
            (42, "magnitude", 0.50),
            (42, "taylor", 0.50),
        ]

    experiments = []

    for seed in SEEDS:
        # Only one dense reference is needed.
        experiments.append(
            (seed, "magnitude", 0.0)
        )

        for sparsity in SPARSITIES[1:]:
            for method in METHODS:
                experiments.append(
                    (seed, method, sparsity)
                )

    return experiments


def run_experiment(seed, method, sparsity):
    folder_name = (
        f"pruned_{method}_"
        f"{int(sparsity * 100)}_seed{seed}"
    )

    output_dir = OUTPUT_ROOT / folder_name
    summary_path = output_dir / "summary.json"

    # Create a separate environment for the child process.
    env = os.environ.copy()
    env["AQUA_SEED"] = str(seed)
    env["AQUA_METHOD"] = method
    env["AQUA_TARGET_SPARSITY"] = str(sparsity)
    env["AQUA_OUTPUT_ROOT"] = str(OUTPUT_ROOT)

    print(
        f"Running seed={seed}, "
        f"method={method}, "
        f"sparsity={sparsity:.0%}",
        flush=True
    )

    # Use the same Python interpreter as our virtual environment.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "train.run_pruned"
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the terminal output from every run.
    log_path = output_dir / "run.log"

    log_path.write_text(
        result.stdout + "\n" + result.stderr,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment failed: {folder_name}\n"
            f"See {log_path}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary: {summary_path}"
        )

    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)

    # Confirm the summary belongs to this configuration.
    if (
        summary["seed"] != seed
        or summary["method"] != method
        or not np.isclose(
            summary["target_sparsity"],
            sparsity
        )
    ):
        raise RuntimeError(
            f"Configuration mismatch in {summary_path}"
        )

    total = int(summary["total_weights"])
    expected_removed = int(
        np.floor(sparsity * total + 1e-10)
    )

    if summary["removed_weights"] != expected_removed:
        raise RuntimeError(
            f"Incorrect pruning budget in {summary_path}"
        )

    print(
        f"Completed: "
        f"test accuracy={summary['test_accuracy']:.2%}, "
        f"sparsity={summary['achieved_sparsity']:.2%}",
        flush=True
    )

    return summary


def save_combined_results(summaries):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_ROOT / "all_results.csv"

    if not summaries:
        raise ValueError("No results were collected")

    fields = list(summaries[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nCombined results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--smoke",
        action="store_true",
        help="Run only three validation experiments"
    )

    group.add_argument(
        "--full",
        action="store_true",
        help="Run all 27 experiment configurations"
    )

    args = parser.parse_args()

    experiments = build_experiments(
        smoke=args.smoke
    )

    summaries = []

    for index, (seed, method, sparsity) in enumerate(
        experiments,
        start=1
    ):
        print(
            f"\nExperiment {index}/{len(experiments)}",
            flush=True
        )

        summary = run_experiment(
            seed,
            method,
            sparsity
        )

        summaries.append(summary)

        # Save progress after each completed run.
        save_combined_results(summaries)

    print("\nAll requested experiments completed.")


if __name__ == "__main__":
    main()
