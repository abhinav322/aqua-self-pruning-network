
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/sweep_full_v1")
INPUT_CSV = ROOT / "all_results.csv"
OUTPUT_CSV = ROOT / "aggregated_results.csv"
OUTPUT_PLOT = ROOT / "accuracy_sparsity.png"

EXPECTED_SEEDS = {42, 123, 2026}
TARGETS = [0.0, 0.50, 0.75, 0.90, 0.95]


def load_results():
    with open(INPUT_CSV, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 27:
        raise ValueError(
            f"Expected 27 experiments, found {len(rows)}"
        )

    return rows


def aggregate_results(rows):
    groups = defaultdict(list)

    for row in rows:
        seed = int(row["seed"])
        method = row["method"]
        target = float(row["target_sparsity"])

        # Treat zero sparsity as one shared dense baseline.
        if target == 0.0:
            method = "dense"

        groups[(method, target)].append({
            "seed": seed,
            "accuracy": float(row["test_accuracy"]),
            "loss": float(row["test_loss"]),
            "achieved": float(row["achieved_sparsity"]),
            "removed": int(row["removed_weights"]),
        })

    expected_groups = {
        ("dense", 0.0)
    }

    for target in TARGETS[1:]:
        for method in ("magnitude", "taylor"):
            expected_groups.add((method, target))

    if set(groups) != expected_groups:
        raise ValueError(
            "Experiment groups are missing or unexpected"
        )

    summaries = []

    for (method, target), runs in sorted(
        groups.items(),
        key=lambda item: (item[0][1], item[0][0])
    ):
        seeds = {run["seed"] for run in runs}

        if len(runs) != 3 or seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"Incomplete or duplicate seeds: "
                f"{method}, {target}"
            )

        accuracies = np.array([
            run["accuracy"] for run in runs
        ])

        losses = np.array([
            run["loss"] for run in runs
        ])

        achieved = np.array([
            run["achieved"] for run in runs
        ])

        removed = {
            run["removed"] for run in runs
        }

        if len(removed) != 1:
            raise ValueError(
                "Inconsistent weight counts across seeds"
            )

        summary = {
            "method": method,
            "target_sparsity": target,
            "mean_achieved_sparsity": float(
                np.mean(achieved)
            ),
            "removed_weights": next(iter(removed)),
            "n_seeds": len(runs),
            "mean_test_accuracy": float(
                np.mean(accuracies)
            ),
            "std_test_accuracy": float(
                np.std(accuracies, ddof=1)
            ),
            "mean_test_loss": float(
                np.mean(losses)
            ),
        }

        summaries.append(summary)

    return summaries


def save_csv(summaries):
    with open(
        OUTPUT_CSV, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summaries[0].keys())
        )

        writer.writeheader()
        writer.writerows(summaries)


def make_plot(summaries):
    dense = next(
        row for row in summaries
        if row["method"] == "dense"
    )

    plt.figure(figsize=(9, 6))

    for method in ("magnitude", "taylor"):
        selected = [
            dense,
            *[
                row for row in summaries
                if row["method"] == method
            ]
        ]

        selected.sort(
            key=lambda row: row["target_sparsity"]
        )

        x = [
            row["mean_achieved_sparsity"] * 100
            for row in selected
        ]

        y = [
            row["mean_test_accuracy"] * 100
            for row in selected
        ]

        errors = [
            row["std_test_accuracy"] * 100
            for row in selected
        ]

        plt.errorbar(
            x,
            y,
            yerr=errors,
            marker="o",
            capsize=4,
            linewidth=2,
            label=method.capitalize()
        )

    plt.xlabel("Achieved weight sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs. Sparsity (3 Seeds)")
    plt.xticks([0, 50, 75, 90, 95])
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=180)
    plt.close()


def main():
    rows = load_results()
    summaries = aggregate_results(rows)

    save_csv(summaries)
    make_plot(summaries)

    print("\nAGGREGATED RESULTS")
    print("-" * 72)

    for row in summaries:
        print(
            f"{row['method']:10s} | "
            f"Target: {row['target_sparsity']:5.0%} | "
            f"Achieved: "
            f"{row['mean_achieved_sparsity']:6.2%} | "
            f"Accuracy: "
            f"{row['mean_test_accuracy']:6.2%} "
            f"+/- {row['std_test_accuracy']:5.2%}"
        )

    print(f"\nCSV saved to: {OUTPUT_CSV}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
