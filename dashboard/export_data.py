import csv
import json
from pathlib import Path

import numpy as np


# Paths are resolved relative to this script,
# so it works regardless of the terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "sweep_full_v1"
OUTPUT_FILE = PROJECT_ROOT / "dashboard" / "data" / "aqua_results.json"


def read_csv(path):
    """Load CSV rows as dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_experiment(row):
    """Load one recorded AQUA experiment."""

    seed = int(row["seed"])
    method = row["method"]
    target = float(row["target_sparsity"])

    target_percent = round(target * 100)

    # The dense baseline exists once per seed.
    # Nonzero targets have both pruning methods.
    if target_percent == 0:
        folder_name = f"pruned_magnitude_0_seed{seed}"
    else:
        folder_name = (
            f"pruned_{method}_{target_percent}_seed{seed}"
        )

    folder = RESULTS_DIR / folder_name

    with (folder / "summary.json").open(
        "r", encoding="utf-8"
    ) as file:
        summary = json.load(file)

    history_rows = read_csv(folder / "history.csv")

    history = []

    for item in history_rows:
        history.append({
            "epoch": int(item["epoch"]),
            "train_loss": float(item["train_loss"]),
            "train_accuracy": float(item["train_accuracy"]),
            "val_loss": float(item["val_loss"]),
            "val_accuracy": float(item["val_accuracy"]),
            "sparsity": float(item["sparsity"]),
        })

    # Extract the actual trained weights and masks.
    with np.load(folder / "model.npz") as model:
        w1 = model["w1"]
        w2 = model["w2"]

        mask1 = model["mask1"]
        mask2 = model["mask2"]

        assert w1.shape == (2, 32)
        assert w2.shape == (32, 2)

        assert mask1.shape == w1.shape
        assert mask2.shape == w2.shape

        active_count = int(
            np.count_nonzero(mask1)
            + np.count_nonzero(mask2)
        )

        total_count = mask1.size + mask2.size

        # Check exported masks against recorded metrics.
        assert total_count == summary["total_weights"]
        assert active_count == summary["active_weights"]

        assert (
            total_count - active_count
            == summary["removed_weights"]
        )

        weights = {
            "w1": w1.tolist(),
            "w2": w2.tolist(),
            "mask1": mask1.tolist(),
            "mask2": mask2.tolist(),
        }

    assert len(history) == summary["epochs"]

    return {
        "id": folder_name,
        "summary": summary,
        "history": history,
        "network": weights,
    }


def main():
    print("Exporting AQUA experiment data...")

    rows = read_csv(RESULTS_DIR / "all_results.csv")

    experiments = []

    for index, row in enumerate(rows, start=1):
        experiment = load_experiment(row)

        experiments.append(experiment)

        print(
            f"{index}/{len(rows)} "
            f"{experiment['id']} "
            f"loaded successfully"
        )

    assert len(experiments) == 27

    output = {
        "project": "AQUA Platform",
        "dashboard": "AQUA Observatory",
        "data_source": "Recorded AQUA experiments",
        "architecture": [2, 32, 2],
        "total_weights": 128,
        "experiments": experiments,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, separators=(",", ":"), allow_nan=False)

    size_kb = OUTPUT_FILE.stat().st_size / 1024

    print()
    print(f"Successfully exported {len(experiments)} experiments.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()