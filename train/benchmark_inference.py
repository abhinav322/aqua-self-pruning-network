
import csv
import time
from pathlib import Path

import numpy as np

from nn.sparse_inference import SparseLinear


ROOT = Path("results/sweep_full_v1")
OUTPUT = ROOT / "inference_benchmark.csv"

CHECKPOINTS = [
    "pruned_magnitude_0_seed42",
    "pruned_taylor_50_seed42",
    "pruned_taylor_75_seed42",
    "pruned_taylor_90_seed42",
    "pruned_taylor_95_seed42",
    "pruned_magnitude_90_seed42",
]

BATCH_SIZES = [1, 32, 128, 1024]

WARMUP = 20
REPEATS = 7
ITERATIONS = 200


def load_checkpoint(folder):
    path = ROOT / folder / "model.npz"

    with np.load(path, allow_pickle=False) as data:
        return {
            key: data[key].copy()
            for key in data.files
        }


def dense_forward(x, model):
    hidden = np.maximum(
        x @ (model["w1"] * model["mask1"])
        + model["b1"],
        0.0
    )

    return (
        hidden @ (model["w2"] * model["mask2"])
        + model["b2"]
    )


def prepare_dense_weights(model):
    """
    Precompute effective weights once before inference.
    """
    return {
        "w1": model["w1"] * model["mask1"],
        "b1": model["b1"],
        "w2": model["w2"] * model["mask2"],
        "b2": model["b2"],
    }


def optimized_dense_forward(x, prepared):
    """
    Dense inference without repeated mask multiplication.
    """
    hidden = np.maximum(
        x @ prepared["w1"] + prepared["b1"],
        0.0
    )

    return (
        hidden @ prepared["w2"] + prepared["b2"]
    )



def make_sparse_layers(model):
    layer1 = SparseLinear(
        model["w1"],
        model["b1"],
        model["mask1"]
    )

    layer2 = SparseLinear(
        model["w2"],
        model["b2"],
        model["mask2"]
    )

    return layer1, layer2


def sparse_forward(x, layers):
    layer1, layer2 = layers

    hidden = np.maximum(
        layer1.forward(x),
        0.0
    )

    return layer2.forward(hidden)


def measure_latency(function):
    # Warm up the implementation.
    for _ in range(WARMUP):
        function()

    measurements = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        for _ in range(ITERATIONS):
            function()

        elapsed = time.perf_counter() - start

        # Average milliseconds per inference call.
        measurements.append(
            elapsed * 1000.0 / ITERATIONS
        )

    return float(np.median(measurements))


def calculate_costs(model, layers, batch_size):
    weight_count = (
        model["w1"].size + model["w2"].size
    )

    active_count = (
        layers[0].nnz + layers[1].nnz
    )

    dense_multiplications = weight_count * batch_size
    sparse_multiplications = active_count * batch_size

    # Count numeric array storage, including biases.
    dense_bytes = sum(
        model[name].nbytes
        for name in ("w1", "b1", "w2", "b2")
    )

    sparse_bytes = (
        layers[0].bias.nbytes
        + layers[1].bias.nbytes
    )

    for layer in layers:
        for indices, values in zip(
            layer.indices, layer.values
        ):
            sparse_bytes += (
                indices.nbytes + values.nbytes
            )

    return {
        "total_weights": weight_count,
        "active_weights": active_count,
        "achieved_sparsity": (
            1.0 - active_count / weight_count
        ),
        "dense_weight_multiplications": (
            dense_multiplications
        ),
        "sparse_weight_multiplications": (
            sparse_multiplications
        ),
        "arithmetic_reduction": (
            1.0 - sparse_multiplications
            / dense_multiplications
        ),
        "dense_parameter_bytes": dense_bytes,
        "sparse_array_bytes": sparse_bytes,
    }


def benchmark():
    results = []

    rng = np.random.default_rng(999)

    for folder in CHECKPOINTS:
        model = load_checkpoint(folder)

        prepared_dense = prepare_dense_weights(model)

        layers = make_sparse_layers(model)

        for batch_size in BATCH_SIZES:
            x = rng.normal(
                size=(batch_size, model["w1"].shape[0])
            )

            # Confirm numerical correctness before timing.
            expected = dense_forward(x, model)

            optimized = optimized_dense_forward(
                x, prepared_dense
            )

            actual = sparse_forward(x, layers)

            np.testing.assert_allclose(
                optimized,
                expected,
                rtol=1e-10,
                atol=1e-10
            )

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-10,
                atol=1e-10
            )

            dense_time = measure_latency(
                lambda: dense_forward(x, model)
            )

            optimized_dense_time = measure_latency(
                lambda: optimized_dense_forward(
                    x, prepared_dense
                )
            )

            sparse_time = measure_latency(
                lambda: sparse_forward(x, layers)
            )

            costs = calculate_costs(
                model,
                layers,
                batch_size
            )

            result = {
                "checkpoint": folder,
                "batch_size": batch_size,
                **costs,
                "dense_latency_ms": dense_time,
                "optimized_dense_latency_ms": optimized_dense_time,
                "sparse_latency_ms": sparse_time,
                "measured_speedup": (
                    dense_time / sparse_time
                ),
                "speedup_vs_optimized_dense": (
                    optimized_dense_time / sparse_time
                ),
            }

            results.append(result)

            print(
                f"{folder} | batch={batch_size} | "
                f"dense={dense_time:.4f} ms | "
                f"optimized={optimized_dense_time:.4f} ms | "
                f"sparse={sparse_time:.4f} ms | "
                f"speedup_vs_optimized="
                f"{optimized_dense_time/sparse_time:.2f}x",
                flush=True
            )

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark saved to: {OUTPUT}")


if __name__ == "__main__":
    benchmark()
