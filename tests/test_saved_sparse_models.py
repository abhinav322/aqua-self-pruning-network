
from pathlib import Path

import numpy as np
import pytest

from nn.sparse_inference import SparseLinear


ROOT = Path("results/sweep_full_v1")

CHECKPOINTS = [
    "pruned_magnitude_0_seed42",
    "pruned_taylor_75_seed42",
    "pruned_taylor_90_seed42",
    "pruned_taylor_95_seed42",
    "pruned_magnitude_90_seed42",
]


@pytest.mark.parametrize("folder", CHECKPOINTS)
def test_saved_model_sparse_matches_dense(folder):
    checkpoint = ROOT / folder / "model.npz"

    assert checkpoint.exists(), (
        f"Missing saved model: {checkpoint}"
    )

    with np.load(checkpoint, allow_pickle=False) as model:
        w1 = model["w1"]
        b1 = model["b1"]
        mask1 = model["mask1"]

        w2 = model["w2"]
        b2 = model["b2"]
        mask2 = model["mask2"]

    rng = np.random.default_rng(9876)

    x = rng.normal(
        size=(128, w1.shape[0])
    )

    # Existing dense masked inference.
    dense_hidden = np.maximum(
        x @ (w1 * mask1) + b1,
        0.0
    )

    dense_logits = (
        dense_hidden @ (w2 * mask2) + b2
    )

    # Our new sparse-aware inference.
    sparse_layer1 = SparseLinear(
        w1, b1, mask1
    )

    sparse_layer2 = SparseLinear(
        w2, b2, mask2
    )

    sparse_hidden = np.maximum(
        sparse_layer1.forward(x),
        0.0
    )

    sparse_logits = sparse_layer2.forward(
        sparse_hidden
    )

    np.testing.assert_allclose(
        sparse_logits,
        dense_logits,
        rtol=1e-10,
        atol=1e-10
    )

    # Classification decisions should also match.
    np.testing.assert_array_equal(
        np.argmax(sparse_logits, axis=1),
        np.argmax(dense_logits, axis=1)
    )

    # Count actual active connections.
    expected_active = (
        np.count_nonzero(mask1)
        + np.count_nonzero(mask2)
    )

    assert (
        sparse_layer1.nnz + sparse_layer2.nnz
        == expected_active
    )
