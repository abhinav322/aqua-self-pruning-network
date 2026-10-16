
import numpy as np
import pytest

from engine.tensor import Tensor
from nn.model import MLP
from nn.sparse_inference import SparseLinear, SparseMLP


def test_sparse_linear_matches_dense():
    weight = np.array([
        [2.0, 0.0],
        [0.0, 3.0],
        [4.0, 0.0]
    ])

    bias = np.array([0.5, -0.5])

    mask = np.array([
        [1, 0],
        [0, 1],
        [1, 0]
    ])

    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    sparse = SparseLinear(weight, bias, mask)

    expected = x @ (weight * mask) + bias

    np.testing.assert_allclose(
        sparse.forward(x),
        expected,
        rtol=1e-12,
        atol=1e-12
    )

    assert sparse.nnz == 3


def test_sparse_mlp_matches_masked_mlp():
    model = MLP(
        input_dim=2,
        hidden_dim=32,
        output_dim=2,
        seed=42
    )

    rng = np.random.default_rng(123)

    mask1 = rng.random(
        model.layer1.weight.data.shape
    ) > 0.75

    mask2 = rng.random(
        model.layer2.weight.data.shape
    ) > 0.75

    model.layer1.set_mask(mask1)
    model.layer2.set_mask(mask2)

    x = rng.normal(size=(16, 2))

    dense_result = model.forward(
        Tensor(x)
    ).data

    sparse_model = SparseMLP(model)

    sparse_result = sparse_model.forward(x)

    np.testing.assert_allclose(
        sparse_result,
        dense_result,
        rtol=1e-10,
        atol=1e-10
    )

    assert sparse_model.active_weights() == (
        np.count_nonzero(mask1)
        + np.count_nonzero(mask2)
    )


def test_fully_pruned_layer_returns_bias():
    weight = np.ones((3, 2))
    bias = np.array([1.5, -2.0])
    mask = np.zeros((3, 2))

    sparse = SparseLinear(weight, bias, mask)

    x = np.ones((4, 3))

    np.testing.assert_array_equal(
        sparse.forward(x),
        np.tile(bias, (4, 1))
    )

    assert sparse.nnz == 0


def test_invalid_sparse_input_shape():
    sparse = SparseLinear(
        weight=np.ones((3, 2)),
        bias=np.zeros(2),
        mask=np.ones((3, 2))
    )

    with pytest.raises(ValueError):
        sparse.forward(np.ones((4, 4)))
