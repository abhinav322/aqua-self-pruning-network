
import numpy as np

from engine.tensor import Tensor
from nn.layers import Linear
from nn.optimizers import Adam


def make_layer():
    rng = np.random.default_rng(42)

    layer = Linear(
        in_features=2,
        out_features=2,
        rng=rng
    )

    layer.weight.data = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    layer.bias.data = np.array([0.0, 0.0])

    return layer


def test_masked_forward():
    layer = make_layer()

    layer.set_mask([
        [1, 0],
        [0, 1]
    ])

    x = Tensor([[2.0, 3.0]])

    output = layer.forward(x)

    # Effective weights:
    # [[1, 0],
    #  [0, 4]]
    # Input [2, 3] gives [2, 12].
    np.testing.assert_allclose(
        output.data,
        [[2.0, 12.0]]
    )


def test_masked_backward():
    layer = make_layer()

    layer.set_mask([
        [1, 0],
        [0, 1]
    ])

    x = Tensor([[2.0, 3.0]])

    loss = layer.forward(x).sum()
    loss.backward()

    np.testing.assert_allclose(
        layer.weight.grad,
        [[2.0, 0.0],
         [0.0, 3.0]]
    )


def test_layer_sparsity():
    layer = make_layer()

    layer.set_mask([
        [1, 0],
        [0, 1]
    ])

    assert layer.sparsity() == 0.5


def test_optimizer_mask_sync():
    layer = make_layer()

    optimizer = Adam(
        layer.parameters(),
        learning_rate=0.01
    )

    new_mask = np.array([
        [1, 0],
        [0, 1]
    ])

    layer.set_mask(new_mask)

    # Weight is index 0 in layer.parameters().
    optimizer.set_mask(0, new_mask)

    np.testing.assert_array_equal(
        layer.mask,
        optimizer.masks[0]
    )

    np.testing.assert_array_equal(
        layer.weight.data[~layer.mask],
        0.0
    )


def test_invalid_mask():
    layer = make_layer()

    try:
        layer.set_mask([[1, 2], [0, 1]])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected invalid mask to be rejected"
        )
