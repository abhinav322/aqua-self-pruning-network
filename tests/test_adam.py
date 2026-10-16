
import numpy as np

from engine.tensor import Tensor
from nn.optimizers import Adam


def test_first_adam_update():
    weight = Tensor(1.0, requires_grad=True)

    optimizer = Adam([weight], learning_rate=0.1)

    weight.grad = np.array(2.0)

    optimizer.step()

    # First Adam update is approximately -learning_rate
    # for a positive gradient of this magnitude.
    np.testing.assert_allclose(
        weight.data,
        0.9,
        atol=1e-7
    )


def test_optimizer_zero_grad():
    weight = Tensor([1.0, 2.0], requires_grad=True)

    optimizer = Adam([weight])

    weight.grad = np.array([3.0, 4.0])

    optimizer.zero_grad()

    np.testing.assert_array_equal(
        weight.grad,
        [0.0, 0.0]
    )


def test_masked_weight_remains_zero():
    weight = Tensor(
        [1.0, 2.0],
        requires_grad=True
    )

    optimizer = Adam([weight], learning_rate=0.1)

    # Build momentum using an active update.
    weight.grad = np.array([1.0, 1.0])
    optimizer.step()

    # Prune the second connection.
    optimizer.set_mask(0, [1, 0])

    for _ in range(5):
        weight.grad = np.array([1.0, 0.0])
        optimizer.step()

        assert weight.data[1] == 0.0
        assert optimizer.m[0][1] == 0.0
        assert optimizer.v[0][1] == 0.0


def test_revived_connection_has_reset_state():
    weight = Tensor(
        [1.0, 2.0],
        requires_grad=True
    )

    optimizer = Adam([weight], learning_rate=0.1)

    # Establish nonzero optimizer state.
    weight.grad = np.array([1.0, 1.0])
    optimizer.step()

    # Prune second connection.
    optimizer.set_mask(0, [1, 0])

    # Revive it.
    optimizer.set_mask(0, [1, 1])

    assert weight.data[1] == 0.0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0

    # Fresh gradient should produce a finite update.
    weight.grad = np.array([0.0, -1.0])
    optimizer.step()

    assert np.isfinite(weight.data[1])
    assert weight.data[1] > 0.0


def test_mask_shape_validation():
    weight = Tensor(
        [1.0, 2.0],
        requires_grad=True
    )

    optimizer = Adam([weight])

    try:
        optimizer.set_mask(0, [1, 0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected mask shape validation to fail"
        )
