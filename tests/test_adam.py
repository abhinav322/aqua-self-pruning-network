
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


def test_revived_connection_uses_fresh_bias_correction():
    """
    A revived connection's first Adam update must match
    the first update of a newly initialized parameter.
    """
    import numpy as np

    from engine.tensor import Tensor
    from nn.optimizers import Adam

    learning_rate = 0.1

    # Create an optimizer with two connections.
    weight = Tensor([1.0, 2.0], requires_grad=True)

    optimizer = Adam(
        [weight],
        learning_rate=learning_rate
    )

    # Build up a long global training history.
    for _ in range(100):
        weight.grad = np.array([1.0, 1.0])
        optimizer.step()

    # Prune the second connection.
    optimizer.set_mask(0, [1, 0])

    # Continue training while that connection is pruned.
    for _ in range(50):
        weight.grad = np.array([1.0, 0.0])
        optimizer.step()

    # Reactivate the second connection.
    optimizer.set_mask(0, [1, 1])

    # Its history must be completely reset.
    assert optimizer.step_counts[0][1] == 0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0
    assert weight.data[1] == 0.0

    # Perform its first new update after revival.
    weight.grad = np.array([1.0, 1.0])
    optimizer.step()

    revived_value = weight.data[1]

    # Build a completely fresh Adam optimizer.
    fresh_weight = Tensor(0.0, requires_grad=True)

    fresh_optimizer = Adam(
        [fresh_weight],
        learning_rate=learning_rate
    )

    fresh_weight.grad = np.array(1.0)
    fresh_optimizer.step()

    # Both first updates must be numerically identical.
    np.testing.assert_allclose(
        revived_value,
        fresh_weight.data,
        rtol=1e-12,
        atol=1e-12
    )

    assert optimizer.step_counts[0][1] == 1
    assert optimizer.step_counts[0][0] == 151