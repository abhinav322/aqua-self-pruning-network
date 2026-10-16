
import numpy as np

from engine.tensor import Tensor


def test_masked_forward_is_exactly_zero():
    weights = Tensor(
        [0.8, 0.3, -0.6, 0.1],
        requires_grad=True
    )

    mask = Tensor([1.0, 0.0, 1.0, 0.0])

    effective = weights * mask

    np.testing.assert_array_equal(
        effective.data,
        [0.8, 0.0, -0.6, 0.0]
    )


def test_masked_weight_gradient_is_zero():
    weights = Tensor(
        [0.8, 0.3, -0.6, 0.1],
        requires_grad=True
    )

    mask = Tensor([1.0, 0.0, 1.0, 0.0])

    effective = weights * mask

    loss = (effective * effective).sum()
    loss.backward()

    np.testing.assert_allclose(
        weights.grad,
        [1.6, 0.0, -1.2, 0.0]
    )


def test_effective_gradient_differs_from_weight_gradient():
    weight = Tensor(0.8, requires_grad=True)
    mask = Tensor(0.0)

    effective = weight * mask

    loss = (effective - 2.0) * (effective - 2.0)

    loss.backward()

    # Gradient with respect to effective weight.
    np.testing.assert_allclose(
        effective.grad,
        -4.0
    )

    # Gradient with respect to stored weight.
    np.testing.assert_allclose(
        weight.grad,
        0.0
    )


def test_masked_gradient_matches_finite_difference():
    weights = np.array([0.8, 0.3, -0.6, 0.1])
    mask_values = np.array([1.0, 0.0, 1.0, 0.0])

    tensor_weights = Tensor(
        weights,
        requires_grad=True
    )

    mask = Tensor(mask_values)

    effective = tensor_weights * mask

    loss = (
        (effective - 2.0)
        * (effective - 2.0)
    ).sum()

    loss.backward()

    epsilon = 1e-5
    numerical = np.zeros_like(weights)

    def loss_function(w):
        effective_w = w * mask_values
        return np.sum((effective_w - 2.0) ** 2)

    for i in range(len(weights)):
        plus = weights.copy()
        minus = weights.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        numerical[i] = (
            loss_function(plus)
            - loss_function(minus)
        ) / (2 * epsilon)

    np.testing.assert_allclose(
        tensor_weights.grad,
        numerical,
        rtol=1e-5,
        atol=1e-7
    )


def test_changing_mask_between_training_steps():
    weight = Tensor(0.8, requires_grad=True)

    # Step 1: Connection is active.
    active_mask = Tensor(1.0)

    effective1 = weight * active_mask
    loss1 = (effective1 - 2.0) * (effective1 - 2.0)

    loss1.backward()

    np.testing.assert_allclose(weight.grad, -2.4)

    # Reset gradients before the next step.
    weight.zero_grad()

    # Step 2: Connection is pruned.
    pruned_mask = Tensor(0.0)

    # Build a NEW forward computation graph.
    effective2 = weight * pruned_mask
    loss2 = (effective2 - 2.0) * (effective2 - 2.0)

    loss2.backward()

    np.testing.assert_allclose(weight.grad, 0.0)

    # Step 3: Connection is reactivated.
    weight.zero_grad()

    effective3 = weight * Tensor(1.0)
    loss3 = (effective3 - 2.0) * (effective3 - 2.0)

    loss3.backward()

    np.testing.assert_allclose(weight.grad, -2.4)
