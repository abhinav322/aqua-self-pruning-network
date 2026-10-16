
import numpy as np
import pytest

from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy


def numerical_gradient(function, values, epsilon=1e-5):
    values = np.asarray(values, dtype=np.float64).copy()
    gradient = np.zeros_like(values)

    for index in np.ndindex(values.shape):
        original = values[index]

        values[index] = original + epsilon
        plus = function(values)

        values[index] = original - epsilon
        minus = function(values)

        values[index] = original

        gradient[index] = (plus - minus) / (2 * epsilon)

    return gradient


def test_softmax_loss_forward():
    logits = Tensor(
        [[2.0, 1.0, 0.1]],
        requires_grad=True
    )

    targets = np.array([0])

    loss = softmax_cross_entropy(logits, targets)

    expected = (
        np.log(np.exp(2.0) + np.exp(1.0) + np.exp(0.1))
        - 2.0
    )

    np.testing.assert_allclose(loss.data, expected)


def test_softmax_gradient_check():
    values = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.0, -1.0]
    ])

    targets = np.array([0, 1])

    logits = Tensor(values, requires_grad=True)

    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    numeric = numerical_gradient(
        lambda v: float(
            softmax_cross_entropy(
                Tensor(v),
                targets
            ).data
        ),
        values
    )

    np.testing.assert_allclose(
        logits.grad,
        numeric,
        rtol=1e-5,
        atol=1e-7
    )


def test_softmax_extreme_logits():
    logits = Tensor(
        [
            [1000.0, 999.0, 998.0],
            [-1000.0, -999.0, -998.0]
        ],
        requires_grad=True
    )

    targets = np.array([0, 2])

    loss = softmax_cross_entropy(logits, targets)

    assert np.isfinite(loss.data)

    loss.backward()

    assert np.all(np.isfinite(logits.grad))


def test_softmax_batch_size_one():
    logits = Tensor(
        [[1.0, 2.0, 3.0]],
        requires_grad=True
    )

    loss = softmax_cross_entropy(
        logits,
        np.array([2])
    )

    loss.backward()

    assert logits.grad.shape == (1, 3)

    # Softmax cross-entropy logit gradients sum to zero.
    np.testing.assert_allclose(
        logits.grad.sum(axis=1),
        [0.0],
        atol=1e-12
    )


def test_softmax_invalid_targets():
    logits = Tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        softmax_cross_entropy(
            logits,
            np.array([5])
        )


def test_softmax_repeated_backward():
    logits = Tensor(
        [[2.0, 1.0, 0.1]],
        requires_grad=True
    )

    loss = softmax_cross_entropy(
        logits,
        np.array([0])
    )

    loss.backward()
    first_gradient = logits.grad.copy()

    loss.backward()

    np.testing.assert_allclose(
        logits.grad,
        2.0 * first_gradient
    )
