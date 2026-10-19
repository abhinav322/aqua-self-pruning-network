
import numpy as np

from engine.tensor import Tensor


def numerical_gradient(function, values, epsilon=1e-5):
    values = np.asarray(values, dtype=np.float64).copy()
    gradients = np.zeros_like(values)

    for index in np.ndindex(values.shape):
        original = values[index]

        values[index] = original + epsilon
        loss_plus = function(values)

        values[index] = original - epsilon
        loss_minus = function(values)

        values[index] = original

        gradients[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    return gradients


def test_mean_forward():
    x = Tensor([2.0, 4.0, 6.0, 8.0])
    result = x.mean()

    np.testing.assert_allclose(result.data, 5.0)


def test_mean_gradient():
    x = Tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)

    y = x.mean()
    y.backward()

    expected = np.array([0.25, 0.25, 0.25, 0.25])

    np.testing.assert_allclose(x.grad, expected)

    numeric = numerical_gradient(
        lambda values: np.mean(values),
        [2.0, 4.0, 6.0, 8.0]
    )

    np.testing.assert_allclose(
        x.grad, numeric, rtol=1e-5, atol=1e-7
    )


def test_relu_forward():
    x = Tensor([-3.0, -1.0, 0.0, 2.0, 5.0])

    y = x.relu()

    np.testing.assert_allclose(
        y.data,
        [0.0, 0.0, 0.0, 2.0, 5.0]
    )


def test_relu_gradient():
    # Avoid exactly zero during numerical differentiation:
    # ReLU is not differentiable at that point.
    values = np.array([-3.0, -1.0, 2.0, 5.0])

    x = Tensor(values, requires_grad=True)

    loss = x.relu().sum()
    loss.backward()

    expected = np.array([0.0, 0.0, 1.0, 1.0])

    np.testing.assert_allclose(x.grad, expected)

    numeric = numerical_gradient(
        lambda v: np.maximum(0.0, v).sum(),
        values
    )

    np.testing.assert_allclose(
        x.grad, numeric, rtol=1e-5, atol=1e-7
    )


def test_relu_at_zero_convention():
    x = Tensor(0.0, requires_grad=True)

    y = x.relu()
    y.backward()

    # Our chosen subgradient convention.
    np.testing.assert_allclose(x.grad, 0.0)


def test_sigmoid_forward():
    x = Tensor([-2.0, 0.0, 2.0])

    y = x.sigmoid()

    expected = 1.0 / (
        1.0 + np.exp(-np.array([-2.0, 0.0, 2.0]))
    )

    np.testing.assert_allclose(
        y.data,
        expected,
        rtol=1e-12
    )


def test_sigmoid_gradient():
    values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    x = Tensor(values, requires_grad=True)

    loss = x.sigmoid().sum()
    loss.backward()

    numeric = numerical_gradient(
        lambda v: np.sum(1.0 / (1.0 + np.exp(-v))),
        values
    )

    np.testing.assert_allclose(
        x.grad,
        numeric,
        rtol=1e-5,
        atol=1e-7
    )


def test_sigmoid_extreme_values():
    x = Tensor(
        [-1000.0, 0.0, 1000.0],
        requires_grad=True
    )

    y = x.sigmoid()

    assert np.all(np.isfinite(y.data))

    np.testing.assert_allclose(
        y.data,
        [0.0, 0.5, 1.0],
        atol=1e-12
    )

    y.sum().backward()

    assert np.all(np.isfinite(x.grad))

def test_sum_gradient():
    values = np.array([
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0]
    ])

    x = Tensor(values, requires_grad=True)

    loss = x.sum()
    loss.backward()

    expected = np.ones_like(values)

    np.testing.assert_allclose(x.grad, expected)

    numeric = numerical_gradient(
        lambda v: np.sum(v),
        values
    )

    np.testing.assert_allclose(
        x.grad,
        numeric,
        rtol=1e-5,
        atol=1e-7
    )
