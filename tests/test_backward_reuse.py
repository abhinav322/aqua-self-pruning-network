
import numpy as np

from engine.tensor import Tensor


def test_repeated_backward_accumulates_leaf_gradient():
    x = Tensor(3.0, requires_grad=True)

    y = x * x

    # First backward pass.
    y.backward()

    np.testing.assert_allclose(x.grad, 6.0)

    # Second backward pass on the same graph.
    y.backward()

    # Two contributions of 6.
    np.testing.assert_allclose(x.grad, 12.0)


def test_zero_grad_resets_gradient():
    x = Tensor(3.0, requires_grad=True)

    y = x * x
    y.backward()

    np.testing.assert_allclose(x.grad, 6.0)

    # Reset the stored gradient.
    x.zero_grad()

    np.testing.assert_allclose(x.grad, 0.0)

    # Backpropagate again.
    y.backward()

    np.testing.assert_allclose(x.grad, 6.0)


def test_shared_graph_backward_twice():
    x = Tensor(2.0, requires_grad=True)

    a = x * x
    b = a + a

    b.backward()

    # b = 2x², derivative = 4x = 8.
    np.testing.assert_allclose(x.grad, 8.0)

    b.backward()

    np.testing.assert_allclose(x.grad, 16.0)


def test_backward_on_leaf():
    x = Tensor(5.0, requires_grad=True)

    x.backward()
    np.testing.assert_allclose(x.grad, 1.0)

    x.backward()
    np.testing.assert_allclose(x.grad, 2.0)
