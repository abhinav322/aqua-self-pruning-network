
import numpy as np
import pytest

from engine.tensor import Tensor


# --------------------------------------------------
# Helper 1: Numerical gradient using finite differences
# --------------------------------------------------

def numerical_gradient(function, values, epsilon=1e-5):
    values = np.asarray(values, dtype=np.float64).copy()

    gradients = np.zeros_like(values)

    # Visit every element, including elements of matrices.
    for index in np.ndindex(values.shape):

        original = values[index]

        # Evaluate at x + epsilon.
        values[index] = original + epsilon
        loss_plus = float(function(values))

        # Evaluate at x - epsilon.
        values[index] = original - epsilon
        loss_minus = float(function(values))

        # Restore the original value.
        values[index] = original

        gradients[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    return gradients


# --------------------------------------------------
# Helper 2: Compare autodiff with numerical gradients
# --------------------------------------------------

def check_binary_operation(operation, a_values, b_values):
    a_values = np.asarray(a_values, dtype=np.float64)
    b_values = np.asarray(b_values, dtype=np.float64)

    # Calculate derivatives using our autodiff engine.
    a = Tensor(a_values, requires_grad=True)
    b = Tensor(b_values, requires_grad=True)

    result = operation(a, b)
    loss = result.sum()

    loss.backward()

    # Independently calculate numerical gradients.
    numeric_a = numerical_gradient(
        lambda values: np.sum(
            operation(values, b_values)
        ),
        a_values
    )

    numeric_b = numerical_gradient(
        lambda values: np.sum(
            operation(a_values, values)
        ),
        b_values
    )

    # Compare both methods.
    np.testing.assert_allclose(
        a.grad,
        numeric_a,
        rtol=1e-5,
        atol=1e-7
    )

    np.testing.assert_allclose(
        b.grad,
        numeric_b,
        rtol=1e-5,
        atol=1e-7
    )


# --------------------------------------------------
# Test 1: Addition
# --------------------------------------------------

def test_addition():
    check_binary_operation(
        lambda a, b: a + b,
        [[1.0, 2.0], [3.0, 4.0]],
        [5.0, 6.0]
    )


# --------------------------------------------------
# Test 2: Subtraction
# --------------------------------------------------

def test_subtraction():
    check_binary_operation(
        lambda a, b: a - b,
        [[1.0, 2.0], [3.0, 4.0]],
        [5.0, 6.0]
    )


# --------------------------------------------------
# Test 3: Multiplication
# --------------------------------------------------

def test_multiplication():
    check_binary_operation(
        lambda a, b: a * b,
        [[1.0, 2.0], [3.0, 4.0]],
        [5.0, 6.0]
    )


# --------------------------------------------------
# Test 4: Division
# --------------------------------------------------

def test_division():
    check_binary_operation(
        lambda a, b: a / b,
        [[10.0, 20.0], [30.0, 40.0]],
        [2.0, 4.0]
    )


# --------------------------------------------------
# Test 5: Matrix multiplication
# --------------------------------------------------

def test_matrix_multiplication():
    check_binary_operation(
        lambda a, b: a @ b,
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]],
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]
    )


# --------------------------------------------------
# Test 6: Shared tensor
# --------------------------------------------------

def test_shared_tensor():
    values = np.array([2.0, 3.0, 4.0])

    x = Tensor(values, requires_grad=True)

    loss = (x * x).sum()
    loss.backward()

    numeric = numerical_gradient(
        lambda v: np.sum(v * v),
        values
    )

    np.testing.assert_allclose(
        x.grad,
        numeric,
        rtol=1e-5,
        atol=1e-7
    )


# --------------------------------------------------
# Test 7: Chained operations
# --------------------------------------------------

def test_chained_operations():
    values = np.array([2.0, 3.0, 4.0])

    x = Tensor(values, requires_grad=True)

    # Combining multiplication, addition and division.
    loss = ((x * x + 3.0) / (x + 1.0)).sum()
    loss.backward()

    numeric = numerical_gradient(
        lambda v: np.sum(
            (v * v + 3.0) / (v + 1.0)
        ),
        values
    )

    np.testing.assert_allclose(
        x.grad,
        numeric,
        rtol=1e-5,
        atol=1e-7
    )
