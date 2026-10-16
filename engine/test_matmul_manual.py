
import numpy as np

from tensor import Tensor


# TEST 1: Forward matrix multiplication
A = Tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    requires_grad=True
)

B = Tensor(
    [[5.0, 6.0], [7.0, 8.0]],
    requires_grad=True
)

C = A @ B
loss = C.sum()
loss.backward()

print("TEST 1: Forward matrix multiplication")
print("Output:\n", C.data)

assert np.allclose(
    C.data,
    [[19.0, 22.0], [43.0, 50.0]]
)


# TEST 2: Gradients of both matrices
print("\nTEST 2: Matrix multiplication gradients")
print("Gradient A:\n", A.grad)
print("Gradient B:\n", B.grad)

expected_A = np.array([
    [11.0, 15.0],
    [11.0, 15.0]
])

expected_B = np.array([
    [4.0, 4.0],
    [6.0, 6.0]
])

assert np.allclose(A.grad, expected_A)
assert np.allclose(B.grad, expected_B)


# TEST 3: Different matrix dimensions
X = Tensor(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    requires_grad=True
)

W = Tensor(
    [[1.0, 2.0],
     [3.0, 4.0],
     [5.0, 6.0]],
    requires_grad=True
)

Y = X @ W
loss2 = Y.sum()
loss2.backward()

print("\nTEST 3: Non-square matrix multiplication")
print("Output shape:", Y.data.shape)
print("Gradient X shape:", X.grad.shape)
print("Gradient W shape:", W.grad.shape)

assert Y.data.shape == (2, 2)
assert X.grad.shape == (2, 3)
assert W.grad.shape == (3, 2)


# TEST 4: Compare with NumPy calculations
expected_grad_X = (
    np.ones_like(Y.data) @ W.data.T
)

expected_grad_W = (
    X.data.T @ np.ones_like(Y.data)
)

assert np.allclose(X.grad, expected_grad_X)
assert np.allclose(W.grad, expected_grad_W)

print("\nAll four matrix multiplication tests passed!")
