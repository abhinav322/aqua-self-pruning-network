
import numpy as np

from tensor import Tensor


# Test 1: Matrix + vector
A = Tensor(
    [[1, 2], [3, 4], [5, 6]],
    requires_grad=True
)

b = Tensor(
    [10, 20],
    requires_grad=True
)

C = A + b
loss = C.sum()
loss.backward()

print("TEST 1: Matrix + vector")
print("Output:\n", C.data)
print("Gradient A:\n", A.grad)
print("Gradient b:", b.grad)

assert np.allclose(A.grad, np.ones((3, 2)))
assert np.allclose(b.grad, [3, 3])


# Test 2: Matrix + scalar
X = Tensor(
    [[1, 2], [3, 4]],
    requires_grad=True
)

s = Tensor(5.0, requires_grad=True)

Y = X + s
loss2 = Y.sum()
loss2.backward()

print("\nTEST 2: Matrix + scalar")
print("Gradient X:\n", X.grad)
print("Gradient scalar:", s.grad)

assert np.allclose(X.grad, np.ones((2, 2)))
assert np.allclose(s.grad, 4.0)


# Test 3: Broadcasting a dimension of size 1
M = Tensor(
    [[1], [2], [3]],
    requires_grad=True
)

N = Tensor(
    [[10, 20, 30, 40]],
    requires_grad=True
)

P = M + N
loss3 = P.sum()
loss3.backward()

print("\nTEST 3: Two-dimensional broadcasting")
print("Output shape:", P.data.shape)
print("Gradient M:\n", M.grad)
print("Gradient N:\n", N.grad)

assert np.allclose(M.grad, [[4], [4], [4]])
assert np.allclose(N.grad, [[3, 3, 3, 3]])

print("\nAll broadcasting tests passed!")
