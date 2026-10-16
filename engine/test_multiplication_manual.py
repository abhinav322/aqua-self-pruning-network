
import numpy as np

from tensor import Tensor


# TEST 1: Basic scalar multiplication
a = Tensor(3.0, requires_grad=True)
b = Tensor(4.0, requires_grad=True)

c = a * b
c.backward()

print("TEST 1: Scalar multiplication")
print("Output:", c.data)
print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)

assert np.allclose(a.grad, 4.0)
assert np.allclose(b.grad, 3.0)


# TEST 2: Element-wise vector multiplication
x = Tensor([2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0], requires_grad=True)

z = x * y
loss = z.sum()
loss.backward()

print("\nTEST 2: Element-wise multiplication")
print("Output:", z.data)
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)

assert np.allclose(x.grad, [4.0, 5.0])
assert np.allclose(y.grad, [2.0, 3.0])


# TEST 3: Broadcasting multiplication
A = Tensor(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    requires_grad=True
)

b = Tensor([10.0, 20.0], requires_grad=True)

C = A * b
loss2 = C.sum()
loss2.backward()

print("\nTEST 3: Broadcasting multiplication")
print("Gradient of A:\n", A.grad)
print("Gradient of b:", b.grad)

assert np.allclose(
    A.grad,
    [[10, 20], [10, 20], [10, 20]]
)

assert np.allclose(b.grad, [9, 12])


# TEST 4: Shared tensor in multiplication
p = Tensor(3.0, requires_grad=True)

q = p * p
q.backward()

print("\nTEST 4: Shared tensor")
print("Output:", q.data)
print("Gradient of p:", p.grad)

assert np.allclose(p.grad, 6.0)


print("\nAll four multiplication tests passed!")
