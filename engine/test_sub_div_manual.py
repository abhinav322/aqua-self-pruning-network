
import numpy as np

from tensor import Tensor


# TEST 1: Scalar subtraction
a = Tensor(10.0, requires_grad=True)
b = Tensor(4.0, requires_grad=True)

c = a - b
c.backward()

print("TEST 1: Subtraction")
print("Output:", c.data)
print("Gradient a:", a.grad)
print("Gradient b:", b.grad)

assert np.allclose(c.data, 6.0)
assert np.allclose(a.grad, 1.0)
assert np.allclose(b.grad, -1.0)


# TEST 2: Scalar division
x = Tensor(12.0, requires_grad=True)
y = Tensor(4.0, requires_grad=True)

z = x / y
z.backward()

print("\nTEST 2: Division")
print("Output:", z.data)
print("Gradient x:", x.grad)
print("Gradient y:", y.grad)

assert np.allclose(z.data, 3.0)
assert np.allclose(x.grad, 0.25)
assert np.allclose(y.grad, -0.75)


# TEST 3: Broadcasting subtraction
A = Tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    requires_grad=True
)

bias = Tensor([10.0, 20.0], requires_grad=True)

loss = (A - bias).sum()
loss.backward()

print("\nTEST 3: Broadcasting subtraction")
print("Gradient A:\n", A.grad)
print("Gradient bias:", bias.grad)

assert np.allclose(A.grad, np.ones((2, 2)))
assert np.allclose(bias.grad, [-2.0, -2.0])


# TEST 4: Broadcasting division
M = Tensor(
    [[10.0, 20.0], [30.0, 40.0]],
    requires_grad=True
)

d = Tensor([2.0, 4.0], requires_grad=True)

loss2 = (M / d).sum()
loss2.backward()

print("\nTEST 4: Broadcasting division")
print("Gradient M:\n", M.grad)
print("Gradient d:", d.grad)

assert np.allclose(
    M.grad,
    [[0.5, 0.25], [0.5, 0.25]]
)

assert np.allclose(
    d.grad,
    [-10.0, -3.75]
)


# TEST 5: Chained operations
p = Tensor(6.0, requires_grad=True)
q = Tensor(2.0, requires_grad=True)

result = (p * q) / q
result.backward()

print("\nTEST 5: Chained operations")
print("Output:", result.data)
print("Gradient p:", p.grad)
print("Gradient q:", q.grad)

assert np.allclose(result.data, 6.0)
assert np.allclose(p.grad, 1.0)
assert np.allclose(q.grad, 0.0, atol=1e-10)

print("\nAll five subtraction/division tests passed!")
