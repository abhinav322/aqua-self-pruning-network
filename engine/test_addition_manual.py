
import numpy as np

from tensor import Tensor


# Test 1: Basic addition
a = Tensor(3.0, requires_grad=True)
b = Tensor(4.0, requires_grad=True)

c = a + b
c.backward()

print("TEST 1: Basic addition")
print("Output:", c.data)
print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)

assert np.allclose(a.grad, 1.0)
assert np.allclose(b.grad, 1.0)


# Test 2: Same tensor used twice
x = Tensor(5.0, requires_grad=True)

y = x + x
y.backward()

print("\nTEST 2: Gradient accumulation")
print("Output:", y.data)
print("Gradient of x:", x.grad)

assert np.allclose(x.grad, 2.0)


# Test 3: Chained operations
p = Tensor(2.0, requires_grad=True)
q = Tensor(3.0, requires_grad=True)

r = p + q
s = r + 4.0

s.backward()

print("\nTEST 3: Chained additions")
print("Output:", s.data)
print("Gradient of p:", p.grad)
print("Gradient of q:", q.grad)

assert np.allclose(p.grad, 1.0)
assert np.allclose(q.grad, 1.0)

print("\nAll three addition tests passed!")
