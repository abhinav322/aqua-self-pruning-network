
from tensor import Tensor


# Create a scalar tensor
a = Tensor(3.0, requires_grad=True)

# Create a vector tensor
b = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Create a matrix tensor
c = Tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    requires_grad=False
)


print("Scalar tensor:")
print(a)

print("\nVector tensor:")
print(b)

print("\nMatrix tensor:")
print(c)

print("\nScalar shape:", a.data.shape)
print("Vector shape:", b.data.shape)
print("Matrix shape:", c.data.shape)

print("\nInitial gradient:", a.grad)
print("Parents:", a.parents)
