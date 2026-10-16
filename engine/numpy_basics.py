
import numpy as np

# Create two input values
x = np.array([3.0, 4.0])

# Create two weights
w = np.array([0.5, 0.2])

# Define a bias
b = 0.1

# Multiply inputs by weights and add the bias
z = np.sum(x * w) + b

print("Inputs:", x)
print("Weights:", w)
print("Weighted products:", x * w)
print("Neuron output:", z)
