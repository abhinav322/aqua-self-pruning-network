
import numpy as np

# Step 1: Define our neuron
x = np.array([3.0, 4.0])
w = np.array([0.5, 0.2])
b = 0.1
target = 3.0


# Step 2: Define the loss function
def calculate_loss(weights):
    prediction = np.sum(x * weights) + b
    loss = (prediction - target) ** 2
    return loss


# Step 3: Calculate analytical gradients
prediction = np.sum(x * w) + b

dL_dz = 2 * (prediction - target)

analytical_gradients = dL_dz * x


# Step 4: Calculate numerical gradients
epsilon = 1e-5

numerical_gradients = np.zeros_like(w)

for i in range(len(w)):
    w_plus = w.copy()
    w_minus = w.copy()

    w_plus[i] += epsilon
    w_minus[i] -= epsilon

    loss_plus = calculate_loss(w_plus)
    loss_minus = calculate_loss(w_minus)

    numerical_gradients[i] = (
        loss_plus - loss_minus
    ) / (2 * epsilon)


# Step 5: Compare the results
print("Prediction:", prediction)
print("Loss:", calculate_loss(w))

print("Analytical gradients:", analytical_gradients)
print("Numerical gradients:", numerical_gradients)

print(
    "Gradient check passed:",
    np.allclose(
        analytical_gradients,
        numerical_gradients,
        rtol=1e-5,
        atol=1e-7
    )
)
