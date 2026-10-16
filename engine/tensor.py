
import numpy as np


def unbroadcast(grad, target_shape):
    """
    Reduce a broadcasted gradient back to
    the original input tensor's shape.
    """
    grad = np.asarray(grad, dtype=np.float64)

    # Remove extra leading dimensions by summing.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum over axes where the original dimension was 1.
    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad



class Tensor:
    def __init__(
        self,
        data,
        requires_grad=False,
        parents=(),
        operation=None
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = tuple(parents)
        self.operation = operation

        # Function used to propagate gradients backward.
        self._backward = lambda: None

    def __repr__(self):
        return (
            f"Tensor(data={self.data}, "
            f"requires_grad={self.requires_grad})"
        )

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Forward pass: calculate the output.
        result = Tensor(
            self.data + other.data,
            requires_grad=(
                self.requires_grad or other.requires_grad
            ),
            parents=(self, other),
            operation="add"
        )

        # Backward rule for scalar addition.
        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += unbroadcast(
                    result.grad,
                    self.data.shape
                )

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                other.grad += unbroadcast(
                    result.grad,
                    other.data.shape
                )

        result._backward = _backward

        return result

    
    def zero_grad(self):
        """
        Reset this tensor's stored gradient to zero.

        Useful before computing gradients for a new
        training batch.
        """
        self.grad = np.zeros_like(self.data)

    
    def backward(self):
        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output."
            )

        if not self.requires_grad:
            raise ValueError(
                "Cannot backpropagate through a tensor "
                "that does not require gradients."
            )

        topo = []
        visited = set()

        # Build the computation graph in topological order.
        def build_graph(tensor):
            if id(tensor) in visited:
                return

            visited.add(id(tensor))

            for parent in tensor.parents:
                build_graph(parent)

            topo.append(tensor)

        build_graph(self)

        # Clear intermediate gradients.
        # Leaf gradients are deliberately preserved.
        for tensor in topo:
            if tensor.parents:
                tensor.grad = None

        # Seed the output gradient.
        if self.parents:
            self.grad = np.ones_like(self.data)
        else:
            # Special case: backward called on a leaf.
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            self.grad += np.ones_like(self.data)

        # Propagate gradients backward.
        for tensor in reversed(topo):
            tensor._backward()


    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Forward pass: element-wise multiplication.
        result = Tensor(
            self.data * other.data,
            requires_grad=(
                self.requires_grad or other.requires_grad
            ),
            parents=(self, other),
            operation="multiply"
        )

        # Backward pass: apply the product rule.
        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                gradient_self = result.grad * other.data

                self.grad += unbroadcast(
                    gradient_self,
                    self.data.shape
                )

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                gradient_other = result.grad * self.data

                other.grad += unbroadcast(
                    gradient_other,
                    other.data.shape
                )

        result._backward = _backward

        return result


    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data - other.data,
            requires_grad=(
                self.requires_grad or other.requires_grad
            ),
            parents=(self, other),
            operation="subtract"
        )

        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += unbroadcast(
                    result.grad,
                    self.data.shape
                )

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                other.grad += unbroadcast(
                    -result.grad,
                    other.data.shape
                )

        result._backward = _backward

        return result

    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        if np.any(other.data == 0):
            raise ZeroDivisionError(
                "Division by zero is not supported."
            )

        result = Tensor(
            self.data / other.data,
            requires_grad=(
                self.requires_grad or other.requires_grad
            ),
            parents=(self, other),
            operation="divide"
        )

        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                gradient_self = result.grad / other.data

                self.grad += unbroadcast(
                    gradient_self,
                    self.data.shape
                )

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                gradient_other = (
                    -result.grad * self.data
                    / (other.data ** 2)
                )

                other.grad += unbroadcast(
                    gradient_other,
                    other.data.shape
                )

        result._backward = _backward

        return result

    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # This initial implementation supports 2D matrices.
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(
                "Matrix multiplication currently requires "
                "two 2D tensors."
            )

        # Forward pass
        result = Tensor(
            self.data @ other.data,
            requires_grad=(
                self.requires_grad or other.requires_grad
            ),
            parents=(self, other),
            operation="matmul"
        )

        def _backward():
            if result.grad is None:
                return

            # Gradient with respect to the first matrix
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += (
                    result.grad @ other.data.T
                )

            # Gradient with respect to the second matrix
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                other.grad += (
                    self.data.T @ result.grad
                )

        result._backward = _backward

        return result

    
    def sum(self):
        result = Tensor(
            np.sum(self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            operation="sum"
        )

        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += (
                    np.ones_like(self.data) * result.grad
                )

        result._backward = _backward

        return result

    
    def mean(self):
        result = Tensor(
            np.mean(self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            operation="mean"
        )

        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                gradient = (
                    np.ones_like(self.data)
                    * result.grad
                    / self.data.size
                )

                self.grad += gradient

        result._backward = _backward

        return result

    
    def relu(self):
        # Forward pass
        result = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            operation="relu"
        )

        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                # ReLU derivative is 1 for positive inputs
                # and 0 for zero or negative inputs.
                derivative = (self.data > 0).astype(np.float64)

                self.grad += result.grad * derivative

        result._backward = _backward

        return result

    
    def sigmoid(self):
        # Numerically stable sigmoid implementation.
        positive = self.data >= 0

        output = np.empty_like(self.data)

        # For nonnegative inputs.
        output[positive] = (
            1.0 / (1.0 + np.exp(-self.data[positive]))
        )

        # For negative inputs, avoid exp(large positive).
        exp_x = np.exp(self.data[~positive])

        output[~positive] = exp_x / (1.0 + exp_x)

        result = Tensor(
            output,
            requires_grad=self.requires_grad,
            parents=(self,),
            operation="sigmoid"
        )

        def _backward():
            if result.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                derivative = output * (1.0 - output)

                self.grad += result.grad * derivative

        result._backward = _backward

        return result


