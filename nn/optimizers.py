
import numpy as np


class Adam:
    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.parameters = list(parameters)

        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Betas must be in [0, 1)")

        if learning_rate <= 0 or epsilon <= 0:
            raise ValueError("Learning rate and epsilon must be positive")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0

        # First and second moment arrays for each parameter.
        self.m = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

        self.v = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

        # Track the number of Adam updates received
        # by each individual parameter element.
        self.step_counts = [
            np.zeros_like(p.data, dtype=np.int64)
            for p in self.parameters
        ]

        # True means that the connection is active.
        self.masks = [
            np.ones_like(p.data, dtype=bool)
            for p in self.parameters
        ]

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def set_mask(self, parameter_index, mask):
        new_mask = np.asarray(mask)

        parameter = self.parameters[parameter_index]

        if new_mask.shape != parameter.data.shape:
            raise ValueError("Mask and parameter shapes must match")

        if not np.all(
            (new_mask == 0) | (new_mask == 1)
        ):
            raise ValueError("Mask must contain only 0 or 1")

        new_mask = new_mask.astype(bool)

        old_mask = self.masks[parameter_index]

        # Identify connections whose active status changed.
        changed = old_mask != new_mask

        # Reset the Adam update age whenever a connection
        # is pruned or reactivated.
        self.step_counts[parameter_index] = np.where(
            changed,
            0,
            self.step_counts[parameter_index]
        )

        # Reset newly pruned or revived connections.
        self.m[parameter_index] = np.asarray(
            np.where(changed, 0.0, self.m[parameter_index]),
            dtype=np.float64
        )

        self.v[parameter_index] = np.asarray(
            np.where(changed, 0.0, self.v[parameter_index]),
            dtype=np.float64
        )

        parameter.data = np.asarray(
            np.where(changed, 0.0, parameter.data),
            dtype=np.float64
        )

        # Guarantee masked parameters remain zero.
        parameter.data = np.asarray(
            np.where(new_mask, parameter.data, 0.0),
            dtype=np.float64
        )

        self.masks[parameter_index] = new_mask.copy()

    def step(self):
        self.t += 1

        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue

            if parameter.grad.shape != parameter.data.shape:
                raise ValueError("Gradient shape mismatch")

            mask = self.masks[i]

            gradient = np.where(
                mask,
                parameter.grad,
                0.0
            )

            if not np.all(np.isfinite(gradient)):
                raise ValueError("Non-finite gradient detected")

            # Update first moment.
            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1) * gradient
            )

            # Update second moment.
            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2) * gradient**2
            )

            # Keep inactive optimizer states at zero.
            self.m[i] = np.asarray(
                np.where(mask, self.m[i], 0.0),
                dtype=np.float64
            )

            self.v[i] = np.asarray(
                np.where(mask, self.v[i], 0.0),
                dtype=np.float64
            )

            # Advance update ages only for active connections.
            self.step_counts[i] += mask.astype(np.int64)

            # Each active connection uses its own update count
            # for Adam bias correction.
            local_t = self.step_counts[i]

            # Avoid dividing by zero for inactive connections.
            safe_t = np.maximum(local_t, 1)

            # Bias correction using per-connection timesteps.
            m_hat = self.m[i] / (
                1.0 - np.power(self.beta1, safe_t)
            )

            v_hat = self.v[i] / (
                1.0 - np.power(self.beta2, safe_t)
            )

            # Adam parameter update.
            update = (
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )

            parameter.data = np.asarray(
                np.where(
                    mask,
                    parameter.data - update,
                    0.0
                ),
                dtype=np.float64
            )