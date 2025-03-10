from typing import Self, Optional

import numpy as np


def hard_lim(v: int):
    return 1 if v >= 0 else 0


class Perceptron:
    MAX_EPOCHS = 100

    def __init__(self, inputs: np.array, targets: np.array, learning_rate=0.1):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                "inputs and targets must have same number of rows")
        if targets.ndim != 1:
            raise ValueError("targets must have only 1 dimension")

        self.inputs = inputs
        self.targets = targets
        self.weights = np.zeros(inputs[0].shape)
        self.learning_rate = learning_rate

    def train(self) -> Self:
        """
        The training procedure, following the perceptron learning rule to modify the weights.
        Returns the trained perceptron (which is itself).
        """
        for epoch in range(self.MAX_EPOCHS):
            print("Epoch ", epoch)

            is_error = False
            for i, input_vector in enumerate(self.inputs):
                actual = hard_lim(self.weights.T.dot(input_vector))
                target = self.targets[i]
                error = target - actual
                self.weights += self.learning_rate * error * input_vector
                if error != 0:
                    is_error = True
                print(f"{input_vector=}{error=}")

            if not is_error:
                print(f"\nWinning weights: {[float(w) for w in self.weights]}")
                print("Error removed in all examples!")
                break

        print(f"\nTraining complete or {self.MAX_EPOCHS} epochs done.\n")

        return self

    def predict(self, inputs: Optional[np.array] = None) -> np.array:
        test_inputs = inputs if inputs else self.inputs
        output = np.zeros(test_inputs.shape[0])
        for i, input_vector in enumerate(test_inputs):
            actual = hard_lim(self.weights.T.dot(input_vector))
            output[i] = actual

        return output
