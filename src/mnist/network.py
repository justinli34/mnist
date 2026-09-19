import itertools
import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32, Int64
from scipy.special import softmax

from mnist import SEED

WeightMatrix = Float32[np.ndarray, "out_features in_features"]
BiasVector = Float32[np.ndarray, "out_features"]


@dataclass
class Gradients:
    weights: list[WeightMatrix]
    biases: list[BiasVector]


class Network:
    weights: list[WeightMatrix]
    biases: list[BiasVector]

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        depth: int,
        width: int,
    ):
        """Initializes a uniform-width MLP with the given parameters."""
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.depth = depth
        self.width = width

        self._rng = np.random.default_rng(SEED)

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self) -> None:
        """Initializes weights using He initialization and biases to zero."""
        self.weights = []
        self.biases = []

        layer_sizes = [
            self.input_dimension,
            *([self.width] * self.depth),
            self.output_dimension,
        ]

        for input_size, output_size in itertools.pairwise(layer_sizes):
            weight_matrix = self._rng.normal(
                loc=0.0,
                scale=math.sqrt(2 / input_size),
                size=(output_size, input_size),
            ).astype(np.float32)

            bias_vector = np.zeros(output_size, dtype=np.float32)

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def learn(
        self,
        X: Float32[np.ndarray, "num_examples input_dimension"],
        y: Int64[np.ndarray, "num_examples"],  # noqa: F821
        learning_rate: float,
        num_epochs: int,
        mini_batch_size: int,
    ) -> None:
        """Trains the network."""
        for _ in range(num_epochs):
            indices = self._rng.permutation(len(X))

            for start in range(0, len(X), mini_batch_size):
                batch_indices = indices[start : start + mini_batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                activations, logits = self.forward(X_batch)
                gradients = self.backward(X_batch, y_batch, activations, logits)
                self.update(gradients, learning_rate)

    def forward(
        self,
        X: Float32[np.ndarray, "batch_size input_dimension"],
    ) -> tuple[
        list[Float32[np.ndarray, "batch_size width"]],
        Float32[np.ndarray, "batch_size output_dimension"],
    ]:
        """Performs a forward pass using ReLU activation. Returns (activations, logits)."""
        activations = []

        activation = X
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activation = np.maximum(0, activation @ weight.T + bias)
            activations.append(activation)

        logits = activation @ self.weights[-1].T + self.biases[-1]

        return activations, logits

    def backward(
        self,
        X: Float32[np.ndarray, "batch_size input_dimension"],
        y: Int64[np.ndarray, "batch_size"],  # noqa: F821
        activations: list[Float32[np.ndarray, "batch_size width"]],
        logits: Float32[np.ndarray, "batch_size output_dimension"],
    ) -> Gradients:
        """Computes and returns the gradients using softmax and cross-entropy."""
        probabilities = softmax(logits, axis=1)
        grad_logits = probabilities.copy()
        grad_logits[np.arange(len(y)), y] -= 1
        grad_logits /= len(y)

        grad_weights = [np.empty_like(weight) for weight in self.weights]
        grad_biases = [np.empty_like(bias) for bias in self.biases]

        grad = grad_logits

        for i in reversed(range(len(self.weights))):
            previous_activation = X if i == 0 else activations[i - 1]

            grad_weights[i] = grad.T @ previous_activation
            grad_biases[i] = np.sum(grad, axis=0)

            if i > 0:
                grad = grad @ self.weights[i]
                grad *= previous_activation > 0

        return Gradients(
            weights=grad_weights,
            biases=grad_biases,
        )

    def update(self, gradients: Gradients, learning_rate: float) -> None:
        """Updates weights and biases using the gradients."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients.weights[i]
            self.biases[i] -= learning_rate * gradients.biases[i]

    def predict(
        self,
        X: Float32[np.ndarray, "num_examples input_dimension"]
        | Float32[np.ndarray, "input_dimension"],  # noqa: F821
    ) -> Int64[np.ndarray, "num_examples"]:  # noqa: F821
        """Returns predicted indices, treating a single input as a batch of size one."""
        _, logits = self.forward(np.atleast_2d(X))
        return np.argmax(logits, axis=1).astype(np.int64)
