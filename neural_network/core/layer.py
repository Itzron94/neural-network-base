# layer.py

import numpy as np
from typing import List
from .perceptron import Perceptron
class Layer:
    def __init__(self, num_perceptrons: int, num_inputs_per_perceptron: int,
                 activation_type: str = "SIGMOID",
                 dropout_rate: float = 0.0) -> None:
        self.perceptrons: List[Perceptron] = [
            Perceptron(num_inputs_per_perceptron, activation_type) for _ in range(num_perceptrons)
        ]
        self.outputs: np.ndarray = np.array([])
        self.dropout_rate: float = dropout_rate
        self.mask: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        outputs = np.array([perceptron.calculate_output(inputs) for perceptron in self.perceptrons]).T
        if training and self.dropout_rate > 0.0:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=outputs.shape).astype(np.float32)
            outputs *= self.mask
        else:
            outputs *= (1 - self.dropout_rate)
        self.outputs = outputs
        return self.outputs

    def get_activation_derivative(self) -> np.ndarray:
        derivatives = np.array([p.get_activation_derivative() for p in self.perceptrons]).T
        if hasattr(self, 'mask') and self.mask.size > 0:
            derivatives *= self.mask
        return derivatives
