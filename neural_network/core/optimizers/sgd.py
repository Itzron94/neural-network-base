"""
Stochastic Gradient Descent optimizer.
"""

import numpy as np
from .base import OptimizerFunction
from ..network import NeuralNetwork


class SGDOptimizer(OptimizerFunction):
    """
    Standard Stochastic Gradient Descent optimizer.
    
    Performs simple gradient descent: weights = weights - learning_rate * gradients
    """
    
    def __init__(self):
        """Initialize SGD optimizer."""
        pass
    
    def update_network(self, network: NeuralNetwork, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Update all weights in the network using SGD.
        `gradients` should be a list of lists: gradients[layer][perceptron]
        """
        for layer_idx, layer in enumerate(network.layers):
            grad = np.array(gradients[layer_idx]).transpose()  # Shape: (input_dim + 1, num_perceptrons)
            network.layers[layer_idx].weights = layer.weights - learning_rate * grad
                   
    def reset_state(self) -> None:
        pass