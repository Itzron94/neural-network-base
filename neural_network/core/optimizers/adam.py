"""
Adam optimizer.
"""

import numpy as np
from .base import OptimizerFunction
from ..network import NeuralNetwork


class AdamOptimizer(OptimizerFunction):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines the advantages of AdaGrad and RMSprop by computing adaptive
    learning rates for each parameter from estimates of first and second
    moments of the gradients.
    """
    
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        # State variables
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.timestep = {}

    def update_network(self, network: NeuralNetwork, gradients, learning_rate: float):
        """
        Update all weights in the network using Adam.
        `gradients` should be a list of lists: gradients[layer][perceptron]
        """
        for layer_idx, layer in enumerate(network.layers):
            key = id(layer)
            grad = np.array(gradients[layer_idx]).transpose()  # Shape: (num_inputs + 1, num_perceptrons)
            weights = layer.weights

            if key not in self.m:
                self.m[key] = np.zeros_like(weights, dtype=np.float32)
                self.v[key] = np.zeros_like(weights, dtype=np.float32)
                self.timestep[key] = 0

            self.timestep[key] += 1
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.timestep[key])
            v_hat = self.v[key] / (1 - self.beta2 ** self.timestep[key])
            network.layers[layer_idx].weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    
        # for layer_idx, layer in enumerate(network.layers):
        #     for p_idx, perceptron in enumerate(layer.perceptrons):
        #         key = id(perceptron)
        #         grad = gradients[layer_idx][p_idx]
        #         weights = perceptron.weights

        #         if key not in self.m:
        #             self.m[key] = np.zeros_like(weights, dtype=np.float32)
        #             self.v[key] = np.zeros_like(weights, dtype=np.float32)
        #             self.timestep[key] = 0

        #         self.timestep[key] += 1
        #         self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        #         self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
        #         m_hat = self.m[key] / (1 - self.beta1 ** self.timestep[key])
        #         v_hat = self.v[key] / (1 - self.beta2 ** self.timestep[key])
        #         perceptron.weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.m = {}
        self.v = {}
        self.timestep = {}