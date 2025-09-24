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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
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
            for p_idx, perceptron in enumerate(layer.perceptrons):
                key = id(perceptron)
                grad = gradients[layer_idx][p_idx]
                weights = perceptron.weights

                if key not in self.m:
                    self.m[key] = np.zeros_like(weights, dtype=np.float32)
                    self.v[key] = np.zeros_like(weights, dtype=np.float32)
                    self.timestep[key] = 0

                self.timestep[key] += 1
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.timestep[key])
                v_hat = self.v[key] / (1 - self.beta2 ** self.timestep[key])
                perceptron.weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    # def update(self, weights: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
    #     # Initialize moment vectors on first update
    #     if self.m is None:
    #         self.m = np.zeros_like(weights, dtype=np.float32)
    #         self.v = np.zeros_like(weights, dtype=np.float32)
        
    #     self.timestep += 1
        
    #     # Update biased first moment estimate
    #     self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
    #     # Update biased second raw moment estimate
    #     self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
    #     # Compute bias-corrected first moment estimate
    #     m_hat = self.m / (1 - self.beta1 ** self.timestep)
        
    #     # Compute bias-corrected second raw moment estimate
    #     v_hat = self.v / (1 - self.beta2 ** self.timestep)
        
    #     # Update weights
    #     return weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.m = {}
        self.v = {}
        self.timestep = {}