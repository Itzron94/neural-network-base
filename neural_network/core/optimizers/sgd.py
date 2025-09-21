"""
Stochastic Gradient Descent optimizer.
"""

import numpy as np
from .base import OptimizerFunction


class SGDOptimizer(OptimizerFunction):
    """
    Standard Stochastic Gradient Descent optimizer.
    
    Performs simple gradient descent: weights = weights - learning_rate * gradients
    """
    
    def __init__(self):
        """Initialize SGD optimizer."""
        pass
    
    def update(self, weights: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        return weights - learning_rate * gradients
    
    def reset_state(self) -> None:
        pass