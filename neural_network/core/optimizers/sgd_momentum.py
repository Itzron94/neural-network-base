"""
SGD with momentum optimizer.
"""

import numpy as np
from .base import OptimizerFunction


class SGDMomentumOptimizer(OptimizerFunction):
    """
    Stochastic Gradient Descent with momentum optimizer.
    
    Accumulates velocity over time to help navigate past local minima
    and accelerate convergence in consistent directions.
    """
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.velocity = None
    
    def update(self, weights: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        # Initialize velocity on first update
        if self.velocity is None:
            self.velocity = np.zeros_like(weights, dtype=np.float32)
        
        # Update velocity: v = momentum * v + gradients
        self.velocity = self.momentum * self.velocity + gradients
        
        # Update weights: w = w - learning_rate * velocity
        return weights - learning_rate * self.velocity
    
    def reset_state(self) -> None:
        self.velocity = None