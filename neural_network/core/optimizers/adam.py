"""
Adam optimizer.
"""

import numpy as np
from .base import OptimizerFunction


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
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.timestep = 0
    
    def update(self, weights: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        # Initialize moment vectors on first update
        if self.m is None:
            self.m = np.zeros_like(weights, dtype=np.float32)
            self.v = np.zeros_like(weights, dtype=np.float32)
        
        self.timestep += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.timestep)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.timestep)
        
        # Update weights
        return weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.timestep = 0