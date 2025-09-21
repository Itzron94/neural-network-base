"""
Base class for optimizer functions.
"""

from abc import ABC, abstractmethod
import numpy as np


class OptimizerFunction(ABC):
    
    @abstractmethod
    def update(self, weights: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        pass
    
    @abstractmethod
    def reset_state(self) -> None:
        pass