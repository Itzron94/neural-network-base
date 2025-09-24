"""
Base class for optimizer functions.
"""

from abc import ABC, abstractmethod
import numpy as np
from ..network import NeuralNetwork


class OptimizerFunction(ABC):
    
    @abstractmethod
    def update_network(self, network: NeuralNetwork, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        pass
    
    @abstractmethod
    def reset_state(self) -> None:
        pass