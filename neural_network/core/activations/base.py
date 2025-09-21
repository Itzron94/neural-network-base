# base.py

import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    
    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre descriptivo de la función de activación"""
        pass

    @property
    @abstractmethod
    def output_range(self) -> tuple:
        """Rango de salida de la función (min, max)"""
        pass