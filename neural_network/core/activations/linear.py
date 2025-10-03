# relu.py

import numpy as np
from .base import ActivationFunction


class LinearActivation(ActivationFunction):
    """
    Función de activación lineal (y=x).
    
    Salida: f(x) = x
    Rango: (-∞, +∞)
    """
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    @property
    def name(self) -> str:
        return "Linear"

    @property
    def output_range(self) -> tuple:
        return (float('-inf'), float('inf'))