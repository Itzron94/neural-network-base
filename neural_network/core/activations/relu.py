# relu.py

import numpy as np
from .base import ActivationFunction


class ReLUActivation(ActivationFunction):
    """
    Función de activación ReLU (Rectified Linear Unit).
    
    Salida: f(x) = max(0, x)
    Rango: [0, +∞)
    
    Características:
    - Computacionalmente eficiente
    - No sufre de saturación para valores positivos
    - Ayuda a mitigar el problema del vanishing gradient
    - Puede sufrir de "dying ReLU" (neurona muerta)
    """
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.0)

    @property
    def name(self) -> str:
        return "ReLU"

    @property
    def output_range(self) -> tuple:
        return (0.0, float('inf'))