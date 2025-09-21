# step.py

import numpy as np
from .base import ActivationFunction


class StepActivation(ActivationFunction):
    """
    Función de activación escalón unitario.
    
    Salida: 1 si x >= 0, 0 si x < 0
    Rango: [0, 1]
    """
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0, 0.0)

    def derivative(self, x: np.ndarray) -> np.ndarray:

        return np.zeros_like(x)

    @property
    def name(self) -> str:
        return "Step"

    @property
    def output_range(self) -> tuple:
        return (0.0, 1.0)