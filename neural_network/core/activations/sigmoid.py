# sigmoid.py

import numpy as np
from .base import ActivationFunction


class SigmoidActivation(ActivationFunction):
    """
    Función de activación sigmoide (logística).
    
    Salida: f(x) = 1 / (1 + e^(-x))
    Rango: (0, 1)
    
    Características:
    - Suave y diferenciable en todos los puntos
    - Saturación en los extremos
    - Salida interpretable como probabilidad
    """
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        # Clamp para evitar overflow en exp
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.activate(x)
        return sig * (1.0 - sig)

    @property
    def name(self) -> str:
        return "Sigmoid"

    @property
    def output_range(self) -> tuple:
        return (0.0, 1.0)