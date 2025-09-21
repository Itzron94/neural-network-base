# step_bipolar.py

import numpy as np
from .base import ActivationFunction


class StepBipolarActivation(ActivationFunction):
    """
    Función de activación escalón bipolar.
    
    Salida: +1 si x >= 0, -1 si x < 0
    Rango: [-1, 1]
    
    Esta función es especialmente útil para el perceptrón clásico
    donde se necesitan salidas bipolares.
    """
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0, -1.0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @property
    def name(self) -> str:
        return "Step Bipolar"

    @property
    def output_range(self) -> tuple:
        return (-1.0, 1.0)