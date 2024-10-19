# activation_functions.py

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto


# -------------------------------
# Enum para Funciones de Activación
# -------------------------------
class ActivationFunctionType(Enum):
    STEP = auto()
    SIGMOID = auto()
    RELU = auto()


# -------------------------------
# Clase Abstracta para Funciones de Activación
# -------------------------------
class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


# -------------------------------
# Implementaciones de Funciones de Activación
# -------------------------------
class StepActivation(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0, 0.0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class SigmoidActivation(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.activate(x)
        return sig * (1 - sig)


class ReLUActivation(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.0)


# -------------------------------
# Factoría para Obtener la Función de Activación
# -------------------------------
def get_activation_function(act_type: ActivationFunctionType) -> ActivationFunction:
    if act_type == ActivationFunctionType.STEP:
        return StepActivation()
    elif act_type == ActivationFunctionType.SIGMOID:
        return SigmoidActivation()
    elif act_type == ActivationFunctionType.RELU:
        return ReLUActivation()
    else:
        raise ValueError(f"Tipo de función de activación '{act_type}' no soportada.")
