import numpy as np
from enum import Enum, auto
from abc import ABC, abstractmethod


# -------------------------------
# Enum para Funciones de Activación
# -------------------------------
class ActivationFunctionType(Enum):
    STEP = auto()
    SIGMOID = auto()
    RELU = auto()
    SOFTMAX = auto()


# -------------------------------
# Clase Abstracta para Funciones de Activación
# -------------------------------
class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x: float) -> float:
        """Aplica la función de activación."""
        pass

    @abstractmethod
    def derivative(self, x: float) -> float:
        """Calcula la derivada de la función de activación."""
        pass


# -------------------------------
# Implementaciones de Funciones de Activación
# -------------------------------
class StepActivation(ActivationFunction):
    def activate(self, x: float) -> float:
        return 1.0 if x >= 0 else 0.0

    def derivative(self, x: float) -> float:
        # La derivada de la función escalón es 0 en todas partes excepto en discontinuidades
        return 0.0


class SigmoidActivation(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)


class ReLUActivation(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: float) -> float:
        return 1.0 if x > 0 else 0.0


class SoftmaxActivation(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)

    def derivative(self, x: float) -> float:
        # La derivada se maneja junto con la entropía cruzada
        return 1.0  # Placeholder


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
    elif act_type == ActivationFunctionType.SOFTMAX:
        return SoftmaxActivation()
    else:
        raise ValueError(f"Tipo de función de activación '{act_type}' no soportada.")
