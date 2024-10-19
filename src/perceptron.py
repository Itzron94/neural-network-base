# perceptron.py

import numpy as np
from .activations import ActivationFunction, get_activation_function, ActivationFunctionType


class Perceptron:
    def __init__(self, num_inputs: int, activation_type: ActivationFunctionType) -> None:
        if num_inputs <= 0:
            raise ValueError("El número de entradas 'num_inputs' debe ser mayor que cero.")

        self.weights: np.ndarray = (np.random.randn(num_inputs) * np.sqrt(2. / num_inputs)).astype(np.float32)
        self.bias: float = np.float32(np.random.randn())
        self.activation: ActivationFunction = get_activation_function(activation_type)
        self.last_input: np.ndarray = np.array([])
        self.last_total: np.ndarray = np.array([])
        self.last_output: np.ndarray = np.array([])

        # Parámetros para Adam
        self.m_w: np.ndarray = np.zeros_like(self.weights, dtype=np.float32)
        self.v_w: np.ndarray = np.zeros_like(self.weights, dtype=np.float32)
        self.m_b: float = np.float32(0.0)
        self.v_b: float = np.float32(0.0)
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.epsilon: float = 1e-8
        self.timestep: int = 0  # Para el sesgo de corrección

    def calculate_output(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError("El número de características en 'inputs' no coincide con el número de pesos.")
        self.last_input = inputs
        self.last_total = np.dot(inputs, self.weights) + self.bias
        self.last_output = self.activation.activate(self.last_total)
        return self.last_output

    def update_weights(self, delta: np.ndarray, learning_rate: float) -> None:
        if self.last_input.size == 0:
            raise ValueError("El método 'calculate_output' debe ser llamado antes de 'update_weights'.")

        gradient_w = np.dot(self.last_input.T, delta) / delta.shape[0]
        gradient_b = np.mean(delta)

        self.timestep += 1

        # Actualizar los momentos para los pesos
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * gradient_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (gradient_w ** 2)

        # Corregir el sesgo de los momentos
        m_w_hat = self.m_w / (1 - self.beta1 ** self.timestep)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.timestep)

        # Actualizar los pesos
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # Actualizar los momentos para el bias
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * gradient_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (gradient_b ** 2)

        # Corregir el sesgo de los momentos
        m_b_hat = self.m_b / (1 - self.beta1 ** self.timestep)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.timestep)

        # Actualizar el bias
        self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def get_activation_derivative(self) -> np.ndarray:
        return self.activation.derivative(self.last_total)
