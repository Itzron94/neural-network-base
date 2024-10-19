import numpy as np
from src.activations import ActivationFunction, ActivationFunctionType, get_activation_function, SoftmaxActivation


class Perceptron:
    def __init__(self, num_inputs: int,
                 activation_type: ActivationFunctionType = ActivationFunctionType.SIGMOID) -> None:
        """
        Inicializa un perceptrón con pesos aleatorios y bias.

        :param num_inputs: Número de entradas que recibe el perceptrón.
        :param activation_type: Tipo de función de activación.
        """
        self.weights: np.ndarray = np.random.randn(num_inputs)
        self.bias: float = np.random.randn()
        self.activation: ActivationFunction = get_activation_function(activation_type)
        self.last_input: np.ndarray = np.array([])
        self.last_total: float = 0.0
        self.last_output: float = 0.0

    def predict(self, inputs: np.ndarray) -> float:
        """
        Realiza la predicción para un conjunto de entradas.

        :param inputs: Array de entradas.
        :return: Salida del perceptrón.
        """
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation.activate(total)

    def calculate_output(self, inputs: np.ndarray) -> float:
        """
        Calcula la salida y almacena la suma ponderada para usar en la derivada.

        :param inputs: Array de entradas.
        :return: Salida del perceptrón.
        """
        self.last_input = inputs
        self.last_total = np.dot(self.weights, inputs) + self.bias
        if not isinstance(self.activation, SoftmaxActivation):
            self.last_output = self.activation.activate(np.array([self.last_total]))[0]
        else:
            # Si la activación es Softmax, la salida se manejará a nivel de capa
            self.last_output = self.last_total
        return self.last_output

    def update_weights(self, delta: float, learning_rate: float) -> None:
        """
        Actualiza los pesos y el bias del perceptrón.

        :param delta: Delta calculado durante la retropropagación.
        :param learning_rate: Tasa de aprendizaje.
        """
        self.weights -= learning_rate * delta * self.last_input
        self.bias -= learning_rate * delta

    def get_activation_derivative(self) -> float:
        """
        Obtiene la derivada de la función de activación evaluada en la última suma ponderada.

        :return: Derivada de la función de activación.
        """
        return self.activation.derivative(self.last_total)
