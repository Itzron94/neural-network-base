import numpy as np
from typing import List
from src.perceptron import Perceptron
from src.activations import ActivationFunctionType, get_activation_function, SoftmaxActivation


class Layer:
    def __init__(self, num_perceptrons: int, num_inputs_per_perceptron: int,
                 activation_type: ActivationFunctionType = ActivationFunctionType.SIGMOID) -> None:
        """
        Inicializa una capa con múltiples perceptrones.

        :param num_perceptrons: Número de perceptrones en la capa.
        :param num_inputs_per_perceptron: Número de entradas por perceptrón.
        :param activation_type: Tipo de función de activación para los perceptrones.
        """
        self.perceptrons: List[Perceptron] = [
            Perceptron(num_inputs_per_perceptron, activation_type) for _ in range(num_perceptrons)
        ]
        self.outputs: np.ndarray = np.array([])
        self.activation = get_activation_function(activation_type)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante para la capa.

        :param inputs: Array de entradas.
        :return: Array de salidas de la capa.
        """
        if isinstance(self.activation, SoftmaxActivation):
            # Calcular la suma ponderada para cada perceptrón
            totals = np.array([np.dot(perceptron.weights, inputs) + perceptron.bias for perceptron in self.perceptrons])
            # Aplicar Softmax a toda la capa
            self.outputs = self.activation.activate(totals)
            # Actualizar perceptrones con los nuevos totales y salidas
            for perceptron, total, output in zip(self.perceptrons, totals, self.outputs):
                perceptron.last_total = total
                perceptron.last_output = output
        else:
            # Aplicar la función de activación a cada perceptrón individualmente
            self.outputs = np.array([perceptron.calculate_output(inputs) for perceptron in self.perceptrons])

        return self.outputs
