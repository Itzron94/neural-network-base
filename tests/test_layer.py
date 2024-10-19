import unittest
import numpy as np
from src.layer import Layer
from src.activations import SoftmaxActivation, ActivationFunctionType, SigmoidActivation


class TestLayer(unittest.TestCase):
    def setUp(self):
        # Crear una capa con 3 perceptrones para Softmax
        self.softmax_activation = SoftmaxActivation()
        self.softmax_layer = Layer(num_perceptrons=3, num_inputs_per_perceptron=2, activation_type=ActivationFunctionType.SOFTMAX)

        # Establecer pesos y biases conocidos
        self.softmax_layer.perceptrons[0].weights = np.array([1.0, 2.0])
        self.softmax_layer.perceptrons[0].bias = 0.0

        self.softmax_layer.perceptrons[1].weights = np.array([0.5, -1.0])
        self.softmax_layer.perceptrons[1].bias = 0.5

        self.softmax_layer.perceptrons[2].weights = np.array([-1.5, 1.0])
        self.softmax_layer.perceptrons[2].bias = -0.5

    def test_forward_softmax(self):
        inputs = np.array([1.0, 2.0])
        outputs = self.softmax_layer.forward(inputs)

        # Calcular la suma ponderada para cada perceptrón
        total1 = 1.0 * 1.0 + 2.0 * 2.0 + 0.0  # 1 + 4 +0 =5
        total2 = 0.5 * 1.0 + (-1.0) * 2.0 + 0.5  # 0.5 -2 +0.5 =-1
        total3 = (-1.5) * 1.0 + 1.0 * 2.0 + (-0.5)  # -1.5 +2 -0.5 =0

        # Aplicar Softmax manualmente
        exps = np.exp(np.array([total1, total2, total3]))
        expected = exps / np.sum(exps)

        np.testing.assert_array_almost_equal(outputs, expected, decimal=5)

    def test_forward_softmax_large_values(self):
        inputs = np.array([10.0, 10.0])
        # Establecer pesos y biases para grandes valores
        self.softmax_layer.perceptrons[0].weights = np.array([10.0, 10.0])
        self.softmax_layer.perceptrons[0].bias = 0.0

        self.softmax_layer.perceptrons[1].weights = np.array([10.0, 10.0])
        self.softmax_layer.perceptrons[1].bias = 0.0

        self.softmax_layer.perceptrons[2].weights = np.array([10.0, 10.0])
        self.softmax_layer.perceptrons[2].bias = 0.0

        outputs = self.softmax_layer.forward(inputs)

        # Calcular la suma ponderada para cada perceptrón
        total1 = 10.0 * 10.0 + 10.0 * 10.0 + 0.0  # 100 + 100 +0 =200
        total2 = 10.0 * 10.0 + 10.0 * 10.0 + 0.0  # 200
        total3 = 10.0 * 10.0 + 10.0 * 10.0 + 0.0  # 200

        # Aplicar Softmax manualmente
        exps = np.exp(np.array([total1, total2, total3]) - np.max([total1, total2, total3]))
        expected = exps / np.sum(exps)

        np.testing.assert_array_almost_equal(outputs, expected, decimal=5)

    def test_forward_sigmoid(self):
        # Prueba anterior para capas con Sigmoid
        activation = SigmoidActivation()
        layer = Layer(num_perceptrons=2, num_inputs_per_perceptron=2, activation_type=ActivationFunctionType.SIGMOID)
        layer.perceptrons[0].weights = np.array([0.5, -0.5])
        layer.perceptrons[0].bias = 0.0
        layer.perceptrons[1].weights = np.array([-0.3, 0.8])
        layer.perceptrons[1].bias = 0.1

        inputs = np.array([1.0, 2.0])
        outputs = layer.forward(inputs)

        # Cálculos esperados
        total1 = 0.5 * 1.0 + (-0.5) * 2.0 + 0.0  # 0.5 -1.0 +0 = -0.5
        output1 = 1 / (1 + np.exp(-total1))  # Sigmoid(-0.5)

        total2 = (-0.3) * 1.0 + 0.8 * 2.0 + 0.1  # -0.3 +1.6 +0.1 =1.4
        output2 = 1 / (1 + np.exp(-total2))  # Sigmoid(1.4)

        expected_outputs = np.array([output1, output2])
        np.testing.assert_array_almost_equal(outputs, expected_outputs, decimal=5)


if __name__ == '__main__':
    unittest.main()
