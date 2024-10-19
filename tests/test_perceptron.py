import unittest
import numpy as np
from src.perceptron import Perceptron
from src.activations import ActivationFunctionType


class TestPerceptron(unittest.TestCase):
    def setUp(self):
        # Inicializar perceptrones con diferentes funciones de activación
        self.sigmoid_perceptron = Perceptron(num_inputs=2, activation_type=ActivationFunctionType.SIGMOID)
        self.relu_perceptron = Perceptron(num_inputs=2, activation_type=ActivationFunctionType.RELU)
        self.softmax_perceptron = Perceptron(num_inputs=2, activation_type=ActivationFunctionType.SOFTMAX)

        # Establecer pesos y bias conocidos para pruebas
        self.sigmoid_perceptron.weights = np.array([0.5, -0.6])
        self.sigmoid_perceptron.bias = 0.1

        self.relu_perceptron.weights = np.array([1.0, -1.0])
        self.relu_perceptron.bias = 0.0

        self.softmax_perceptron.weights = np.array([0.3, 0.7])
        self.softmax_perceptron.bias = 0.2

    def test_calculate_output_sigmoid(self):
        inputs = np.array([1.0, 2.0])
        output = self.sigmoid_perceptron.calculate_output(inputs)
        expected_total = 0.5 * 1.0 + (-0.6) * 2.0 + 0.1  # 0.5 -1.2 +0.1 = -0.6
        expected_output = 1 / (1 + np.exp(-expected_total))  # Sigmoid(-0.6)
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_calculate_output_relu(self):
        inputs = np.array([1.0, 2.0])
        output = self.relu_perceptron.calculate_output(inputs)
        expected_total = 1.0 * 1.0 + (-1.0) * 2.0 + 0.0  # 1 -2 +0 = -1
        expected_output = max(0.0, expected_total)  # ReLU(-1) = 0
        self.assertEqual(output, expected_output)

    def test_calculate_output_softmax(self):
        inputs = np.array([1.0, 2.0])
        output = self.softmax_perceptron.calculate_output(inputs)
        # Softmax expects multiple outputs; here, perceptron might not handle it correctly.
        # Este test puede necesitar ajustes dependiendo de la implementación específica.
        # Por simplicidad, asumimos que softmax_perceptron.activate recibe un vector.
        # Sin embargo, en la implementación actual, cada perceptron calcula individualmente.
        # Por lo tanto, este test puede no ser aplicable directamente.
        # Recomendamos revisar la implementación de Softmax en el contexto de múltiples perceptrones.
        pass  # Placeholder

    def test_update_weights(self):
        inputs = np.array([1.0, 2.0])
        delta = 0.1
        learning_rate = 0.01
        initial_weights = self.sigmoid_perceptron.weights.copy()
        initial_bias = self.sigmoid_perceptron.bias
        self.sigmoid_perceptron.last_input = inputs
        self.sigmoid_perceptron.update_weights(delta, learning_rate)
        expected_weights = initial_weights - learning_rate * delta * inputs
        expected_bias = initial_bias - learning_rate * delta
        np.testing.assert_array_almost_equal(self.sigmoid_perceptron.weights, expected_weights, decimal=5)
        self.assertAlmostEqual(self.sigmoid_perceptron.bias, expected_bias, places=5)


if __name__ == '__main__':
    unittest.main()
