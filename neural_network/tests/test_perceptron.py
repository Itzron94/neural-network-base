# test_perceptron.py

import unittest
import numpy as np
from neural_network.core.perceptron import Perceptron
from neural_network.core.activations import (
    ActivationFunctionType,
    SigmoidActivation,
    ReLUActivation
)
from neural_network.config import WeightInitConfig


class TestPerceptron(unittest.TestCase):

    def test_initialization(self):
        """Prueba la inicialización del perceptrón."""
        num_inputs = 5
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")

        # Verificar dimensiones de los pesos (incluye bias como último peso)
        self.assertEqual(perceptron.weights.shape[0], num_inputs + 1)

        # Verificar que los pesos son float32
        self.assertEqual(perceptron.weights.dtype, np.float32)

        # Verificar función de activación
        self.assertIsInstance(perceptron.activation, SigmoidActivation)


    def test_calculate_output(self):
        """Prueba el método calculate_output."""
        num_inputs = 3
        perceptron = Perceptron(num_inputs, activation_type="RELU")

        # Configurar pesos conocidos (incluye bias como último peso)
        perceptron.weights = np.array([0.5, -0.25, 0.75, 0.1], dtype=np.float32)  # [w1, w2, w3, bias]

        # Entrada de prueba
        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape: (1, 3)

        # Calcular salida
        output = perceptron.calculate_output(inputs)

        # Calcular salida esperada manualmente
        # El método agrega columna de 1s: [[1.0, 2.0, 3.0, 1.0]]
        inputs_with_bias = np.array([[1.0, 2.0, 3.0, 1.0]], dtype=np.float32)
        total_input = np.dot(inputs_with_bias, perceptron.weights)
        expected_output = np.maximum(0.0, total_input)

        # Verificar salida
        np.testing.assert_array_almost_equal(output, expected_output)

        # Verificar que last_input incluye la columna de bias
        expected_last_input = np.array([[1.0, 2.0, 3.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(perceptron.last_input, expected_last_input)
        np.testing.assert_array_equal(perceptron.last_total, total_input)
        np.testing.assert_array_equal(perceptron.last_output, output)


    def test_integration_with_activation_functions(self):
        """Prueba la integración con diferentes funciones de activación."""
        num_inputs = 2
        inputs = np.array([[1.0, -1.0]], dtype=np.float32)

        # Prueba con SigmoidActivation
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")
        perceptron.weights = np.array([0.5, -0.5, 0.0], dtype=np.float32)  # [w1, w2, bias]
        output = perceptron.calculate_output(inputs)
        # Cálculo manual: inputs se convierte a [1.0, -1.0, 1.0]
        inputs_with_bias = np.array([[1.0, -1.0, 1.0]], dtype=np.float32)
        expected_total = np.dot(inputs_with_bias, perceptron.weights)
        expected_output = 1 / (1 + np.exp(-expected_total))
        np.testing.assert_array_almost_equal(output, expected_output)

        # Prueba con ReLUActivation
        perceptron = Perceptron(num_inputs, activation_type="RELU")
        perceptron.weights = np.array([0.5, -0.5, 0.0], dtype=np.float32)  # [w1, w2, bias]
        output = perceptron.calculate_output(inputs)
        expected_output = np.maximum(0.0, expected_total)
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_invalid_input_dimensions(self):
        """Verifica el comportamiento con dimensiones de entrada incorrectas."""
        num_inputs = 3
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")
        inputs = np.array([[1.0, 2.0]], dtype=np.float32)  # Solo 2 entradas en lugar de 3

        with self.assertRaises(ValueError):
            perceptron.calculate_output(inputs)

    def test_empty_input(self):
        """Prueba el comportamiento al crear un perceptrón con num_inputs = 0."""
        num_inputs = 0
        with self.assertRaises(ValueError):
            perceptron = Perceptron(num_inputs, activation_type="SIGMOID")


    def test_get_activation_derivative(self):
        """Prueba el método get_activation_derivative."""
        num_inputs = 2
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")
        perceptron.last_total = np.array([0.0, 1.0], dtype=np.float32)

        derivative = perceptron.get_activation_derivative()
        expected_derivative = perceptron.activation.derivative(perceptron.last_total)

        np.testing.assert_array_almost_equal(derivative, expected_derivative)


    def test_weight_initialization_types(self):
        """Prueba diferentes tipos de inicialización de pesos."""
        num_inputs = 3
        
        # Test zeros initialization
        weight_config = WeightInitConfig(init_type="zeros")
        perceptron = Perceptron(num_inputs, "SIGMOID", weight_init_config=weight_config)
        np.testing.assert_array_equal(perceptron.weights, np.zeros(num_inputs + 1, dtype=np.float32))
        
        # Test ones initialization
        weight_config = WeightInitConfig(init_type="ones")
        perceptron = Perceptron(num_inputs, "SIGMOID", weight_init_config=weight_config)
        np.testing.assert_array_equal(perceptron.weights, np.ones(num_inputs + 1, dtype=np.float32))



if __name__ == '__main__':
    unittest.main()
