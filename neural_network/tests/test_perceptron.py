# test_perceptron.py

import unittest
import numpy as np
from neural_network.core.perceptron import Perceptron
from neural_network.core.activations import (
    ActivationFunctionType,
    SigmoidActivation,
    ReLUActivation
)
from neural_network.config import OptimizerConfig, WeightInitConfig


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

        # Verificar optimizer existe
        self.assertIsNotNone(perceptron.optimizer)

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

    def test_update_weights(self):
        """Prueba el método update_weights con el optimizador."""
        num_inputs = 4
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")

        # Configurar entradas y delta conocidos (incluye bias column)
        perceptron.last_input = np.array([[0.1, 0.2, 0.3, 0.4, 1.0]], dtype=np.float32)
        delta = np.array([0.5], dtype=np.float32)
        learning_rate = 0.01

        # Guardar copia de pesos antes de la actualización
        weights_before = perceptron.weights.copy()

        # Actualizar pesos
        perceptron.update_weights(delta, learning_rate)

        # Verificar que los pesos cambiaron
        self.assertFalse(np.array_equal(perceptron.weights, weights_before))

        # Verificar que las dimensiones son correctas
        self.assertEqual(perceptron.weights.shape[0], num_inputs + 1)

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

    def test_update_weights_without_previous_forward(self):
        """Verifica que update_weights maneja adecuadamente si calculate_output no ha sido llamado."""
        num_inputs = 3
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")
        delta = np.array([0.5], dtype=np.float32)
        learning_rate = 0.01

        # Intentar actualizar pesos sin haber llamado a calculate_output
        with self.assertRaises(ValueError):
            perceptron.update_weights(delta, learning_rate)

    def test_weight_updates_with_batch_inputs(self):
        """Prueba update_weights con batch de entradas y deltas."""
        num_inputs = 2
        perceptron = Perceptron(num_inputs, activation_type="SIGMOID")
        # last_input debe incluir la columna de bias para batch
        perceptron.last_input = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]], dtype=np.float32)  # Batch con bias
        delta = np.array([0.1, -0.2], dtype=np.float32)  # Batch de deltas
        learning_rate = 0.01

        # Guardar pesos antes de la actualización
        weights_before = perceptron.weights.copy()

        # Actualizar pesos
        perceptron.update_weights(delta, learning_rate)

        # Verificar que los pesos se actualizan
        self.assertFalse(np.array_equal(perceptron.weights, weights_before))

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

    def test_optimizer_configuration(self):
        """Prueba la configuración de diferentes optimizadores."""
        num_inputs = 2
        
        # Test SGD optimizer
        optimizer_config = OptimizerConfig(type="SGD")
        perceptron = Perceptron(num_inputs, "SIGMOID", optimizer_config=optimizer_config)
        self.assertIsNotNone(perceptron.optimizer)
        
        # Test Adam optimizer
        optimizer_config = OptimizerConfig(type="ADAM", beta1=0.8, beta2=0.99)
        perceptron = Perceptron(num_inputs, "SIGMOID", optimizer_config=optimizer_config)
        self.assertIsNotNone(perceptron.optimizer)


if __name__ == '__main__':
    unittest.main()
