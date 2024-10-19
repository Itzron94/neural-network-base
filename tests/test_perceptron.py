# test_perceptron.py

import unittest
import numpy as np
from src.perceptron import Perceptron
from src.activations import (
    ActivationFunctionType,
    SigmoidActivation,
    ReLUActivation
)


class TestPerceptron(unittest.TestCase):

    def test_initialization(self):
        """Prueba la inicialización del perceptrón."""
        num_inputs = 5
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)

        # Verificar dimensiones de los pesos
        self.assertEqual(perceptron.weights.shape[0], num_inputs)

        # Verificar que los pesos son float32
        self.assertEqual(perceptron.weights.dtype, np.float32)

        # Verificar que el bias es float32
        self.assertIsInstance(perceptron.bias, np.float32)

        # Verificar momentos inicializados en cero
        np.testing.assert_array_equal(perceptron.m_w, np.zeros(num_inputs, dtype=np.float32))
        np.testing.assert_array_equal(perceptron.v_w, np.zeros(num_inputs, dtype=np.float32))
        self.assertEqual(perceptron.m_b, 0.0)
        self.assertEqual(perceptron.v_b, 0.0)

        # Verificar función de activación
        self.assertIsInstance(perceptron.activation, SigmoidActivation)

    def test_calculate_output(self):
        """Prueba el método calculate_output."""
        num_inputs = 3
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.RELU)

        # Configurar pesos y bias conocidos
        perceptron.weights = np.array([0.5, -0.25, 0.75], dtype=np.float32)
        perceptron.bias = np.float32(0.1)

        # Entrada de prueba
        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape: (1, 3)

        # Calcular salida
        output = perceptron.calculate_output(inputs)

        # Calcular salida esperada manualmente
        total_input = np.dot(inputs, perceptron.weights) + perceptron.bias
        expected_output = np.maximum(0.0, total_input)

        # Verificar salida
        np.testing.assert_array_almost_equal(output, expected_output)

        # Verificar que last_input, last_total y last_output se almacenan correctamente
        np.testing.assert_array_equal(perceptron.last_input, inputs)
        np.testing.assert_array_equal(perceptron.last_total, total_input)
        np.testing.assert_array_equal(perceptron.last_output, output)

    def test_update_weights(self):
        """Prueba el método update_weights con el optimizador Adam."""
        num_inputs = 4
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)

        # Configurar entradas y delta conocidos
        perceptron.last_input = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        delta = np.array([0.5], dtype=np.float32)
        learning_rate = 0.01

        # Guardar copia de pesos y bias antes de la actualización
        weights_before = perceptron.weights.copy()
        bias_before = perceptron.bias

        # Actualizar pesos
        perceptron.update_weights(delta, learning_rate)

        # Calcular gradiente esperado
        gradient_w = np.dot(perceptron.last_input.T, delta) / delta.shape[0]
        gradient_b = np.mean(delta)

        # Actualizar momentos manualmente
        m_w_expected = perceptron.beta1 * np.zeros_like(gradient_w) + (1 - perceptron.beta1) * gradient_w
        v_w_expected = perceptron.beta2 * np.zeros_like(gradient_w) + (1 - perceptron.beta2) * (gradient_w ** 2)
        m_b_expected = perceptron.beta1 * 0.0 + (1 - perceptron.beta1) * gradient_b
        v_b_expected = perceptron.beta2 * 0.0 + (1 - perceptron.beta2) * (gradient_b ** 2)

        # Corregir el sesgo de los momentos
        m_w_hat = m_w_expected / (1 - perceptron.beta1 ** perceptron.timestep)
        v_w_hat = v_w_expected / (1 - perceptron.beta2 ** perceptron.timestep)
        m_b_hat = m_b_expected / (1 - perceptron.beta1 ** perceptron.timestep)
        v_b_hat = v_b_expected / (1 - perceptron.beta2 ** perceptron.timestep)

        # Calcular pesos y bias esperados
        weights_expected = weights_before - learning_rate * m_w_hat / (np.sqrt(v_w_hat) + perceptron.epsilon)
        bias_expected = bias_before - learning_rate * m_b_hat / (np.sqrt(v_b_hat) + perceptron.epsilon)

        # Verificar pesos y bias actualizados
        np.testing.assert_array_almost_equal(perceptron.weights, weights_expected)
        self.assertAlmostEqual(perceptron.bias, bias_expected, places=7)

        # Verificar actualización de momentos
        np.testing.assert_array_almost_equal(perceptron.m_w, m_w_expected)
        np.testing.assert_array_almost_equal(perceptron.v_w, v_w_expected)
        self.assertAlmostEqual(perceptron.m_b, m_b_expected, places=7)
        self.assertAlmostEqual(perceptron.v_b, v_b_expected, places=7)

    def test_integration_with_activation_functions(self):
        """Prueba la integración con diferentes funciones de activación."""
        num_inputs = 2
        inputs = np.array([[1.0, -1.0]], dtype=np.float32)

        # Prueba con SigmoidActivation
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)
        perceptron.weights = np.array([0.5, -0.5], dtype=np.float32)
        perceptron.bias = np.float32(0.0)
        output = perceptron.calculate_output(inputs)
        expected_total = np.dot(inputs, perceptron.weights) + perceptron.bias
        expected_output = 1 / (1 + np.exp(-expected_total))
        np.testing.assert_array_almost_equal(output, expected_output)

        # Prueba con ReLUActivation
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.RELU)
        perceptron.weights = np.array([0.5, -0.5], dtype=np.float32)
        perceptron.bias = np.float32(0.0)
        output = perceptron.calculate_output(inputs)
        expected_output = np.maximum(0.0, expected_total)
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_invalid_input_dimensions(self):
        """Verifica el comportamiento con dimensiones de entrada incorrectas."""
        num_inputs = 3
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)
        inputs = np.array([[1.0, 2.0]], dtype=np.float32)  # Solo 2 entradas en lugar de 3

        with self.assertRaises(ValueError):
            perceptron.calculate_output(inputs)

    def test_empty_input(self):
        """Prueba el comportamiento al crear un perceptrón con num_inputs = 0."""
        num_inputs = 0
        with self.assertRaises(ValueError):
            perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)


    def test_get_activation_derivative(self):
        """Prueba el método get_activation_derivative."""
        num_inputs = 2
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)
        perceptron.last_total = np.array([[0.0, 1.0]], dtype=np.float32)

        derivative = perceptron.get_activation_derivative()
        expected_derivative = perceptron.activation.derivative(perceptron.last_total)

        np.testing.assert_array_almost_equal(derivative, expected_derivative)

    def test_update_weights_without_previous_forward(self):
        """Verifica que update_weights maneja adecuadamente si calculate_output no ha sido llamado."""
        num_inputs = 3
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)
        delta = np.array([0.5], dtype=np.float32)
        learning_rate = 0.01

        # Intentar actualizar pesos sin haber llamado a calculate_output
        with self.assertRaises(ValueError):
            perceptron.update_weights(delta, learning_rate)

    def test_weight_updates_with_batch_inputs(self):
        """Prueba update_weights con batch de entradas y deltas."""
        num_inputs = 2
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)
        perceptron.last_input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # Batch de tamaño 2
        delta = np.array([0.1, -0.2], dtype=np.float32)  # Batch de deltas
        learning_rate = 0.01

        # Guardar pesos y bias antes de la actualización
        weights_before = perceptron.weights.copy()
        bias_before = perceptron.bias

        # Actualizar pesos
        perceptron.update_weights(delta, learning_rate)

        # Verificar que los pesos y bias se actualizan (no se calculan manualmente aquí)
        self.assertFalse(np.array_equal(perceptron.weights, weights_before))
        self.assertNotEqual(perceptron.bias, bias_before)

    def test_timestep_increment(self):
        """Verifica que el timestep se incrementa en cada llamada a update_weights."""
        num_inputs = 2
        perceptron = Perceptron(num_inputs, activation_type=ActivationFunctionType.SIGMOID)
        perceptron.last_input = np.array([[1.0, 2.0]], dtype=np.float32)
        delta = np.array([0.1], dtype=np.float32)
        learning_rate = 0.01

        initial_timestep = perceptron.timestep
        perceptron.update_weights(delta, learning_rate)
        self.assertEqual(perceptron.timestep, initial_timestep + 1)

    def test_bias_type(self):
        """Verifica que el bias es del tipo correcto."""
        perceptron = Perceptron(3, activation_type=ActivationFunctionType.SIGMOID)
        self.assertIsInstance(perceptron.bias, np.float32)
        perceptron.bias = 0.5  # Asignar nuevo valor
        self.assertIsInstance(perceptron.bias, float)  # Asegurarse de que sigue siendo float


if __name__ == '__main__':
    unittest.main()
