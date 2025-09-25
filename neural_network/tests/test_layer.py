# test_layer.py

import unittest
import numpy as np
from neural_network.core.layer import Layer
from neural_network.core.activations import ActivationFunctionType, SigmoidActivation, ReLUActivation


class TestLayer(unittest.TestCase):

    def test_initialization(self):
        """Prueba la inicialización de la capa."""
        num_perceptrons = 5
        num_inputs_per_perceptron = 3
        activation_type = "SIGMOID"
        dropout_rate = 0.2

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Verificar que el número de perceptrones es correcto
        self.assertEqual(layer.weights.shape[1], num_perceptrons)

        # Verificar que cada perceptrón tiene el número correcto de entradas y activación
        self.assertEqual(layer.weights.shape[0] - 1, num_inputs_per_perceptron)
        self.assertIsInstance(layer.activation, SigmoidActivation)

        # Verificar que el dropout_rate se asigna correctamente
        self.assertEqual(layer.dropout_rate, dropout_rate)

        # Verificar que outputs y mask se inicializan correctamente
        self.assertEqual(layer.outputs.size, 0)
        self.assertEqual(layer.mask.size, 0)

    def test_forward_without_dropout(self):
        """Prueba el método forward sin Dropout."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 3
        activation_type = "RELU"
        dropout_rate = 0.0

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Configurar pesos y bias conocidos para los perceptrones
        for i in range(num_perceptrons):
            layer.weights[:, i] = np.array([0.5, -0.25, 0.75, 0.1], dtype=np.float32)  # Último elemento es bias

        # Entrada de prueba
        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape: (1, 3)

        # Calcular la salida de la capa
        outputs = layer.forward(inputs, training=True)

        # Cálculo manual de la salida esperada
        bias = np.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        inputs_with_bias = np.hstack([inputs, bias])
        z = np.dot(inputs_with_bias, layer.weights)
        expected_outputs = np.maximum(z, 0)  # ReLU

        # Verificar que las salidas son correctas
        np.testing.assert_array_almost_equal(outputs, expected_outputs)

        # Verificar que los outputs se almacenan correctamente
        np.testing.assert_array_equal(layer.outputs, outputs)

    def test_forward_with_dropout(self):
        """Prueba el método forward con Dropout."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 3
        activation_type = "RELU"
        dropout_rate = 0.5  # 50% de dropout

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Configurar pesos y bias conocidos para los perceptrones
        for i in range(num_perceptrons):
            layer.weights[:, i] = np.array([0.2, -0.1, 0.3, 0.0], dtype=np.float32)  # Último elemento es bias

        # Entrada de prueba
        inputs = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)  # Shape: (1, 3)

        # Calcular la salida de la capa
        outputs = layer.forward(inputs, training=True)

        # Verificar que la máscara tiene la forma correcta
        self.assertEqual(layer.mask.shape, outputs.shape)

        # Calcular outputs sin dropout
        bias = np.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        inputs_with_bias = np.hstack([inputs, bias])
        z = np.dot(inputs_with_bias, layer.weights)
        outputs_without_dropout = np.maximum(z, 0)
        outputs_expected = outputs_without_dropout * layer.mask

        np.testing.assert_array_almost_equal(outputs, outputs_expected)

    def test_forward_with_dropout_inference_mode(self):
        """Prueba el método forward con Dropout en modo inferencia (training=False)."""
        num_perceptrons = 3
        num_inputs_per_perceptron = 2
        activation_type = "SIGMOID"
        dropout_rate = 0.3

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Entrada de prueba
        inputs = np.array([[0.5, -0.5]], dtype=np.float32)  # Shape: (1, 2)

        # Calcular la salida de la capa en modo inferencia
        outputs = layer.forward(inputs, training=False)

        # Cálculo manual de la salida esperada
        bias = np.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        inputs_with_bias = np.hstack([inputs, bias])
        z = np.dot(inputs_with_bias, layer.weights)
        outputs_expected = 1 / (1 + np.exp(-z))
        outputs_expected *= (1 - dropout_rate)

        np.testing.assert_array_almost_equal(outputs, outputs_expected)

    def test_get_activation_derivative(self):
        """Prueba el método get_activation_derivative."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        activation_type = "SIGMOID"
        dropout_rate = 0.5

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Simular una pasada hacia adelante para inicializar last_total y mask
        inputs = np.array([[1.0, -1.0]], dtype=np.float32)
        layer.forward(inputs, training=True)

        # Obtener las derivadas de activación
        derivatives = layer.get_activation_derivative()

        # Cálculo manual de las derivadas esperadas
        z = layer.last_z
        expected_derivatives = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
        expected_derivatives *= layer.mask

        np.testing.assert_array_almost_equal(derivatives, expected_derivatives)

    def test_integration_with_different_activation_functions(self):
        """Prueba la capa con diferentes funciones de activación."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        inputs = np.array([[0.0, 1.0]], dtype=np.float32)

        # Prueba con SigmoidActivation
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, "SIGMOID")
        outputs = layer.forward(inputs)
        self.assertIsInstance(layer.activation, SigmoidActivation)
        self.assertTrue(np.all(outputs >= 0.0) and np.all(outputs <= 1.0))

        # Prueba con ReLUActivation
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, "RELU")
        outputs = layer.forward(inputs)
        self.assertIsInstance(layer.activation, ReLUActivation)
        self.assertTrue(np.all(outputs >= 0.0))

    def test_invalid_input_dimensions(self):
        """Verifica el comportamiento con dimensiones de entrada incorrectas."""
        num_perceptrons = 3
        num_inputs_per_perceptron = 4
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, "SIGMOID")

        # Entrada con dimensiones incorrectas
        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Solo 3 características en lugar de 4

        with self.assertRaises(ValueError):
            layer.forward(inputs)

    def test_forward_multiple_samples(self):
        """Prueba el método forward con múltiples muestras (batch processing)."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 3
        activation_type = "SIGMOID"

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type)

        # Entrada de prueba con batch de tamaño 2
        inputs = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]], dtype=np.float32)  # Shape: (2, 3)

        # Calcular la salida de la capa
        outputs = layer.forward(inputs)

        # Verificar que las salidas tienen la forma correcta
        self.assertEqual(outputs.shape, (2, num_perceptrons))

    def test_dropout_mask_applied_correctly(self):
        """Verifica que la máscara de dropout se aplica correctamente."""
        num_perceptrons = 4
        num_inputs_per_perceptron = 3
        activation_type = "RELU"
        dropout_rate = 0.25

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Entrada de prueba
        inputs = np.ones((5, num_inputs_per_perceptron), dtype=np.float32)  # Batch de tamaño 5

        # Calcular la salida de la capa
        outputs = layer.forward(inputs, training=True)

        # Verificar que la máscara tiene la forma correcta
        self.assertEqual(layer.mask.shape, outputs.shape)

        # Contar la proporción de ceros en la máscara
        dropout_actual = np.mean(layer.mask == 0)
        # Permitir cierta tolerancia debido a la aleatoriedad
        self.assertAlmostEqual(dropout_actual, dropout_rate, delta=0.1)

    def test_forward_training_flag(self):
        """Prueba el efecto del flag 'training' en el método forward."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        dropout_rate = 0.5

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, "SIGMOID", dropout_rate)

        inputs = np.array([[0.5, -0.5]], dtype=np.float32)

        # Calcular la salida con training=True
        outputs_training = layer.forward(inputs, training=True)

        # Guardar la máscara aplicada
        mask_training = layer.mask.copy()

        # Calcular la salida con training=False
        outputs_inference = layer.forward(inputs, training=False)

        # Verificar que las salidas son diferentes
        self.assertFalse(np.array_equal(outputs_training, outputs_inference))

        # Cálculo manual de la salida esperada en modo inferencia
        bias = np.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        inputs_with_bias = np.hstack([inputs, bias])
        z = np.dot(inputs_with_bias, layer.weights)
        expected_outputs = 1 / (1 + np.exp(-z))
        expected_outputs *= (1 - dropout_rate)
        np.testing.assert_array_almost_equal(outputs_inference, expected_outputs)

    def test_get_activation_derivative_without_forward(self):
        """Verifica el comportamiento de get_activation_derivative sin haber llamado a forward."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, "SIGMOID")

        # Intentar obtener las derivadas
        derivatives = layer.get_activation_derivative()

        # Verificar que las derivadas son arrays vacíos
        self.assertEqual(derivatives.size, 0)


if __name__ == '__main__':
    unittest.main()
