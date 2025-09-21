# test_layer.py

import unittest
import numpy as np
from neural_network.core.layer import Layer
from neural_network.activations.functions import ActivationFunctionType, SigmoidActivation, ReLUActivation
from neural_network.core.perceptron import Perceptron


class TestLayer(unittest.TestCase):

    def test_initialization(self):
        """Prueba la inicialización de la capa."""
        num_perceptrons = 5
        num_inputs_per_perceptron = 3
        activation_type = ActivationFunctionType.SIGMOID
        dropout_rate = 0.2

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Verificar que el número de perceptrones es correcto
        self.assertEqual(len(layer.perceptrons), num_perceptrons)

        # Verificar que cada perceptrón tiene el número correcto de entradas y activación
        for perceptron in layer.perceptrons:
            self.assertEqual(perceptron.weights.shape[0], num_inputs_per_perceptron)
            self.assertIsInstance(perceptron.activation, SigmoidActivation)

        # Verificar que el dropout_rate se asigna correctamente
        self.assertEqual(layer.dropout_rate, dropout_rate)

        # Verificar que outputs y mask se inicializan correctamente
        self.assertEqual(layer.outputs.size, 0)
        self.assertEqual(layer.mask.size, 0)

    def test_forward_without_dropout(self):
        """Prueba el método forward sin Dropout."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 3
        activation_type = ActivationFunctionType.RELU
        dropout_rate = 0.0

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Configurar pesos y bias conocidos para los perceptrones
        for perceptron in layer.perceptrons:
            perceptron.weights = np.array([0.5, -0.25, 0.75], dtype=np.float32)
            perceptron.bias = np.float32(0.1)

        # Entrada de prueba
        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape: (1, 3)

        # Calcular la salida de la capa
        outputs = layer.forward(inputs, training=True)

        # Calcular la salida esperada manualmente
        expected_outputs = []
        for perceptron in layer.perceptrons:
            total_input = np.dot(inputs, perceptron.weights) + perceptron.bias
            output = np.maximum(0.0, total_input)  # ReLU Activation
            expected_outputs.append(output)

        expected_outputs = np.array(expected_outputs).T  # Shape: (1, num_perceptrons)

        # Verificar que las salidas son correctas
        np.testing.assert_array_almost_equal(outputs, expected_outputs)

        # Verificar que los outputs se almacenan correctamente
        np.testing.assert_array_equal(layer.outputs, outputs)

    def test_forward_with_dropout(self):
        """Prueba el método forward con Dropout."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 3
        activation_type = ActivationFunctionType.RELU
        dropout_rate = 0.5  # 50% de dropout

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Configurar pesos y bias conocidos para los perceptrones
        for perceptron in layer.perceptrons:
            perceptron.weights = np.array([0.2, -0.1, 0.3], dtype=np.float32)
            perceptron.bias = np.float32(0.0)

        # Entrada de prueba
        inputs = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)  # Shape: (1, 3)

        # Calcular la salida de la capa
        outputs = layer.forward(inputs, training=True)

        # Verificar que la máscara tiene la forma correcta
        self.assertEqual(layer.mask.shape, outputs.shape)

        # Verificar que los outputs se multiplican por la máscara
        outputs_without_dropout = np.array([perceptron.calculate_output(inputs) for perceptron in layer.perceptrons]).T
        outputs_expected = outputs_without_dropout * layer.mask

        np.testing.assert_array_almost_equal(outputs, outputs_expected)

    def test_forward_with_dropout_inference_mode(self):
        """Prueba el método forward con Dropout en modo inferencia (training=False)."""
        num_perceptrons = 3
        num_inputs_per_perceptron = 2
        activation_type = ActivationFunctionType.SIGMOID
        dropout_rate = 0.3

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Entrada de prueba
        inputs = np.array([[0.5, -0.5]], dtype=np.float32)  # Shape: (1, 2)

        # Calcular la salida de la capa en modo inferencia
        outputs = layer.forward(inputs, training=False)

        # Calcular la salida esperada sin aplicar máscara
        outputs_expected = []
        for perceptron in layer.perceptrons:
            total_input = np.dot(inputs, perceptron.weights) + perceptron.bias
            output = 1 / (1 + np.exp(-total_input))  # Sigmoid Activation
            outputs_expected.append(output)

        outputs_expected = np.array(outputs_expected).T  # Shape: (1, num_perceptrons)

        # En modo inferencia, se escala la salida por (1 - dropout_rate)
        outputs_expected *= (1 - dropout_rate)

        np.testing.assert_array_almost_equal(outputs, outputs_expected)

    def test_get_activation_derivative(self):
        """Prueba el método get_activation_derivative."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        activation_type = ActivationFunctionType.SIGMOID
        dropout_rate = 0.5

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, activation_type, dropout_rate)

        # Simular una pasada hacia adelante para inicializar last_total y mask
        inputs = np.array([[1.0, -1.0]], dtype=np.float32)
        layer.forward(inputs, training=True)

        # Obtener las derivadas de activación
        derivatives = layer.get_activation_derivative()

        # Calcular derivadas esperadas
        expected_derivatives = []
        for perceptron in layer.perceptrons:
            derivative = perceptron.get_activation_derivative()
            expected_derivatives.append(derivative)

        expected_derivatives = np.array(expected_derivatives).T  # Shape: (1, num_perceptrons)

        # Aplicar máscara
        expected_derivatives *= layer.mask

        np.testing.assert_array_almost_equal(derivatives, expected_derivatives)

    def test_integration_with_different_activation_functions(self):
        """Prueba la capa con diferentes funciones de activación."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        inputs = np.array([[0.0, 1.0]], dtype=np.float32)

        # Prueba con SigmoidActivation
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, ActivationFunctionType.SIGMOID)
        outputs = layer.forward(inputs)
        for perceptron in layer.perceptrons:
            self.assertIsInstance(perceptron.activation, SigmoidActivation)

        # Verificar que las salidas están en el rango (0, 1)
        self.assertTrue(np.all(outputs >= 0.0) and np.all(outputs <= 1.0))

        # Prueba con ReLUActivation
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, ActivationFunctionType.RELU)
        outputs = layer.forward(inputs)
        for perceptron in layer.perceptrons:
            self.assertIsInstance(perceptron.activation, ReLUActivation)

        # Verificar que las salidas son no negativas
        self.assertTrue(np.all(outputs >= 0.0))

    def test_invalid_input_dimensions(self):
        """Verifica el comportamiento con dimensiones de entrada incorrectas."""
        num_perceptrons = 3
        num_inputs_per_perceptron = 4
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, ActivationFunctionType.SIGMOID)

        # Entrada con dimensiones incorrectas
        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Solo 3 características en lugar de 4

        with self.assertRaises(ValueError):
            layer.forward(inputs)

    def test_forward_multiple_samples(self):
        """Prueba el método forward con múltiples muestras (batch processing)."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 3
        activation_type = ActivationFunctionType.SIGMOID

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
        activation_type = ActivationFunctionType.RELU
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

        layer = Layer(num_perceptrons, num_inputs_per_perceptron, ActivationFunctionType.SIGMOID, dropout_rate)

        inputs = np.array([[0.5, -0.5]], dtype=np.float32)

        # Calcular la salida con training=True
        outputs_training = layer.forward(inputs, training=True)

        # Guardar la máscara aplicada
        mask_training = layer.mask.copy()

        # Calcular la salida con training=False
        outputs_inference = layer.forward(inputs, training=False)

        # En modo inferencia, la máscara no se actualiza, por lo que no debemos compararla
        # En su lugar, podemos verificar que las salidas son diferentes debido al dropout aplicado en training=True

        # Verificar que las salidas son diferentes
        self.assertFalse(np.array_equal(outputs_training, outputs_inference))

        # Opcionalmente, podemos verificar que la salida en modo inferencia es la salida sin aplicar máscara
        # Calcular la salida esperada en modo inferencia
        expected_outputs = np.array([perceptron.calculate_output(inputs) for perceptron in layer.perceptrons]).T
        expected_outputs *= (1 - dropout_rate)
        np.testing.assert_array_almost_equal(outputs_inference, expected_outputs)

    def test_get_activation_derivative_without_forward(self):
        """Verifica el comportamiento de get_activation_derivative sin haber llamado a forward."""
        num_perceptrons = 2
        num_inputs_per_perceptron = 2
        layer = Layer(num_perceptrons, num_inputs_per_perceptron, ActivationFunctionType.SIGMOID)

        # Intentar obtener las derivadas
        derivatives = layer.get_activation_derivative()

        # Verificar que las derivadas son arrays vacíos
        self.assertEqual(derivatives.size, 0)


if __name__ == '__main__':
    unittest.main()
