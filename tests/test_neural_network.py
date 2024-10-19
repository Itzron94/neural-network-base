# test_neural_network.py

import unittest
import numpy as np
import os
from src.neural_network import NeuralNetwork
from src.activations import ActivationFunctionType


class TestNeuralNetwork(unittest.TestCase):

    def test_initialization(self):
        """Prueba la inicialización de la red neuronal."""
        topology = [4, 5, 3]
        activation_type = ActivationFunctionType.SIGMOID
        learning_rate = 0.01
        epochs = 100
        dropout_rate = 0.2

        nn = NeuralNetwork(
            topology=topology,
            activation_type=activation_type,
            learning_rate=learning_rate,
            epochs=epochs,
            dropout_rate=dropout_rate
        )

        # Verificar que las capas se crearon correctamente
        self.assertEqual(len(nn.layers), len(topology) - 1)

        # Verificar que cada capa tiene el número correcto de perceptrones
        for i, layer in enumerate(nn.layers):
            self.assertEqual(len(layer.perceptrons), topology[i + 1])
            for perceptron in layer.perceptrons:
                self.assertEqual(perceptron.weights.shape[0], topology[i])

        # Verificar que los parámetros se asignan correctamente
        self.assertEqual(nn.learning_rate, learning_rate)
        self.assertEqual(nn.epochs, epochs)

    def test_forward_pass(self):
        """Prueba el método forward con una entrada sintética."""
        topology = [3, 4, 2]
        nn = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.RELU)

        # Entrada sintética
        inputs = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)  # Shape: (1, 3)

        # Propagación hacia adelante
        outputs = nn.forward(inputs, training=False)

        # Verificar que la salida tiene la forma correcta
        self.assertEqual(outputs.shape, (1, topology[-1]))

        # Verificar que las salidas son números reales
        self.assertTrue(np.all(np.isfinite(outputs)))

    def test_train_on_synthetic_data(self):
        """Prueba el método train con datos sintéticos."""
        topology = [2, 3, 2]  # Cambiado a 2 neuronas en la salida
        nn = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.SIGMOID, epochs=10)

        # Datos sintéticos (XOR problem)
        inputs = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]], dtype=np.float32)
        labels = np.array([[0],
                           [1],
                           [1],
                           [0]], dtype=np.float32)

        # Convertir etiquetas a formato one-hot
        labels_one_hot = np.hstack([1 - labels, labels])  # Shape: (4, 2)

        # Entrenar la red
        nn.train(inputs, labels_one_hot, batch_size=4)

        # Verificar que los pesos han cambiado (no se verifica numéricamente)
        initial_weights = [layer.perceptrons[0].weights.copy() for layer in nn.layers]
        nn.train(inputs, labels_one_hot, batch_size=4)
        for i, layer in enumerate(nn.layers):
            self.assertFalse(np.array_equal(layer.perceptrons[0].weights, initial_weights[i]))

    def test_evaluate(self):
        """Prueba el método evaluate con datos sintéticos."""
        topology = [2, 4, 2]  # Aumentar el número de neuronas en la capa oculta
        nn = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.SIGMOID, epochs=5000,
                           learning_rate=0.1)

        # Datos sintéticos (XOR problem)
        inputs = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]], dtype=np.float32)
        labels = np.array([[0],
                           [1],
                           [1],
                           [0]], dtype=np.float32)

        # Convertir etiquetas a formato one-hot
        labels_one_hot = np.hstack([1 - labels, labels])  # Shape: (4, 2)

        # Entrenar la red
        nn.train(inputs, labels_one_hot, batch_size=4)

        # Evaluar en los mismos datos
        accuracy = nn.evaluate(inputs, labels_one_hot)

        # Verificar que la precisión es alta (mayor al 90%)
        self.assertGreaterEqual(accuracy, 0.9)

    def test_predict(self):
        """Prueba el método predict con una entrada sintética."""
        topology = [2, 3, 2]
        nn = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.SIGMOID)

        # Entrada sintética
        inputs = np.array([[0.5, 0.5]], dtype=np.float32)

        # Realizar predicción
        probabilities = nn.predict(inputs)

        # Verificar que las probabilidades suman aproximadamente 1
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)

        # Verificar que las probabilidades están en el rango [0, 1]
        self.assertTrue(np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0))

    def test_save_and_load_weights(self):
        """Prueba los métodos save_weights y load_weights."""
        topology = [2, 2]
        nn = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.SIGMOID)

        # Configurar pesos conocidos
        for layer in nn.layers:
            for perceptron in layer.perceptrons:
                perceptron.weights = np.array([0.1, -0.2], dtype=np.float32)
                perceptron.bias = np.float32(0.05)

        # Guardar pesos
        weight_file = 'test_weights.npz'
        nn.save_weights(weight_file)

        # Crear una nueva instancia de la red y cargar los pesos
        nn_loaded = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.SIGMOID)
        nn_loaded.load_weights(weight_file)

        # Verificar que los pesos se cargaron correctamente
        for layer_original, layer_loaded in zip(nn.layers, nn_loaded.layers):
            for perceptron_original, perceptron_loaded in zip(layer_original.perceptrons, layer_loaded.perceptrons):
                np.testing.assert_array_almost_equal(perceptron_original.weights, perceptron_loaded.weights)
                self.assertAlmostEqual(perceptron_original.bias, perceptron_loaded.bias, places=6)

        # Limpiar archivo de prueba
        os.remove(weight_file)

    def test_invalid_topology(self):
        """Verifica que se maneja adecuadamente una topología inválida."""
        topology = [5]  # Solo una capa

        with self.assertRaises(ValueError):
            nn = NeuralNetwork(topology=topology)

    def test_train_with_mismatched_labels(self):
        """Prueba el entrenamiento con etiquetas que no coinciden en tamaño."""
        topology = [2, 3, 2]
        nn = NeuralNetwork(topology=topology)

        # Datos sintéticos
        inputs = np.array([[0.1, 0.2],
                           [0.3, 0.4]], dtype=np.float32)
        labels = np.array([[1],
                           [0],
                           [1]], dtype=np.float32)  # Una etiqueta extra

        with self.assertRaises(ValueError):
            nn.train(inputs, labels)

    def test_evaluate_with_mismatched_labels(self):
        """Prueba el método evaluate con etiquetas que no coinciden en tamaño."""
        topology = [2, 3, 2]
        nn = NeuralNetwork(topology=topology)

        # Datos sintéticos
        inputs = np.array([[0.1, 0.2]], dtype=np.float32)
        labels = np.array([[1, 0],
                           [0, 1]], dtype=np.float32)  # Etiquetas extra

        with self.assertRaises(ValueError):
            nn.evaluate(inputs, labels)

    def test_predict_with_invalid_input_dimensions(self):
        """Prueba el método predict con entradas de dimensiones incorrectas."""
        topology = [3, 3, 2]
        nn = NeuralNetwork(topology=topology)

        # Entrada con dimensiones incorrectas
        inputs = np.array([[0.1, 0.2]], dtype=np.float32)  # Solo 2 características en lugar de 3

        with self.assertRaises(ValueError):
            nn.predict(inputs)

    def test_forward_with_invalid_input_dimensions(self):
        """Prueba el método forward con entradas de dimensiones incorrectas."""
        topology = [3, 3, 2]
        nn = NeuralNetwork(topology=topology)

        inputs = np.array([[0.1, 0.2]], dtype=np.float32)  # Solo 2 características en lugar de 3

        with self.assertRaises(ValueError):
            nn.forward(inputs)

    def test_train_with_zero_epochs(self):
        """Prueba el entrenamiento con epochs=0."""
        topology = [2, 2]
        nn = NeuralNetwork(topology=topology, epochs=0)

        # Datos sintéticos
        inputs = np.array([[0.1, 0.2]], dtype=np.float32)
        labels = np.array([[1, 0]], dtype=np.float32)

        # Entrenar la red
        nn.train(inputs, labels)

        # Verificar que los pesos no han cambiado (opcional)

    def test_learning_rate_effect(self):
        """Prueba el efecto de diferentes tasas de aprendizaje en el entrenamiento."""
        topology = [2, 3, 2]  # Cambiado a 2 neuronas en la salida
        learning_rates = [0.0001, 0.01, 1.0]
        inputs = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]], dtype=np.float32)
        labels = np.array([[0],
                           [1],
                           [1],
                           [0]], dtype=np.float32)
        labels_one_hot = np.hstack([1 - labels, labels])

        accuracies = []
        for lr in learning_rates:
            nn = NeuralNetwork(topology=topology, activation_type=ActivationFunctionType.SIGMOID, learning_rate=lr,
                               epochs=1000)
            nn.train(inputs, labels_one_hot, batch_size=4)
            accuracy = nn.evaluate(inputs, labels_one_hot)
            accuracies.append(accuracy)

        # Verificar que una tasa de aprendizaje más adecuada produce mejor precisión
        self.assertGreater(accuracies[1], accuracies[0])  # 0.01 vs 0.0001
        self.assertGreater(accuracies[1], accuracies[2])  # 0.01 vs 1.0


if __name__ == '__main__':
    unittest.main()
