import unittest
import numpy as np
from src.neural_network import NeuralNetwork
from src.activations import ReLUActivation, SoftmaxActivation
from src.loss_functions import cross_entropy_loss


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        topology = [2, 2, 2]
        activation_functions = [ReLUActivation(), SoftmaxActivation()]
        self.nn = NeuralNetwork(
            topology=topology,
            activation_functions=activation_functions,
            learning_rate=0.1,
            epochs=10
        )
        # Datos de entrenamiento simple
        self.x_train = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        self.y_train = np.array([
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0]
        ])

    def test_forward(self):
        outputs = self.nn.forward(np.array([1, 1]))
        self.assertEqual(outputs.shape, (2,))

    def test_train(self):
        initial_weights = [perceptron.weights.copy() for layer in self.nn.layers for perceptron in layer.perceptrons]
        self.nn.train(self.x_train, self.y_train, batch_size=2)
        updated_weights = [perceptron.weights for layer in self.nn.layers for perceptron in layer.perceptrons]
        # Verificar que los pesos han cambiado después del entrenamiento
        for initial, updated in zip(initial_weights, updated_weights):
            self.assertFalse(np.array_equal(initial, updated))

    def test_evaluate(self):
        accuracy = self.nn.evaluate(self.x_train, self.y_train)
        self.assertTrue(0.0 <= accuracy <= 1.0)


if __name__ == '__main__':
    unittest.main()
