import unittest
import numpy as np
from src.activations import SigmoidActivation, ReLUActivation, SoftmaxActivation


class TestActivations(unittest.TestCase):
    def setUp(self):
        self.sigmoid = SigmoidActivation()
        self.relu = ReLUActivation()
        self.softmax = SoftmaxActivation()

    def test_sigmoid_activation(self):
        x = np.array([-1, 0, 1])
        expected = 1 / (1 + np.exp(-x))
        output = self.sigmoid.activate(x)
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_sigmoid_derivative(self):
        x = -1.0
        expected = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
        output = self.sigmoid.derivative(x)
        self.assertAlmostEqual(output, expected, places=5)

    def test_relu_activation(self):
        x = np.array([-1, 0, 1])
        expected = np.maximum(0, x)
        output = self.relu.activate(x)
        np.testing.assert_array_equal(output, expected)

    def test_relu_derivative(self):
        self.assertEqual(self.relu.derivative(-0.5), 0.0)
        self.assertEqual(self.relu.derivative(0.0), 0.0)
        self.assertEqual(self.relu.derivative(0.5), 1.0)

    def test_softmax_activation(self):
        x = np.array([2.0, 1.0, 0.1])
        exps = np.exp(x - np.max(x))
        expected = exps / np.sum(exps)
        output = self.softmax.activate(x)
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_softmax_activation_large_values(self):
        x = np.array([1000, 1001, 1002])
        output = self.softmax.activate(x)
        expected = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_softmax_derivative(self):
        # Como la derivada de Softmax se maneja con la entropía cruzada, esta prueba solo verifica el valor placeholder
        self.assertEqual(self.softmax.derivative(1.0), 1.0)


if __name__ == '__main__':
    unittest.main()
