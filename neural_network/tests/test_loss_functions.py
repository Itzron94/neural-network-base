# test_loss_functions.py

import unittest
import numpy as np
from neural_network.core.losses.functions import softmax_cross_entropy_with_logits


class TestSoftmaxCrossEntropyWithLogits(unittest.TestCase):

    def test_basic_case(self):
        """Prueba con entradas y etiquetas simples."""
        logits = np.array([[1.0, 2.0, 3.0]])
        labels = np.array([[0, 0, 1]])  # Clase correcta es la tercera

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

        # Cálculo manual del softmax
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        expected_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cálculo manual de la pérdida
        epsilon = 1e-8
        expected_loss = -np.sum(labels * np.log(np.clip(expected_softmax, epsilon, 1. - epsilon))) / logits.shape[0]

        # Verificar la pérdida y las probabilidades softmax
        self.assertAlmostEqual(loss, expected_loss, places=7)
        np.testing.assert_array_almost_equal(softmax_probs, expected_softmax)

    def test_batch_inputs(self):
        """Prueba con un batch de entradas y etiquetas."""
        logits = np.array([[2.0, 1.0, 0.1],
                           [0.1, 0.2, 0.7]])
        labels = np.array([[0, 1, 0],
                           [0, 0, 1]])

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

        # Cálculo manual
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        expected_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        epsilon = 1e-8
        expected_loss = -np.sum(labels * np.log(np.clip(expected_softmax, epsilon, 1. - epsilon))) / logits.shape[0]

        # Verificar
        self.assertAlmostEqual(loss, expected_loss, places=7)
        np.testing.assert_array_almost_equal(softmax_probs, expected_softmax)

    def test_extreme_values(self):
        """Prueba con valores extremos en logits para verificar estabilidad numérica."""
        logits = np.array([[1000, 1000, 1000],
                           [-1000, -1000, -1000]])
        labels = np.array([[1, 0, 0],
                           [0, 1, 0]])

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

        # Esperamos que el softmax sea uniforme debido a la estabilidad numérica
        expected_softmax = np.array([[1 / 3, 1 / 3, 1 / 3],
                                     [1 / 3, 1 / 3, 1 / 3]])

        # Verificar que las probabilidades son aproximadamente iguales
        np.testing.assert_allclose(softmax_probs, expected_softmax, atol=1e-5)

        # La pérdida debe ser cercana a -log(1/3)
        expected_loss = -np.mean(np.log(1 / 3))
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_zero_probabilities(self):
        """Prueba para verificar que no hay log(0) en el cálculo de la pérdida."""
        logits = np.array([[0.0, 0.0, 0.0]])
        labels = np.array([[1, 0, 0]])

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

        # Las probabilidades softmax deben ser uniformes
        expected_softmax = np.array([[1 / 3, 1 / 3, 1 / 3]])
        np.testing.assert_array_almost_equal(softmax_probs, expected_softmax)

        # La pérdida debe ser -log(1/3)
        expected_loss = -np.log(1 / 3)
        self.assertAlmostEqual(loss, expected_loss, places=7)

    def test_incorrect_labels_shape(self):
        """Prueba que se lanza una excepción si labels y logits no tienen la misma forma."""
        logits = np.array([[1.0, 2.0, 3.0]])
        labels = np.array([[0, 1]])  # Forma incorrecta

        with self.assertRaises(ValueError):
            loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

    def test_labels_not_one_hot(self):
        """Prueba con etiquetas que no son one-hot."""
        logits = np.array([[1.0, 2.0, 3.0]])
        labels = np.array([2])  # Índice de la clase correcta

        # Convertir etiquetas a one-hot
        labels_one_hot = np.zeros_like(logits)
        labels_one_hot[np.arange(logits.shape[0]), labels] = 1

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels_one_hot)

        # Cálculo manual
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        expected_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        epsilon = 1e-8
        expected_loss = -np.sum(labels_one_hot * np.log(np.clip(expected_softmax, epsilon, 1. - epsilon))) / \
                        logits.shape[0]

        # Verificar
        self.assertAlmostEqual(loss, expected_loss, places=7)
        np.testing.assert_array_almost_equal(softmax_probs, expected_softmax)

    def test_large_number_of_classes(self):
        """Prueba con un gran número de clases."""
        num_classes = 1000
        logits = np.random.randn(1, num_classes)
        labels = np.zeros((1, num_classes))
        labels[0, np.random.randint(0, num_classes)] = 1  # Etiqueta aleatoria

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

        # Verificar que las probabilidades suman 1
        self.assertAlmostEqual(np.sum(softmax_probs), 1.0, places=5)

        # Verificar que la pérdida es finita
        self.assertTrue(np.isfinite(loss))

    def test_gradient_numerical_approximation(self):
        """Comprueba que el gradiente numérico coincide con el gradiente analítico (si se implementa gradiente)."""
        # Nota: Este test es relevante si la función retorna el gradiente con respecto a los logits
        pass  # Implementación del test si el gradiente es calculado



    def test_multi_dimensional_batch(self):
        """Prueba con un batch de múltiples muestras."""
        logits = np.array([[0.2, 0.8],
                           [0.5, 0.5],
                           [0.9, 0.1]])
        labels = np.array([[0, 1],
                           [1, 0],
                           [1, 0]])

        loss, softmax_probs = softmax_cross_entropy_with_logits(logits, labels)

        # Verificar que las probabilidades suman 1 en cada muestra
        sums = np.sum(softmax_probs, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

        # Verificar que la pérdida es finita
        self.assertTrue(np.isfinite(loss))



if __name__ == '__main__':
    unittest.main()
