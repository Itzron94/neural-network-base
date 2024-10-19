import unittest
import numpy as np
from src.loss_functions import cross_entropy_loss


class TestLossFunctions(unittest.TestCase):
    def test_cross_entropy_loss_perfect_prediction(self):
        y_true = np.array([1, 0, 0])
        y_pred = np.array([1, 0, 0])
        loss = cross_entropy_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_cross_entropy_loss_no_prediction(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([1, 0, 0])
        loss = cross_entropy_loss(y_true, y_pred)
        # Lógica: -1*log(0 + epsilon) = -log(epsilon) ≈ 27.631021
        self.assertGreater(loss, 27.0)

    def test_cross_entropy_loss_partial_prediction(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.2, 0.5, 0.3])
        loss = cross_entropy_loss(y_true, y_pred)
        expected = - (0 * np.log(0.2) + 1 * np.log(0.5) + 0 * np.log(0.3))
        self.assertAlmostEqual(loss, -np.log(0.5), places=5)

    def test_cross_entropy_loss_multiple_samples(self):
        y_true = np.array([
            [1, 0],
            [0, 1]
        ])
        y_pred = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        loss = cross_entropy_loss(y_true, y_pred)
        expected = -(1 * np.log(0.9) + 0 * np.log(0.1) + 0 * np.log(0.2) + 1 * np.log(0.8))
        self.assertAlmostEqual(loss, -np.log(0.9) - np.log(0.8), places=5)


if __name__ == '__main__':
    unittest.main()
