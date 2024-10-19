import numpy as np


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred))
