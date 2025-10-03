import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> (float, np.ndarray):
    """
    Computes the Mean Squared Error (MSE) loss.

    Args:
        y_pred: Predicted values, shape (batch_size, output_dim)
        y_true: True values, shape (batch_size, output_dim)

    Returns:
        loss: Scalar MSE loss value
        y_pred: Predictions (unchanged)
    """
    return np.mean((y_pred - y_true) ** 2), y_pred


