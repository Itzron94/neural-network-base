"""
Loss functions for neural networks.
"""

from .functions import softmax_cross_entropy_with_logits
from .mse import mse_loss

__all__ = [
    'softmax_cross_entropy_with_logits',
    'mse_loss'
]
