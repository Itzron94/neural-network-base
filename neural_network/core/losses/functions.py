# loss_functions.py

import numpy as np


# -------------------------------
# Función de Pérdida: Softmax con Entropía Cruzada
# -------------------------------
def softmax_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> (float, np.ndarray):
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    epsilon = 1e-8
    softmax_probs = np.clip(softmax_probs, epsilon, 1. - epsilon)
    loss = -np.sum(labels * np.log(softmax_probs)) / logits.shape[0]
    return loss, softmax_probs
