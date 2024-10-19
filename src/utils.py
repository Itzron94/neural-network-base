import numpy as np
import os
import tensorflow as tf


# -------------------------------
# Función para Cargar y Preprocesar MNIST
# -------------------------------
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    y_train = np.eye(10, dtype=np.float32)[y_train]
    y_test = np.eye(10, dtype=np.float32)[y_test]

    return x_train, y_train, x_test, y_test


def shuffle_data(inputs: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Mezcla los datos de entrada y etiquetas de manera aleatoria.

    :param inputs: Array de entradas.
    :param labels: Array de etiquetas.
    :return: Tupla de arrays mezclados (inputs, labels).
    """
    assert len(inputs) == len(labels), "Las entradas y etiquetas deben tener la misma longitud."
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    return inputs[indices], labels[indices]


# def save_weights(neural_network, file_path: str) -> None:
#     data = {}
#     for layer_num, layer in enumerate(neural_network.layers):
#         layer_weights = [perceptron.weights.tolist() for perceptron in layer.perceptrons]
#         layer_biases = [perceptron.bias for perceptron in layer.perceptrons]
#         data[f'layer_{layer_num}_weights'] = layer_weights
#         data[f'layer_{layer_num}_biases'] = layer_biases
#     np.savez(file_path, **data)
#     print(f"Pesos guardados en '{file_path}'.")
#
#
# def load_weights(neural_network, file_path: str) -> None:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
#
#     data = np.load(file_path, allow_pickle=True)
#     num_layers = len(neural_network.layers)
#     for layer_num in range(num_layers):
#         layer_weights = data[f'layer_{layer_num}_weights']
#         layer_biases = data[f'layer_{layer_num}_biases']
#         for perceptron, w, b in zip(neural_network.layers[layer_num].perceptrons, layer_weights, layer_biases):
#             perceptron.weights = np.array(w)
#             perceptron.bias = float(b)
#     print(f"Pesos cargados desde '{file_path}'.")
