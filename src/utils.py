import numpy as np
import os


def save_weights(neural_network, file_path: str) -> None:
    data = {}
    for layer_num, layer in enumerate(neural_network.layers):
        layer_weights = [perceptron.weights.tolist() for perceptron in layer.perceptrons]
        layer_biases = [perceptron.bias for perceptron in layer.perceptrons]
        data[f'layer_{layer_num}_weights'] = layer_weights
        data[f'layer_{layer_num}_biases'] = layer_biases
    np.savez(file_path, **data)
    print(f"Pesos guardados en '{file_path}'.")


def load_weights(neural_network, file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

    data = np.load(file_path, allow_pickle=True)
    num_layers = len(neural_network.layers)
    for layer_num in range(num_layers):
        layer_weights = data[f'layer_{layer_num}_weights']
        layer_biases = data[f'layer_{layer_num}_biases']
        for perceptron, w, b in zip(neural_network.layers[layer_num].perceptrons, layer_weights, layer_biases):
            perceptron.weights = np.array(w)
            perceptron.bias = float(b)
    print(f"Pesos cargados desde '{file_path}'.")
