# neural_network.py

import numpy as np
import os
from typing import List
from .layer import Layer
from .activations import ActivationFunctionType
from .loss_functions import softmax_cross_entropy_with_logits


class NeuralNetwork:
    def __init__(
            self,
            topology: List[int],
            activation_type: ActivationFunctionType = ActivationFunctionType.SIGMOID,
            learning_rate: float = 0.0005,
            epochs: int = 1000,
            dropout_rate: float = 0.0
    ) -> None:
        if len(topology) < 2:
            raise ValueError("La topología debe tener al menos dos capas (entrada y salida).")

        self.layers: List[Layer] = []
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs

        for i in range(1, len(topology)):
            layer_dropout_rate = dropout_rate if i < len(topology) - 1 else 0.0
            layer = Layer(
                num_perceptrons=topology[i],
                num_inputs_per_perceptron=topology[i - 1],
                activation_type=activation_type,
                dropout_rate=layer_dropout_rate
            )
            self.layers.append(layer)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if inputs.shape[1] != self.layers[0].perceptrons[0].weights.shape[0]:
            raise ValueError("El número de características en las entradas no coincide con el esperado por la red.")
        for layer in self.layers:
            inputs = layer.forward(inputs, training=training)
        return inputs

    def train(self, training_inputs: np.ndarray, training_labels: np.ndarray, batch_size: int = 32) -> None:
        if training_inputs.shape[0] != training_labels.shape[0]:
            raise ValueError("El número de muestras en 'training_inputs' y 'training_labels' debe ser el mismo.")

        num_samples = training_inputs.shape[0]
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            shuffled_inputs = training_inputs[indices]
            shuffled_labels = training_labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]

                # Propagación hacia adelante
                logits = self.forward(batch_inputs, training=True)

                # Verificar valores inválidos en logits
                if np.isnan(logits).any() or np.isinf(logits).any():
                    print("Advertencia: logits contiene valores NaN o Inf en la época", epoch)
                    continue  # O romper el bucle si es necesario

                # Calcular pérdida y probabilidades
                batch_loss, softmax_probs = softmax_cross_entropy_with_logits(logits, batch_labels)
                total_loss += batch_loss

                # Retropropagación del error
                deltas: List[np.ndarray] = []

                delta_output = softmax_probs - batch_labels
                deltas.insert(0, delta_output)

                for l in range(len(self.layers) - 2, -1, -1):
                    current_layer = self.layers[l]
                    next_layer = self.layers[l + 1]
                    weights_next_layer = np.array([p.weights for p in next_layer.perceptrons])

                    delta_next = deltas[0]
                    delta = np.dot(delta_next, weights_next_layer) * current_layer.get_activation_derivative()
                    deltas.insert(0, delta)

                # Actualizar pesos y biases
                for l, layer in enumerate(self.layers):
                    inputs_to_use = batch_inputs if l == 0 else self.layers[l - 1].outputs
                    delta = deltas[l]
                    for i, perceptron in enumerate(layer.perceptrons):
                        perceptron.last_input = inputs_to_use
                        perceptron.update_weights(delta[:, i], self.learning_rate)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Época {epoch}/{self.epochs} - Pérdida: {total_loss}")

    def evaluate(self, test_inputs: np.ndarray, test_labels: np.ndarray) -> float:
        if test_inputs.shape[0] != test_labels.shape[0]:
            raise ValueError("El número de muestras en 'test_inputs' y 'test_labels' debe ser el mismo.")

        logits = self.forward(test_inputs, training=False)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predicted_classes = np.argmax(softmax_probs, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.forward(inputs, training=False)

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits))
        softmax_probs = exp_logits / np.sum(exp_logits)
        return softmax_probs

    def save_weights(self, file_path: str) -> None:
        data = {}
        for layer_num, layer in enumerate(self.layers):
            layer_weights = [perceptron.weights.tolist() for perceptron in layer.perceptrons]
            layer_biases = [perceptron.bias for perceptron in layer.perceptrons]
            data[f'layer_{layer_num}_weights'] = layer_weights
            data[f'layer_{layer_num}_biases'] = layer_biases
        np.savez(file_path, **data)
        print(f"Pesos guardados en '{file_path}'.")

    def load_weights(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

        data = np.load(file_path, allow_pickle=True)
        num_layers = len(self.layers)
        for layer_num in range(num_layers):
            layer_weights = data[f'layer_{layer_num}_weights']
            layer_biases = data[f'layer_{layer_num}_biases']
            for perceptron, w, b in zip(self.layers[layer_num].perceptrons, layer_weights, layer_biases):
                perceptron.weights = np.array(w).astype(np.float32)
                perceptron.bias = np.float32(b)
        print(f"Pesos cargados desde '{file_path}'.")
