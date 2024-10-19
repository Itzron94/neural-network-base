import numpy as np
from typing import List
from src.layer import Layer
from src.activations import ActivationFunction, SigmoidActivation, ReLUActivation, SoftmaxActivation
from src.loss_functions import cross_entropy_loss
from src.utils import save_weights, load_weights


class NeuralNetwork:
    def __init__(
            self,
            topology: List[int],
            activation_functions: List[ActivationFunction],
            learning_rate: float = 0.01,
            epochs: int = 1000
    ) -> None:
        """
        Inicializa la red neuronal con una topología específica.

        :param topology: Lista que define el número de perceptrones en cada capa.
                         Por ejemplo, [784, 128, 64, 10] representa:
                         - 784 perceptrones en la capa de entrada
                         - 128 en la primera capa oculta
                         - 64 en la segunda capa oculta
                         - 10 en la capa de salida
        :param activation_functions: Lista de funciones de activación para cada capa (excepto la capa de entrada).
        :param learning_rate: Tasa de aprendizaje para la actualización de pesos.
        :param epochs: Número de iteraciones sobre el conjunto de entrenamiento.
        """
        if len(topology) < 2:
            raise ValueError("La topología debe tener al menos dos capas (entrada y salida).")
        if len(activation_functions) != len(topology) - 1:
            raise ValueError(
                "El número de funciones de activación debe ser igual al número de capas ocultas y de salida.")

        self.layers: List[Layer] = []
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs

        # Crear capas ocultas y de salida
        for i in range(1, len(topology)):
            activation = activation_functions[i - 1]
            layer = Layer(
                num_perceptrons=topology[i],
                num_inputs_per_perceptron=topology[i - 1],
                activation=activation
            )
            self.layers.append(layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Realiza la propagación hacia adelante a través de todas las capas.

        :param inputs: Array de entradas.
        :return: Salida final de la red.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, training_inputs: np.ndarray, training_labels: np.ndarray, batch_size: int = 32) -> None:
        """
        Entrena la red neuronal utilizando el algoritmo de retropropagación con Softmax y Entropía Cruzada.
        Implementa entrenamiento por lotes.

        :param training_inputs: Array de entradas de entrenamiento.
        :param training_labels: Array de etiquetas de entrenamiento (one-hot).
        :param batch_size: Tamaño del lote para entrenamiento.
        """
        num_samples = training_inputs.shape[0]
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            # Mezclar los datos para cada época
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            shuffled_inputs = training_inputs[indices]
            shuffled_labels = training_labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]

                # Propagación hacia adelante
                batch_outputs = self.forward(batch_inputs)

                # Calcular pérdida de entropía cruzada
                batch_loss = cross_entropy_loss(batch_labels, batch_outputs)
                total_loss += batch_loss

                # Retropropagación del error
                deltas: List[np.ndarray] = []

                # Delta para la capa de salida (Softmax + Entropía Cruzada)
                delta_output = batch_outputs - batch_labels  # derivada simplificada
                deltas.insert(0, delta_output)

                # Calcular delta para las capas ocultas
                for l in range(len(self.layers) - 2, -1, -1):
                    current_layer = self.layers[l]
                    next_layer = self.layers[l + 1]
                    delta = np.zeros((len(current_layer.perceptrons), batch_size))
                    for i, perceptron in enumerate(current_layer.perceptrons):
                        # Sumar los deltas ponderados de la siguiente capa
                        sum_delta = sum(next_layer.perceptrons[j].weights[i] * deltas[0][j] for j in
                                        range(len(next_layer.perceptrons)))
                        delta[i] = sum_delta * current_layer.perceptrons[i].get_activation_derivative()
                    deltas.insert(0, delta)

                # Actualizar pesos y biases
                for l, layer in enumerate(self.layers):
                    # Usar las entradas del lote para la primera capa, o las salidas de la capa anterior
                    inputs_to_use = batch_inputs if l == 0 else self.layers[l - 1].outputs
                    for j, perceptron in enumerate(layer.perceptrons):
                        # Promediar los deltas sobre el lote
                        perceptron.weights -= self.learning_rate * np.mean(deltas[l][j] * inputs_to_use, axis=0)
                        perceptron.bias -= self.learning_rate * np.mean(deltas[l][j])

            # Opcional: Imprimir la pérdida cada cierta cantidad de épocas
            if epoch % 100 == 0 or epoch == 1:
                print(f"Época {epoch}/{self.epochs} - Pérdida: {total_loss}")

    def evaluate(self, test_inputs: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Evalúa la precisión de la red neuronal en un conjunto de datos de prueba.

        :param test_inputs: Array de entradas de prueba.
        :param test_labels: Array de etiquetas de prueba (one-hot).
        :return: Precisión de la red.
        """
        predictions = self.forward(test_inputs)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Realiza una predicción con la red neuronal.

        :param inputs: Array de entradas.
        :return: Array de salidas de la red.
        """
        return self.forward(inputs)

    def save_weights(self, file_path: str) -> None:
        """
        Guarda los pesos y biases de la red neuronal en un archivo.

        :param file_path: Ruta del archivo donde se guardarán los pesos.
        """
        save_weights(self, file_path)

    def load_weights(self, file_path: str) -> None:
        """
        Carga los pesos y biases de la red neuronal desde un archivo.

        :param file_path: Ruta del archivo desde donde se cargarán los pesos.
        """
        load_weights(self, file_path)
