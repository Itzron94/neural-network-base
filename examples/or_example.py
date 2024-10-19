# examples/or_example.py

import numpy as np
from src.neural_network import NeuralNetwork
from src.activations import ActivationFunctionType


def main():
    # Definir la topología de la red: 2 entradas, 2 neuronas en capa oculta, 1 salida
    topology = [2, 2, 1]
    nn = NeuralNetwork(
        topology=topology,
        activation_type=ActivationFunctionType.SIGMOID,
        learning_rate=0.1,
        epochs=10000
    )

    # Datos de entrenamiento para el operador OR lógico
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    training_labels = np.array([
        [0],
        [1],
        [1],
        [1]
    ])

    # Entrenar la red
    nn.train(training_inputs, training_labels, batch_size=4)

    # Guardar los pesos después del entrenamiento
    nn.save_weights("or_perceptron_weights.npz")

    # Realizar predicciones
    print("\nPredicciones después del entrenamiento:")
    for inputs in training_inputs:
        output = nn.predict(inputs)
        print(f"Entrada: {inputs} - Salida: {output.round(3)}")

    # Cargar los pesos guardados en una nueva red
    nn_loaded = NeuralNetwork(
        topology=topology,
        activation_type=ActivationFunctionType.SIGMOID,
        learning_rate=0.1,
        epochs=10000
    )
    nn_loaded.load_weights("or_perceptron_weights.npz")

    # Realizar predicciones con la red cargada
    print("\nPredicciones con la red cargada:")
    for inputs in training_inputs:
        output = nn_loaded.predict(inputs)
        print(f"Entrada: {inputs} - Salida: {output.round(3)}")


if __name__ == "__main__":
    main()