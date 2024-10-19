import numpy as np
from src.neural_network import NeuralNetwork
from src.activations import ReLUActivation, SoftmaxActivation
from src.utils import save_weights, load_weights
import tensorflow as tf


def load_mnist():
    # Cargar el conjunto de datos MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Aplanar las imágenes de 28x28 a 784 dimensiones
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32)
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32)

    # Normalizar los valores de píxeles a [0, 1]
    x_train /= 255.0
    x_test /= 255.0

    # Convertir etiquetas a vectores one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


def main():
    # Cargar y preprocesar los datos MNIST
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Datos de entrenamiento: {x_train.shape}, {y_train.shape}")
    print(f"Datos de prueba: {x_test.shape}, {y_test.shape}")

    # Definir la topología de la red: 784 entradas, 128 neuronas en capa oculta 1, 64 neuronas en capa oculta 2, 10 salidas
    topology = [784, 128, 64, 10]

    # Definir las funciones de activación para cada capa (ReLU para ocultas, Softmax para salida)
    activation_functions = [ReLUActivation(), ReLUActivation(), SoftmaxActivation()]

    nn = NeuralNetwork(
        topology=topology,
        activation_functions=activation_functions,
        learning_rate=0.01,
        epochs=1000
    )

    # Entrenar la red
    nn.train(x_train, y_train, batch_size=64)

    # Evaluar la red en el conjunto de prueba
    accuracy = nn.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

    # Guardar los pesos después del entrenamiento
    nn.save_weights("mnist_mlp_weights.npz")

    # Realizar predicciones
    print("\nPredicciones en el conjunto de prueba:")
    for i in range(10):  # Mostrar las primeras 10 predicciones
        inputs = x_test[i]
        true_label = np.argmax(y_test[i])
        output = nn.predict(inputs)
        predicted_label = np.argmax(output)
        print(f"Entrada: {i + 1} - Verdadera: {true_label} - Predicha: {predicted_label}")

    # Crear una nueva red con la misma topología
    nn_loaded = NeuralNetwork(
        topology=topology,
        activation_functions=activation_functions,
        learning_rate=0.01,
        epochs=1000
    )

    # Cargar los pesos guardados
    nn_loaded.load_weights("mnist_mlp_weights.npz")

    # Evaluar la red cargada en el conjunto de prueba
    loaded_accuracy = nn_loaded.evaluate(x_test, y_test)
    print(f"Precisión de la red cargada en el conjunto de prueba: {loaded_accuracy * 100:.2f}%")

    # Realizar predicciones con la red cargada
    print("\nPredicciones con la red cargada en el conjunto de prueba:")
    for i in range(10):  # Mostrar las primeras 10 predicciones
        inputs = x_test[i]
        true_label = np.argmax(y_test[i])
        output = nn_loaded.predict(inputs)
        predicted_label = np.argmax(output)
        print(f"Entrada: {i + 1} - Verdadera: {true_label} - Predicha: {predicted_label}")


if __name__ == "__main__":
    main()
