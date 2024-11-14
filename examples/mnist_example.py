# main.py

import numpy as np
from src.neural_network import NeuralNetwork
from src.activations import ActivationFunctionType
import os


def load_mnist():
    """Carga el conjunto de datos MNIST utilizando tensorflow.keras."""
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizar las imágenes y aplanarlas
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def preprocess_data(X, y):
    """Convierte las etiquetas a formato one-hot."""
    # Las imágenes ya están normalizadas
    # Convertir las etiquetas a formato one-hot
    num_classes = 10
    y_one_hot = np.zeros((y.size, num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.size), y] = 1.0

    return X, y_one_hot


def main():
    # Paso 1: Cargar y preprocesar los datos
    print("Cargando datos de MNIST...")
    X_train, y_train_labels, X_test, y_test_labels = load_mnist()
    X_train, y_train = preprocess_data(X_train, y_train_labels)
    X_test, y_test = preprocess_data(X_test, y_test_labels)

    # Paso 2: Crear e inicializar la red neuronal
    topology = [784, 128, 64, 10]
    activation_type = ActivationFunctionType.RELU
    learning_rate = 0.001
    epochs = 50
    batch_size = 64
    dropout_rate = 0.0

    print("Inicializando la red neuronal...")
    nn = NeuralNetwork(
        topology=topology,
        activation_type=activation_type,
        learning_rate=learning_rate,
        epochs=epochs,
        dropout_rate=dropout_rate
    )

    # Paso 3: Entrenar la red
    print("Entrenando la red neuronal...")
    nn.train(X_train, y_train, batch_size=batch_size)

    # Paso 4: Evaluar la red en el conjunto de prueba
    print("Evaluando la red en el conjunto de prueba...")
    accuracy_before_saving = nn.evaluate(X_test, y_test)
    print(f"Precisión antes de guardar los pesos: {accuracy_before_saving * 100:.2f}%")

    # Paso 5: Guardar los pesos entrenados (incluyendo la topología y activación)
    weight_folder = 'weights'
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    weight_file = os.path.join(weight_folder, 'mnist_weights.npz')
    nn.save_weights(weight_file)
    print(f"Pesos y configuración guardados en '{weight_file}'.")

    # Paso 6: Crear una nueva instancia de la red usando from_weights_file
    print("Creando una nueva instancia de la red neuronal desde el archivo de pesos...")
    nn_loaded = NeuralNetwork.from_weights_file(weight_file)

    # Paso 7: Evaluar la nueva instancia en el conjunto de prueba
    print("Evaluando la nueva instancia de la red en el conjunto de prueba...")
    accuracy_after_loading = nn_loaded.evaluate(X_test, y_test)
    print(f"Precisión después de cargar los pesos: {accuracy_after_loading * 100:.2f}%")

    # Verificar si las precisiones son iguales
    if np.isclose(accuracy_before_saving, accuracy_after_loading, atol=1e-6):
        print("La precisión es la misma antes y después de cargar los pesos.")
    else:
        print("La precisión difiere después de cargar los pesos.")

    # Verificar que las predicciones son iguales
    print("Verificando que las predicciones son iguales...")
    predictions_original = nn.predict(X_test[:10])
    predictions_loaded = nn_loaded.predict(X_test[:10])

    if np.allclose(predictions_original, predictions_loaded, atol=1e-6):
        print("Las predicciones son iguales en ambas redes.")
    else:
        print("Las predicciones difieren entre las redes.")

    # Opcional: Eliminar el archivo de pesos si no se necesita
    # os.remove(weight_file)


if __name__ == '__main__':
    main()
