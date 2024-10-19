# main.py

from src.neural_network import NeuralNetwork
from src.activations import ActivationFunctionType
from src.utils import load_mnist

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Datos de entrenamiento: {x_train.shape}, {y_train.shape}")
    print(f"Datos de prueba: {x_test.shape}, {y_test.shape}")

    topology = [784, 128, 64, 10]
    nn = NeuralNetwork(
        topology=topology,
        activation_type=ActivationFunctionType.RELU,
        learning_rate=0.0005,
        epochs=50,
        dropout_rate=0.2  # 20% de Dropout en capas ocultas
    )

    nn.train(x_train, y_train, batch_size=128)

    accuracy = nn.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

    nn.save_weights("mnist_mlp_weights.npz")
