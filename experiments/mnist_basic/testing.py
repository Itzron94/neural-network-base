import csv
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_network.core.network import NeuralNetwork
from neural_network.config import OptimizerConfig
from neural_network.core.losses.functions import softmax_cross_entropy_with_logits
from neural_network.core.trainer import Trainer


# === Funciones auxiliares ===
def load_mnist():
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    return X_train, y_train, X_test, y_test

def preprocess_data(X, y):
    num_classes = 10
    y_one_hot = np.zeros((y.size, num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.size), y] = 1.0
    return X, y_one_hot


def main():
    print("Cargando datos de MNIST...")
    X_train, y_train_labels, X_test, y_test_labels = load_mnist()
    X_train, y_train = preprocess_data(X_train, y_train_labels)
    X_test, y_test = preprocess_data(X_test, y_test_labels)

    # === Parámetros generales ===
    topology = [784, 128, 64, 10]
    activation_type = "RELU"
    learning_rate = 0.001
    epochs = 50
    batch_size = 32
    dropout_rate = 0.0

    plots_dir = Path("outputs/mnist_basic/plots"); plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("outputs/mnist_basic/results"); results_dir.mkdir(parents=True, exist_ok=True)

    print("Inicializando la red neuronal...")
    nn = NeuralNetwork(topology, activation_type, dropout_rate)
    tr = Trainer(learning_rate, epochs, nn, softmax_cross_entropy_with_logits, OptimizerConfig("SGD"))

    # === Validación ===
    val_split = 0.1
    split_idx = int((1 - val_split) * len(X_train))
    X_train_main, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_main, y_val = y_train[:split_idx], y_train[split_idx:]

    train_losses, val_losses = tr.train(
        X_train_main, y_train_main,
        batch_size=batch_size, verbose=True,
        x_val=X_val, y_val=y_val,
        patience=10
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Early Stopping - Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / "mnist_early_stopping_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Evaluando la red en el conjunto de prueba...")
    accuracy_before_saving = nn.evaluate(X_test, y_test)
    print(f"Precisión antes de guardar los pesos: {accuracy_before_saving * 100:.2f}%")

    y_pred = np.argmax(nn.forward(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicho"); plt.ylabel("Verdadero")
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "mnist_confusion_matrix.png", dpi=300)
    plt.show()

    class_accuracies = [(cm[i,i] / np.sum(cm[i]) * 100) if np.sum(cm[i]) > 0 else 0 for i in range(10)]
    for i, acc in enumerate(class_accuracies):
        print(f"  Dígito {i}: {acc:.2f}%")
    print(f"\nPrecisión promedio por clase: {np.mean(class_accuracies):.2f}%")

    plt.figure(figsize=(8, 5))
    plt.bar(range(10), class_accuracies, color="skyblue", edgecolor="black")
    plt.xlabel("Dígito"); plt.ylabel("Precisión (%)")
    plt.title("Precisión por Clase")
    plt.ylim(0, 100); plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(plots_dir / "mnist_class_accuracy.png", dpi=300)
    plt.show()

    print("\n=== Comparando optimizadores ===")
    optimizers = [
        {"name": "SGD", "config": OptimizerConfig("SGD")},
        {"name": "SGD + Momentum", "config": OptimizerConfig("SGD_MOMENTUM", momentum=0.9)},
        {"name": "Adam", "config": OptimizerConfig("ADAM")},
    ]

    results = []

    for opt in optimizers:
        print(f"Entrenando con {opt['name']}...")
        net = NeuralNetwork(topology, activation_type, dropout_rate)
        trainer = Trainer(learning_rate, epochs, net, softmax_cross_entropy_with_logits, opt["config"])

        t0 = time.perf_counter()
        train_losses, val_losses = trainer.train(
            X_train_main, y_train_main,
            batch_size=batch_size,
            x_val=X_val, y_val=y_val,
            verbose=False, patience=10
        )
        elapsed = time.perf_counter() - t0

        acc = net.evaluate(X_test, y_test) * 100
        print(f"  → Precisión: {acc:.2f}%, Tiempo: {elapsed:.2f}s")
        results.append({"optimizer": opt["name"], "test_acc": acc, "val_loss": val_losses[-1], "time_sec": elapsed})

    # Bar plot de comparativa
    plt.figure(figsize=(7, 5))
    names = [r["optimizer"] for r in results]
    accs = [r["test_acc"] for r in results]
    plt.bar(names, accs, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
    plt.ylabel("Precisión en test (%)")
    plt.title("Comparación de optimizadores")
    plt.ylim(0, 100); plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "mnist_optimizer_comparison.png", dpi=300)
    plt.show()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"mnist_optimizer_results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "test_acc", "val_loss", "time_sec"])
        writer.writeheader(); writer.writerows(results)

    print(f"\nResultados guardados en: {csv_path}")

    # === Guardar pesos y verificar consistencia ===
    weights_path = Path("weights/mnist_weights.npz")
    weights_path.parent.mkdir(exist_ok=True)
    nn.save_weights(weights_path)
    print(f"Pesos guardados en '{weights_path}'")

    nn_loaded = NeuralNetwork.from_weights_file(weights_path)
    acc_loaded = nn_loaded.evaluate(X_test, y_test)
    print(f"Precisión tras cargar pesos: {acc_loaded * 100:.2f}%")

    if np.isclose(acc_loaded, accuracy_before_saving, atol=1e-6):
        print("✅ Precisión igual antes y después de guardar/cargar.")

if __name__ == '__main__':
    main()
