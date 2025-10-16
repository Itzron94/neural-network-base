import csv
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import seaborn as sns
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
    optimizer_histories = {}

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

    print("\n=== Comparando optimizadores con varios learning rates===")
    # === Extended Optimizer + Learning Rate Comparison with CSV Summary ===
    print("\n=== Comparing optimizers across multiple learning rates ===")

    optimizers_to_test = [
        {"name": "SGD", "config": OptimizerConfig("SGD")},
        {"name": "SGD + Momentum", "config": OptimizerConfig("SGD_MOMENTUM", momentum=0.9)},
        {"name": "Adam", "config": OptimizerConfig("ADAM")},
    ]

    learning_rates = [0.1, 0.001, 0.0001]
    epochs = 5000
    batch_size = 32
    patience = 50

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    optimizer_histories = []
    summary_rows = []

    for opt in optimizers_to_test:
        for lr in learning_rates:
            label = f"{opt['name']} (lr={lr})"
            print(f"\n--- Training {label} ---")

            nn_opt = NeuralNetwork(topology=topology, activation_type=activation_type, dropout_rate=dropout_rate)
            tr_opt = Trainer(
                learning_rate=lr,
                epochs=epochs,
                network=nn_opt,
                loss_func=softmax_cross_entropy_with_logits,
                optimizer_config=opt["config"]
            )

            start_time = time.perf_counter()
            train_losses, val_losses = tr_opt.train(
                X_train_main, y_train_main,
                batch_size=batch_size,
                verbose=False,
                x_val=X_val,
                y_val=y_val,
                patience=patience
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Compute test accuracy
            test_acc = nn_opt.evaluate(X_test, y_test) * 100

            # Compute accuracy per epoch (train and val)
            train_acc_hist, val_acc_hist = [], []
            for i in range(len(train_losses)):
                preds_train = nn_opt.forward(X_train_main)
                preds_val = nn_opt.forward(X_val)
                train_acc = np.mean(np.argmax(preds_train, axis=1) == np.argmax(y_train_main, axis=1)) * 100
                val_acc = np.mean(np.argmax(preds_val, axis=1) == np.argmax(y_val, axis=1)) * 100
                train_acc_hist.append(train_acc)
                val_acc_hist.append(val_acc)

            optimizer_histories.append({
                "optimizer": opt["name"],
                "lr": lr,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_acc": train_acc_hist,
                "val_acc": val_acc_hist,
                "test_acc": test_acc,
                "best_val_loss": min(val_losses) if len(val_losses) > 0 else None,
                "time_sec": elapsed
            })

            summary_rows.append({
                "Optimizer": opt["name"],
                "Learning Rate": lr,
                "Best Val Loss": min(val_losses) if len(val_losses) > 0 else None,
                "Final Test Accuracy (%)": test_acc,
                "Training Time (s)": round(elapsed, 2)
            })

    # === Create summary CSV ===
    summary_csv = Path("outputs/optimizer_summary.csv")
    with open(summary_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✅ Summary CSV saved at: {summary_csv.resolve()}")

    # === Plot loss curves (all optimizers + all LRs) ===
    plt.figure(figsize=(12, 7))
    for hist in optimizer_histories:
        label = f"{hist['optimizer']} (lr={hist['lr']})"
        plt.plot(hist["train_losses"], label=f"{label} - Train", linestyle='-')
        plt.plot(hist["val_losses"], label=f"{label} - Val", linestyle='--')
    plt.title("Training and Validation Loss vs Epochs (All Optimizers & Learning Rates)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "optimizer_lr_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # === Plot accuracy curves (all optimizers + all LRs) ===
    plt.figure(figsize=(12, 7))
    for hist in optimizer_histories:
        label = f"{hist['optimizer']} (lr={hist['lr']})"
        plt.plot(hist["train_acc"], label=f"{label} - Train", linestyle='-')
        plt.plot(hist["val_acc"], label=f"{label} - Val", linestyle='--')
    plt.title("Training and Validation Accuracy vs Epochs (All Optimizers & Learning Rates)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "optimizer_lr_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # === Plot ONLY lr = 0.1 comparison (for clarity) ===
    plt.figure(figsize=(12, 7))
    for hist in optimizer_histories:
        if hist["lr"] == 0.1:
            label = f"{hist['optimizer']} (lr={hist['lr']})"
            plt.plot(hist["train_losses"], label=f"{label} - Train", linestyle='-')
            plt.plot(hist["val_losses"], label=f"{label} - Val", linestyle='--')
    plt.title("Training and Validation Loss vs Epochs (LR = 0.1 Only)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "optimizer_lr0.1_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    summary_csv = Path("outputs/optimizer_summary.csv")
    with open(summary_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✅ Summary CSV saved at: {summary_csv.resolve()}")

    # === Load summary CSV and create heatmaps ===
    summary_csv = Path("outputs/optimizer_summary.csv")
    df_summary = pd.read_csv(summary_csv)

    # Pivot for heatmap (Optimizers as rows, Learning Rates as columns)
    df_acc = df_summary.pivot(index="Optimizer", columns="Learning Rate", values="Final Test Accuracy (%)")
    df_loss = df_summary.pivot(index="Optimizer", columns="Learning Rate", values="Best Val Loss")

    plt.figure(figsize=(8, 5))
    sns.heatmap(df_acc, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Test Accuracy (%)"})
    plt.title("Optimizer vs Learning Rate — Test Accuracy (%)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Optimizer")
    plt.tight_layout()
    plt.savefig(plots_dir / "optimizer_lr_accuracy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.heatmap(df_loss, annot=True, fmt=".4f", cmap="YlOrRd_r", cbar_kws={"label": "Best Validation Loss"})
    plt.title("Optimizer vs Learning Rate — Best Validation Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Optimizer")
    plt.tight_layout()
    plt.savefig(plots_dir / "optimizer_lr_loss_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

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
