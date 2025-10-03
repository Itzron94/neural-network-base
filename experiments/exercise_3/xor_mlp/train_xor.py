"""
Exercise 3 - XOR Logical Function using Multilayer Perceptron
Simple binary classification with Sigmoid activation and MSE loss
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from neural_network.core.network import NeuralNetwork
from neural_network.core.trainer import Trainer
from neural_network.core.losses.mse import mse_loss
from neural_network.config.config_loader import ConfigLoader
import matplotlib.pyplot as plt


def create_xor_data():
    """
    Create data for XOR logical function.
    Using bipolar representation {-1, 1} for inputs.
    Output normalized to {0, 1} to match sigmoid activation range.

    XOR Truth Table:
    Input1 | Input2 | Output (bipolar) | Output (normalized)
    ---------------------------------------------------------
      -1   |   -1   |       -1         |        0
      -1   |    1   |        1         |        1
       1   |   -1   |        1         |        1
       1   |    1   |       -1         |        0
    """
    # Inputs in bipolar format {-1, 1}
    X = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ], dtype=np.float32)

    # Single output: 0 or 1 (normalized for sigmoid)
    y = np.array([
        [0],  # -1 XOR -1 = -1 → 0
        [1],  # -1 XOR  1 =  1 → 1
        [1],  #  1 XOR -1 =  1 → 1
        [0]   #  1 XOR  1 = -1 → 0
    ], dtype=np.float32)

    return X, y


def print_results(network, X, y):
    """Print the test results in a readable format."""
    print("\n" + "="*60)
    print("XOR NETWORK PREDICTIONS")
    print("="*60)

    predictions = network.forward(X, training=False)

    # Convert sigmoid output (0 to 1) to binary (0 or 1)
    # If output > 0.5, predict 1, else predict 0
    predicted_classes = np.where(predictions > 0.5, 1, 0)

    # Convert to bipolar for display
    bipolar_map = {0: -1, 1: 1}

    print(f"\n{'Input':<15} | {'Expected':<10} | {'Raw Output':<12} | {'Predicted':<12} | {'Correct'}")
    print("-" * 85)

    for i in range(len(X)):
        input_str = f"[{X[i, 0]:2.0f}, {X[i, 1]:2.0f}]"
        expected_bipolar = bipolar_map[int(y[i, 0])]
        raw_output = predictions[i, 0]
        predicted_bipolar = bipolar_map[int(predicted_classes[i, 0])]
        correct = "✓" if predicted_classes[i, 0] == y[i, 0] else "✗"

        print(f"{input_str:<15} | {expected_bipolar:^10} | {raw_output:^12.4f} | {predicted_bipolar:^12} | {correct}")

    accuracy = np.mean(predicted_classes == y) * 100
    print("-" * 85)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("="*60 + "\n")


def plot_training_history(loss_history, accuracy_history, output_dir):
    """Plot training loss and accuracy over epochs."""
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, accuracy_history, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'xor_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {plot_path}")
    plt.close()


def plot_decision_boundary(network, X, y, output_dir):
    """Plot the decision boundary learned by the network."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on the mesh
    mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
    Z = network.forward(mesh_inputs, training=False)
    Z = np.where(Z > 0.5, 1, -1)  # Convert to bipolar for visualization
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=[-1, 0, 1])
    plt.colorbar(label='Class', ticks=[-1, 1])

    # Plot training points
    bipolar_map = {0: -1, 1: 1}
    colors = ['red' if y[i, 0] == 1 else 'blue' for i in range(len(y))]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)

    # Add labels for each point
    for i, (x, y_val) in enumerate(X):
        label = str(bipolar_map[int(y[i, 0])])
        plt.annotate(label, (x, y_val), fontsize=14, ha='center', va='center', fontweight='bold', color='white')

    plt.xlabel('Input 1', fontsize=12)
    plt.ylabel('Input 2', fontsize=12)
    plt.title('XOR Decision Boundary', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'xor_decision_boundary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundary plot saved to: {plot_path}")
    plt.close()


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("MLP - XOR VALIDATION (Binary with MSE)")
    print("="*60)

    config_path = Path(__file__).parent / "config.yaml"
    config = ConfigLoader.load_config(str(config_path))

    print(f"\nExperiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Topology: {config.network.architecture.topology}")
    print(f"Activation: {config.network.architecture.activation_type}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Optimizer: {config.training.optimizer.type}")
    print(f"Loss function: Mean Squared Error (MSE)")

    if config.seed is not None:
        np.random.seed(config.seed)
        print(f"Random seed: {config.seed}")

    X, y = create_xor_data()
    print(f"\nDataset size: {len(X)} samples")
    print("\nXOR Truth Table:")
    print("Input1 | Input2 | Output")
    print("-" * 30)
    bipolar_map = {0: -1, 1: 1}
    for i in range(len(X)):
        output_bipolar = bipolar_map[int(y[i, 0])]
        print(f"  {X[i, 0]:2.0f}   |   {X[i, 1]:2.0f}   |   {output_bipolar:2.0f}")

    network = NeuralNetwork(
        topology=config.network.architecture.topology,
        activation_type=config.network.architecture.activation_type,
        dropout_rate=config.network.architecture.dropout_rate
    )

    trainer = Trainer(
        learning_rate=config.training.learning_rate,
        epochs=config.training.epochs,
        network=network,
        loss_func=mse_loss,
        optimizer_config=config.training.optimizer
    )

    loss_history = []
    accuracy_history = []

    print("\n" + "-"*60)
    print("STARTING TRAINING")
    print("-"*60)

    # Custom training loop with metrics tracking
    num_samples = X.shape[0]
    batch_size = config.training.batch_size

    for epoch in range(1, config.training.epochs + 1):
        total_loss = 0.0

        # Shuffle data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_inputs = X[indices]
        shuffled_labels = y[indices]

        # Mini-batch training
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_inputs = shuffled_inputs[start_idx:end_idx]
            batch_labels = shuffled_labels[start_idx:end_idx]

            # Forward pass
            predictions = network.forward(batch_inputs, training=True)

            if np.isnan(predictions).any() or np.isinf(predictions).any():
                print(f"Warning: NaN or Inf values detected at epoch {epoch}")
                continue

            batch_loss, _ = mse_loss(predictions, batch_labels)
            total_loss += batch_loss

            # Backward pass - calculate deltas
            # For MSE loss with sigmoid output: delta = (y_pred - y_true) * sigmoid'(z)
            deltas = []
            delta_output = (predictions - batch_labels) * network.layers[-1].get_activation_derivative()
            deltas.insert(0, delta_output)

            # Backpropagate through hidden layers
            for l in range(len(network.layers) - 2, -1, -1):
                current_layer = network.layers[l]
                next_layer = network.layers[l + 1]
                weights_next_layer = next_layer.weights[:-1, :]
                delta_next = deltas[0]
                delta = np.dot(delta_next, weights_next_layer.T) * current_layer.get_activation_derivative()
                deltas.insert(0, delta)

            # Calculate gradients
            all_gradients = []
            for l, layer in enumerate(network.layers):
                inputs_to_use = batch_inputs if l == 0 else network.layers[l - 1].outputs
                delta = deltas[l]
                grad_weights = np.dot(inputs_to_use.T, delta) / delta.shape[0]
                grad_bias = np.mean(delta, axis=0)
                gradients = np.vstack([grad_weights, grad_bias])
                layer_gradients = [gradients[:, i] for i in range(gradients.shape[1])]
                all_gradients.append(layer_gradients)

            # Update weights using optimizer
            trainer.optimizer.update_network(network, all_gradients, config.training.learning_rate)

        # Calculate accuracy
        predictions = network.forward(X, training=False)
        predicted_classes = np.where(predictions > 0.5, 1, 0)
        accuracy = np.mean(predicted_classes == y) * 100

        # Store metrics
        loss_history.append(total_loss)
        accuracy_history.append(accuracy)

        # Log progress
        if epoch % config.training.log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{config.training.epochs} - Loss: {total_loss:.6f} - Accuracy: {accuracy:.2f}%")

        # Check if converged
        if accuracy == 100.0 and epoch > 100:
            print(f"\n✓ Perfect accuracy achieved at epoch {epoch}!")
            break

    print("-"*60)
    print("TRAINING COMPLETE")
    print("-"*60)

    print_results(network, X, y)

    # Save weights
    if config.training.save_weights:
        weights_dir = Path(config.training.weights_path)
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_file = weights_dir / "xor_mlp_weights.npz"
        network.save_weights(str(weights_file))

    # Save plots
    if config.metrics.save_plots:
        plots_dir = Path(config.metrics.plots_path)
        plot_training_history(loss_history, accuracy_history, str(plots_dir))
        plot_decision_boundary(network, X, y, str(plots_dir))

    print("\n" + "="*60)
    print("VALIDATION COMPLETE - XOR PROBLEM SOLVED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
