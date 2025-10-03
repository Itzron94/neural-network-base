"""
TP3 - Exercise 3: Multilayer Perceptron (MLP) for XOR and Parity Discrimination

Two problems in one script:
1. XOR: Classic non-linearly separable logical function
2. Parity Discrimination: Classify digits (0-9) as odd or even using 7x5 binary patterns
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_network.core.network import NeuralNetwork
from neural_network.core.trainer import Trainer
from neural_network.core.losses.mse import mse_loss
from neural_network.config import OptimizerConfig, ConfigLoader
import matplotlib.pyplot as plt


def create_xor_data():
    # Inputs in bipolar format {-1, 1}
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    # Single output: 0 or 1 (normalized for sigmoid)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return X, y

def load_digit_patterns(file_path):
    """
    Load 7x5 digit patterns from file.
    Each digit is represented as 7 rows x 5 columns = 35 binary features.

    Returns:
        X: Array of shape (10, 35) - 10 digits, each with 35 features
        y: Array of shape (10, 1) - Parity labels (0=even, 1=odd)
        digit_labels: List of digit values [0, 1, 2, ..., 9]
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    X = np.array([list(map(int, " ".join(lines[i:i+7]).split())) for i in range(0, len(lines), 7)], dtype=np.float32)
    y = np.array([[i % 2] for i in range(len(X))], dtype=np.float32)
    return X, y, list(range(len(X)))


def print_digit(pattern, digit_label, rows=7, cols=5):
    """Print a digit pattern in a readable format."""
    print(f"\nDigit: {digit_label}")
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = i * cols + j
            row_str += "█" if pattern[idx] == 1 else " "
        print(f"  {row_str}")


def create_optimizer_config(optimizer_type: str, momentum: float = 0.9) -> OptimizerConfig:
    if optimizer_type == "sgd":
        return OptimizerConfig(type="sgd")
    elif optimizer_type == "sgd_momentum":
        return OptimizerConfig(type="sgd_momentum", momentum=momentum)
    elif optimizer_type == "adam":
        return OptimizerConfig(type="adam", beta1=0.9, beta2=0.999, epsilon=1e-8)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def train_mlp(problem_type: str, X, y, topology, learning_rate=0.5, epochs=10000,
              batch_size=None, optimizer_type="sgd_momentum", momentum=0.9,
              log_interval=1000, seed=42, verbose=True, extra_data=None):
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING MLP FOR {problem_type.upper()}")
        print(f"{'='*60}")

    if seed is not None:
        np.random.seed(seed)

    if batch_size is None:
        batch_size = len(X)

    network = NeuralNetwork(
        topology=topology,
        activation_type="SIGMOID",
        dropout_rate=0.0
    )

    optimizer_config = create_optimizer_config(optimizer_type, momentum)

    trainer = Trainer(
        learning_rate=learning_rate,
        epochs=epochs,
        network=network,
        loss_func=mse_loss,
        optimizer_config=optimizer_config
    )

    loss_history = []
    accuracy_history = []

    print("STARTING TRAINING")

    num_samples = X.shape[0]
    for epoch in range(1, epochs + 1):
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
                if verbose:
                    print(f"Warning: NaN or Inf values detected at epoch {epoch}")
                continue

            batch_loss, _ = mse_loss(predictions, batch_labels)
            total_loss += batch_loss

            # Backward pass - calculate deltas
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
            trainer.optimizer.update_network(network, all_gradients, learning_rate)

        # Calculate accuracy
        predictions = network.forward(X, training=False)
        predicted_classes = np.where(predictions > 0.5, 1, 0)
        accuracy = np.mean(predicted_classes == y) * 100

        loss_history.append(total_loss)
        accuracy_history.append(accuracy)

        if verbose and (epoch % log_interval == 0 or epoch == 1):
            print(f"Epoch {epoch:5d}/{epochs} - Loss: {total_loss:.6f} - Accuracy: {accuracy:.2f}%")

        if accuracy == 100.0 and epoch > 100:
            print(f"\n✓ Perfect accuracy achieved at epoch {epoch}!")
            break

    print("-"*60)
    print("TRAINING COMPLETE")
    print("-"*60)

    # Final evaluation
    predictions = network.forward(X, training=False)
    predicted_classes = np.where(predictions > 0.5, 1, 0)
    final_accuracy = np.mean(predicted_classes == y) * 100

    if verbose:
        print(f"\nFinal Accuracy: {final_accuracy:.2f}%")

    return network, loss_history, accuracy_history, final_accuracy


def print_xor_results(network, X, y):
    """Print XOR test results in a readable format."""
    print("\n" + "="*60)
    print("XOR NETWORK PREDICTIONS")
    print("="*60)

    predictions = network.forward(X, training=False)
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


def print_parity_results(network, X, y, digit_labels):
    """Print parity discrimination results in a readable format."""
    print("\n" + "="*60)
    print("PARITY DISCRIMINATION PREDICTIONS")
    print("="*60)

    predictions = network.forward(X, training=False)
    predicted_classes = np.where(predictions > 0.5, 1, 0)

    print(f"\n{'Digit':<8} | {'Expected':<12} | {'Raw Output':<12} | {'Predicted':<12} | {'Correct'}")
    print("-" * 75)

    for i, digit in enumerate(digit_labels):
        expected_parity = "ODD" if y[i, 0] == 1 else "EVEN"
        predicted_parity = "ODD" if predicted_classes[i, 0] == 1 else "EVEN"
        raw_output = predictions[i, 0]
        correct = "✓" if predicted_classes[i, 0] == y[i, 0] else "✗"

        print(f"{digit:^8} | {expected_parity:<12} | {raw_output:^12.4f} | {predicted_parity:<12} | {correct:^8}")

    accuracy = np.mean(predicted_classes == y) * 100
    print("-" * 75)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("="*60 + "\n")


def plot_training_comparison(xor_history, parity_history, output_dir):
    """Plot training history comparison between XOR and Parity problems."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # XOR Loss
    ax1.plot(range(1, len(xor_history['loss']) + 1), xor_history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('XOR - Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # XOR Accuracy
    ax2.plot(range(1, len(xor_history['accuracy']) + 1), xor_history['accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('XOR - Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Parity Loss
    ax3.plot(range(1, len(parity_history['loss']) + 1), parity_history['loss'], 'r-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (MSE)', fontsize=12)
    ax3.set_title('Parity Discrimination - Training Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Parity Accuracy
    ax4.plot(range(1, len(parity_history['accuracy']) + 1), parity_history['accuracy'], 'orange', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Parity Discrimination - Training Accuracy', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining comparison plots saved to: {plot_path}")
    plt.close()


def plot_decision_boundary(network, X, y, output_dir, filename='xor_decision_boundary.png'):
    """Plot the decision boundary learned by the network (for 2D inputs only)."""
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

    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundary plot saved to: {plot_path}")
    plt.close()


def plot_training_history(loss_history, accuracy_history, output_dir, problem_name):
    """Plot training loss and accuracy over epochs for a single problem."""
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'{problem_name} - Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, accuracy_history, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{problem_name} - Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    safe_name = problem_name.lower().replace(' ', '_')
    plot_path = os.path.join(output_dir, f'{safe_name}_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to: {plot_path}")
    plt.close()


def run_experiment_from_config(config_path: str):
    config = ConfigLoader.load_config(config_path)

    print("\n" + "="*60)
    print(f"EXPERIMENT: {config.name.upper()}")
    print("="*60)
    print(f"\nDescription: {config.description}")

    problem_type = config.problem.type
    dataset_path = config.problem.dataset_path
    if dataset_path:
        dataset_path = Path(__file__).parent.parent.parent / dataset_path

    if config.seed is not None:
        np.random.seed(config.seed)

    # Load data based on problem type
    if problem_type == "xor":
        print("\nProblem Type: XOR")
        X, y = create_xor_data()
        extra_data = None
    elif problem_type == "parity":
        print("\nProblem Type: Parity Discrimination")
        if not dataset_path:
            raise ValueError("dataset_path is required for parity problem")
        X, y, digit_labels = load_digit_patterns(str(dataset_path))
        extra_data = {'digit_labels': digit_labels}

        print(f"\nDataset loaded: {len(X)} digits")
        print(f"Each digit: 7x5 = 35 binary features")
        print("\n--- Sample Digits ---")
        for i in [0, 1, 5, 9]:  # Show digits 0, 1, 5, 9
            if i < len(X):
                print_digit(X[i], digit_labels[i])
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


    network, loss_history, accuracy_history, final_accuracy = train_mlp(
        problem_type=problem_type,
        X=X,
        y=y,
        topology=config.network.architecture.topology,
        learning_rate=config.training.learning_rate,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        optimizer_type=config.training.optimizer.type,
        momentum=config.training.optimizer.momentum,
        log_interval=config.training.log_interval,
        seed=config.seed,
        verbose=True,
        extra_data=extra_data
    )

    # Print results
    if problem_type == "xor":
        print_xor_results(network, X, y)
    elif problem_type == "parity":
        print_parity_results(network, X, y, extra_data['digit_labels'])

    if config.metrics.save_plots:
        plots_dir = Path(config.metrics.plots_path)
        plot_training_history(loss_history, accuracy_history, str(plots_dir), config.name)

        # Decision boundary for XOR
        if problem_type == "xor":
            safe_name = config.name.lower().replace(' ', '_')
            plot_decision_boundary(network, X, y, str(plots_dir), filename=f'{safe_name}_decision_boundary.png')

    # Save weights
    if config.training.save_weights:
        weights_dir = Path(config.training.weights_path)
        weights_dir.mkdir(parents=True, exist_ok=True)
        safe_name = config.name.lower().replace(' ', '_')
        weights_file = weights_dir / f"{safe_name}_weights.npz"
        network.save_weights(str(weights_file))

    print("\n" + "="*60)
    print(" EXPERIMENT SUMMARY:")
    print(f"\n Problem: {problem_type.upper()}")
    print(f" Topology: {config.network.architecture.topology}")
    print(f" Final Accuracy: {final_accuracy:.2f}%")
    print(f" Epochs trained: {len(loss_history)}")
    print(f" Status: {'SOLVED ✓' if final_accuracy >= 90.0 else 'NEEDS IMPROVEMENT'}")

    print("\n EXPERIMENT COMPLETE!")
    print("="*60 + "\n")

    return network, loss_history, accuracy_history, final_accuracy


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="TP3 - Exercise 3: MLP for XOR and Parity Discrimination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_tp3.py xor_config.yaml
  python train_tp3.py parity_config.yaml
  python train_tp3.py experiments/exercise_3/xor_config.yaml
        """
    )

    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default=None,
        help='Path to configuration file (YAML). If not provided, runs both problems.'
    )

    args = parser.parse_args()

    if args.config:
        # Single experiment mode
        config_path = Path(args.config)

        # If relative path, try to find it in the exercise_3 directory
        if not config_path.is_absolute():
            if not config_path.exists():
                exercise_dir = Path(__file__).parent
                config_path = exercise_dir / args.config

        if not config_path.exists():
            print(f"❌ Error: Configuration file not found: {config_path}")
            print("\nAvailable configs in exercise_3 directory:")
            exercise_dir = Path(__file__).parent
            for cfg in exercise_dir.glob("*.yaml"):
                print(f"  - {cfg.name}")
            sys.exit(1)

        print(f"\n Loading config: {config_path}")
        run_experiment_from_config(str(config_path))

    else:
        # Legacy mode: Run both problems sequentially
        print("\n" + "="*60)
        print("RUNNING BOTH EXPERIMENTS (LEGACY MODE)")
        print("="*60)
        print("\nTip: Run individual experiments with:")
        print("  python train_tp3.py xor_config.yaml")
        print("  python train_tp3.py parity_config.yaml")

        exercise_dir = Path(__file__).parent

        # Run XOR
        xor_config = exercise_dir / "xor_config.yaml"
        if xor_config.exists():
            run_experiment_from_config(str(xor_config))
        else:
            print("\n⚠️  Warning: xor_config.yaml not found, skipping XOR experiment")

        # Run Parity
        parity_config = exercise_dir / "parity_config.yaml"
        if parity_config.exists():
            run_experiment_from_config(str(parity_config))
        else:
            print("\n⚠️  Warning: parity_config.yaml not found, skipping Parity experiment")


if __name__ == "__main__":
    print("="*60)
    print("TP3 - EJERCICIO 3: MULTILAYER PERCEPTRON (MLP)")
    print("Implementación con NumPy y backpropagation manual")
    print("="*60)

    main()
