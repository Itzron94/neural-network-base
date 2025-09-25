import numpy as np
from typing import List, Optional
from .network import NeuralNetwork
from .optimizers import OptimizerFunction, OptimizerFunctionFactory
from ..config import OptimizerConfig

class Trainer:
    def __init__(self, learning_rate, epochs, network: NeuralNetwork, loss_func,
                optimizer_config: Optional[OptimizerConfig] = None,):
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.network = network
        self.loss_f = loss_func
        if optimizer_config is None:
            optimizer_config = OptimizerConfig()
        self.optimizer: OptimizerFunction = self._create_optimizer(optimizer_config)

    def train(self, training_inputs: np.ndarray, training_labels: np.ndarray, batch_size: int = 32, verbose=False) -> None:
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

                y = self.network.forward(batch_inputs, training=True)

                if np.isnan(y).any() or np.isinf(y).any():
                    print("Advertencia: y contiene valores NaN o Inf en la época", epoch)
                    continue

                batch_loss, softmax_probs = self.loss_f(y, batch_labels)
                total_loss += batch_loss

                deltas: List[np.ndarray] = []
                delta_output = softmax_probs - batch_labels
                deltas.insert(0, delta_output)

                for l in range(len(self.network.layers) - 2, -1, -1):
                    current_layer = self.network.layers[l]
                    next_layer = self.network.layers[l + 1]
                    weights_next_layer = next_layer.weights[:-1, :] #Excluir el bias
                    #weights_next_layer = np.array([p.weights[:-1] for p in next_layer.perceptrons]) #Excluir el bias
                    delta_next = deltas[0]
                    delta = np.dot(delta_next, weights_next_layer.transpose()) * current_layer.get_activation_derivative()
                    deltas.insert(0, delta)

                all_gradients = []
                for l, layer in enumerate(self.network.layers):
                    inputs_to_use = batch_inputs if l == 0 else self.network.layers[l - 1].outputs
                    delta = deltas[l]
                    # Gradient for weights (excluding bias): shape (input_dim, num_perceptrons)
                    grad_weights = np.dot(inputs_to_use.T, delta) / delta.shape[0]
                    # Gradient for bias: shape (num_perceptrons,)
                    grad_bias = np.mean(delta, axis=0)
                    # Combine gradients: shape (input_dim + 1, num_perceptrons)
                    gradients = np.vstack([grad_weights, grad_bias])
                    # Split gradients for each perceptron in the layer
                    layer_gradients = [gradients[:, i] for i in range(gradients.shape[1])]
                    all_gradients.append(layer_gradients)
                self.optimizer.update_network(self.network, all_gradients, self.learning_rate)
                    
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Época {epoch}/{self.epochs} - Pérdida: {total_loss}")
    def _create_optimizer(self, optimizer_config: OptimizerConfig) -> OptimizerFunction:
        """Create optimizer instance 
        ased on configuration."""
        # Pass all optimizer config parameters, factory will filter what each optimizer needs
        return OptimizerFunctionFactory.create(
            optimizer_config.type,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            momentum=optimizer_config.momentum
        )