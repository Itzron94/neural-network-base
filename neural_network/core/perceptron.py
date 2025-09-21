# perceptron.py

import numpy as np
from typing import Optional
from .activations import ActivationFunction, ActivationFunctionFactory
from .optimizers import OptimizerFunction, OptimizerFunctionFactory
from ..config import OptimizerConfig, WeightInitConfig

BIAS = 1


class Perceptron:
    def __init__(self, num_inputs: int, activation_type: str, 
                 optimizer_config: Optional[OptimizerConfig] = None,
                 weight_init_config: Optional[WeightInitConfig] = None) -> None:
        if num_inputs <= 0:
            raise ValueError("El número de entradas 'num_inputs' debe ser mayor que cero.")

        # Use default configs if not provided
        if weight_init_config is None:
            weight_init_config = WeightInitConfig()
        if optimizer_config is None:
            optimizer_config = OptimizerConfig()

        self.weights: np.ndarray = self._initialize_weights(num_inputs + BIAS, weight_init_config)
        self.activation: ActivationFunction = ActivationFunctionFactory.create(activation_type)
        self.last_input: np.ndarray = np.array([])
        self.last_total: np.ndarray = np.array([])
        self.last_output: np.ndarray = np.array([])
        self.optimizer: OptimizerFunction = self._create_optimizer(optimizer_config)

    def calculate_output(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[1] != self.weights.shape[0] - 1:
            raise ValueError(f"El número de características en 'inputs' ({inputs.shape[1]}) debe ser {self.weights.shape[0] - 1}.")
        
        bias_column = np.ones((inputs.shape[0], BIAS), dtype=np.float32)
        inputs_with_bias = np.hstack([inputs, bias_column])
        
        self.last_input = inputs_with_bias
        self.last_total = np.dot(inputs_with_bias, self.weights)
        self.last_output = self.activation.activate(self.last_total)
        return self.last_output

    def update_weights(self, delta: np.ndarray, learning_rate: float) -> None:
        if self.last_input.size == 0:
            raise ValueError("El método 'calculate_output' debe ser llamado antes de 'update_weights'.")

        gradients = np.dot(self.last_input.T, delta) / delta.shape[0]
        self.weights = self.optimizer.update(self.weights, gradients, learning_rate)

    def get_activation_derivative(self) -> np.ndarray:
        """Get derivative of the activation function at the last output."""
        if self.last_total.size == 0:
            raise ValueError("calculate_output debe ser llamado antes de get_activation_derivative.")
        return self.activation.derivative(self.last_total)

    def _initialize_weights(self, num_inputs: int, weight_config: WeightInitConfig) -> np.ndarray:
        """Initialize weights based on configuration."""
        if weight_config.seed is not None:
            np.random.seed(weight_config.seed)
        
        init_type = weight_config.init_type.lower()
        
        if init_type == "zeros":
            return np.zeros(num_inputs, dtype=np.float32)
        elif init_type == "ones":
            return np.ones(num_inputs, dtype=np.float32)
        elif init_type == "random":
            return (np.random.randn(num_inputs) * 0.1).astype(np.float32)
        else:
            raise ValueError(f"Tipo de inicialización de pesos no soportado: {init_type}")

    def _create_optimizer(self, optimizer_config: OptimizerConfig) -> OptimizerFunction:
        """Create optimizer instance based on configuration."""
        # Pass all optimizer config parameters, factory will filter what each optimizer needs
        return OptimizerFunctionFactory.create(
            optimizer_config.type,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            momentum=optimizer_config.momentum
        )
