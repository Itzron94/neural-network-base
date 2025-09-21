"""
Core components of the neural network framework.
"""

from .network import NeuralNetwork
from .layer import Layer  
from .perceptron import Perceptron

# Re-export activation and optimizer components for convenience
from .activations import ActivationFunction, ActivationFunctionFactory
from .optimizers import OptimizerFunction, OptimizerFunctionFactory

__all__ = [
    "NeuralNetwork", 
    "Layer", 
    "Perceptron",
    "ActivationFunction",
    "ActivationFunctionFactory", 
    "OptimizerFunction",
    "OptimizerFunctionFactory"
]