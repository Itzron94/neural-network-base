"""
Configuration loader for neural network experiments.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArchitectureConfig:
    """Configuration for neural network architecture."""
    topology: list[int] = field(default_factory=lambda: [2, 1])
    activation_type: str = "SIGMOID"
    dropout_rate: float = 0.0


@dataclass
class WeightInitConfig:
    """Configuration for weight initialization."""
    init_type: str = "random"  # "random", "zeros", "ones"
    seed: Optional[int] = None


@dataclass
class OptimizerConfig:
    """Configuration for optimizer parameters."""
    type: str = "adam"
    # Adam parameters
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    # SGD momentum parameter
    momentum: float = 0.9


@dataclass
class NetworkConfig:
    """Configuration for the complete network."""
    architecture: ArchitectureConfig = field(default_factory=lambda: ArchitectureConfig(topology=[2, 1]))
    weight_init: WeightInitConfig = field(default_factory=WeightInitConfig)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    learning_rate: float = 0.0005
    epochs: int = 1000
    batch_size: int = 32
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    save_weights: bool = True
    weights_path: str = "outputs/weights/"
    log_interval: int = 10
    early_stopping: bool = False
    patience: int = 50
    validation_split: float = 0.1


@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""
    track_loss: bool = True
    track_accuracy: bool = True
    track_gradients: bool = False
    save_plots: bool = True
    plots_path: str = "outputs/plots/"


@dataclass
class ProblemConfig:
    """Configuration for problem-specific settings."""
    type: str = ""
    dataset_path: Optional[str] = None
    noise_level: float = 0.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    network: NetworkConfig
    description: str = ""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    seed: Optional[int] = None


class ConfigLoader:
    """
    Loads and validates configuration files for neural network experiments.
    
    Supports YAML configuration files with validation and default values.
    """
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            ExperimentConfig object with loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        
        # Validate and create config object
        return ConfigLoader._create_config(config_data)
    
    @staticmethod
    def _create_config(config_data: Dict[str, Any]) -> ExperimentConfig:
        """Create ExperimentConfig from dictionary with validation."""
        
        # Extract main sections
        network_data = config_data.get('network', {})
        training_data = config_data.get('training', {})
        metrics_data = config_data.get('metrics', {})
        problem_data = config_data.get('problem', {})

        # Extract architecture data (with fallback for old format)
        architecture_data = network_data.get('architecture', network_data)
        
        # Validate network topology
        if 'topology' not in architecture_data:
            raise ValueError("Network topology is required in configuration")
        
        topology = architecture_data['topology']
        if not isinstance(topology, list) or len(topology) < 2:
            raise ValueError("Topology must be a list with at least 2 layers")
        
        # Create architecture config
        architecture_config = ArchitectureConfig(
            topology=topology,
            activation_type=architecture_data.get('activation_type', 'SIGMOID'),
            dropout_rate=architecture_data.get('dropout_rate', 0.0)
        )
        
        # Create weight initialization config
        weight_init_data = network_data.get('weight_init', {})
        weight_init_config = WeightInitConfig(
            init_type=weight_init_data.get('init_type', 'random'),
            seed=weight_init_data.get('seed')
        )
        
        # Create network config
        network_config = NetworkConfig(
            architecture=architecture_config,
            weight_init=weight_init_config
        )
        
        # Handle optimizer configuration (moved to training)
        optimizer_data = training_data.get('optimizer', {})
        optimizer_config = OptimizerConfig(
            type=optimizer_data.get('type', 'adam'),
            beta1=optimizer_data.get('beta1', 0.9),
            beta2=optimizer_data.get('beta2', 0.999),
            epsilon=optimizer_data.get('epsilon', 1e-8),
            momentum=optimizer_data.get('momentum', 0.9)
        )
        
        training_config = TrainingConfig(
            learning_rate=training_data.get('learning_rate', 0.0005),
            epochs=training_data.get('epochs', 1000),
            batch_size=training_data.get('batch_size', 32),
            optimizer=optimizer_config,
            save_weights=training_data.get('save_weights', True),
            weights_path=training_data.get('weights_path', 'outputs/weights/'),
            log_interval=training_data.get('log_interval', 10),
            early_stopping=training_data.get('early_stopping', False),
            patience=training_data.get('patience', 50),
            validation_split=training_data.get('validation_split', 0.1)
        )
        
        metrics_config = MetricsConfig(
            track_loss=metrics_data.get('track_loss', True),
            track_accuracy=metrics_data.get('track_accuracy', True),
            track_gradients=metrics_data.get('track_gradients', False),
            save_plots=metrics_data.get('save_plots', True),
            plots_path=metrics_data.get('plots_path', 'outputs/plots/')
        )
        
        problem_config = ProblemConfig(
            type=problem_data.get('type', 'xor'),
            dataset_path=problem_data.get('dataset_path'),
            noise_level=problem_data.get('noise_level', 0.0)
        )

        return ExperimentConfig(
            name=config_data.get('name', 'unnamed_experiment'),
            description=config_data.get('description', ''),
            network=network_config,
            training=training_config,
            metrics=metrics_config,
            problem=problem_config,
            seed=config_data.get('seed')
        )
    
    @staticmethod
    def save_config(config: ExperimentConfig, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: ExperimentConfig object to save
            config_path: Path where to save the configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Convert to dictionary
        config_dict = {
            'name': config.name,
            'description': config.description,
            'seed': config.seed,
            'network': {
                'architecture': {
                    'topology': config.network.architecture.topology,
                    'activation_type': config.network.architecture.activation_type,
                    'dropout_rate': config.network.architecture.dropout_rate
                },
                'weight_init': {
                    'init_type': config.network.weight_init.init_type,
                    'seed': config.network.weight_init.seed
                }
            },
            'training': {
                'learning_rate': config.training.learning_rate,
                'epochs': config.training.epochs,
                'batch_size': config.training.batch_size,
                'optimizer': {
                    'type': config.training.optimizer.type,
                    'beta1': config.training.optimizer.beta1,
                    'beta2': config.training.optimizer.beta2,
                    'epsilon': config.training.optimizer.epsilon,
                    'momentum': config.training.optimizer.momentum
                },
                'save_weights': config.training.save_weights,
                'weights_path': config.training.weights_path,
                'log_interval': config.training.log_interval,
                'early_stopping': config.training.early_stopping,
                'patience': config.training.patience,
                'validation_split': config.training.validation_split
            },
            'metrics': {
                'track_loss': config.metrics.track_loss,
                'track_accuracy': config.metrics.track_accuracy,
                'track_gradients': config.metrics.track_gradients,
                'save_plots': config.metrics.save_plots,
                'plots_path': config.metrics.plots_path
            },
            'problem': {
                'type': config.problem.type,
                'dataset_path': config.problem.dataset_path,
                'noise_level': config.problem.noise_level
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
    
    @staticmethod
    def create_default_config(name: str = "default_experiment") -> ExperimentConfig:
        """Create a default configuration for quick experimentation."""
        return ExperimentConfig(
            name=name,
            description="Default neural network experiment configuration"
        )