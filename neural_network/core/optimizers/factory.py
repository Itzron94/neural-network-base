"""
Factory for creating optimizer functions.
"""

from enum import Enum
from typing import Dict, Type
from .base import OptimizerFunction
from .sgd import SGDOptimizer
from .sgd_momentum import SGDMomentumOptimizer
from .adam import AdamOptimizer


class OptimizerType(Enum):
    SGD = "SGD"
    SGD_MOMENTUM = "SGD_MOMENTUM"
    ADAM = "ADAM"


class OptimizerFunctionFactory:
    _optimizers: Dict[OptimizerType, Type[OptimizerFunction]] = {
        OptimizerType.SGD: SGDOptimizer,
        OptimizerType.SGD_MOMENTUM: SGDMomentumOptimizer,
        OptimizerType.ADAM: AdamOptimizer,
    }
    
    _string_mapping: Dict[str, OptimizerType] = {
        "SGD": OptimizerType.SGD,
        "SGD_MOMENTUM": OptimizerType.SGD_MOMENTUM,
        "ADAM": OptimizerType.ADAM,
    }
    
    @classmethod
    def create(cls, optimizer_name: str, **kwargs) -> OptimizerFunction:
        optimizer_name_upper = optimizer_name.upper()
        
        if optimizer_name_upper not in cls._string_mapping:
            available = list(cls._string_mapping.keys())
            raise ValueError(f"Optimizer '{optimizer_name}' not supported. Available: {available}")
        
        optimizer_type = cls._string_mapping[optimizer_name_upper]
        optimizer_class = cls._optimizers[optimizer_type]
        
        # Filter kwargs to only include valid parameters for this optimizer
        import inspect
        valid_params = inspect.signature(optimizer_class.__init__).parameters
        # Exclude 'self' parameter
        valid_param_names = {name for name in valid_params.keys() if name != 'self'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_param_names}
        
        return optimizer_class(**filtered_kwargs)
    
    @classmethod
    def create_from_enum(cls, optimizer_type: OptimizerType, **kwargs) -> OptimizerFunction:
        if optimizer_type not in cls._optimizers:
            available = list(cls._optimizers.keys())
            raise ValueError(f"Optimizer type '{optimizer_type}' not supported. Available: {available}")
        
        optimizer_class = cls._optimizers[optimizer_type]
        return optimizer_class(**kwargs)
    
    @classmethod
    def get_available_optimizers(cls) -> list[str]:
        """Get list of available optimizer names."""
        return list(cls._string_mapping.keys())