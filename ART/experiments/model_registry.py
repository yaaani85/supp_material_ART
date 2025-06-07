from dataclasses import dataclass
from typing import Any, Dict, Tuple
from src.models import ARM  # Import ARM class for isinstance check

@dataclass
class ModelOutput:
    model: Any
    dataset: Any
    engine: Any = None

class ModelRegistry:
    _models: Dict[str, Any] = {}

    @classmethod
    def register(cls, model_type: str):
        """Decorator to register model initialization methods"""
        def decorator(func):
            cls._models[model_type] = func
            return func
        return decorator

    @classmethod
    def create_model(cls, config: dict) -> ModelOutput:
        model_type = config['model_type']
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_output = cls._models[model_type](config)
        
        # Only set engine if model is ARM instance
        if isinstance(model_output.model, ARM):
            return model_output
        else:
            # For non-ARM models, explicitly set engine to None
            return ModelOutput(
                model=model_output.model,
                dataset=model_output.dataset,
                engine=None
            )

# Define model initialization methods outside the class
@ModelRegistry.register('complex')
@ModelRegistry.register('complex2')
def init_complex(config: dict) -> ModelOutput:
    from external_models.wrappers.gekcs import GEKCS
    model = GEKCS(config)
    dataset = model.dataset

    if config.get('with_calibration', False):
        from external_models.wrappers.callibration import CalibrationModel
        model = CalibrationModel(model, config)
    
    return ModelOutput(model=model, dataset=dataset)

@ModelRegistry.register('nbf')
def init_nbf(config: dict) -> ModelOutput:
    from external_models.wrappers.nbf import NBF
    model = NBF(config)
    dataset = model.dataset
    
    config['dataset']['n_entities'] = model.n_ent
    config['dataset']['n_relations'] = model.n_relations
    
    return ModelOutput(model=model, dataset=dataset)

@ModelRegistry.register('arm_transformer')
@ModelRegistry.register('arm_convolution')
def init_autoregressive(config: dict) -> ModelOutput:
    from src.engine import Engine
    engine = Engine(config)
    return ModelOutput(
        model=engine.model,
        dataset=engine.dataset,
        engine=engine
    )

