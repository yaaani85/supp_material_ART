from src.autoregressive_models.base import AutoRegressiveModel
from src.autoregressive_models.arm_transformer import ARMTransformer
from src.autoregressive_models.arm_convolution import ARMConvolution


def get_arm_model(config: dict, number_of_entities: int, number_of_relations: int) -> AutoRegressiveModel:
    """Create an ARM architecture (Transformer or Convolution) based on config.
    
    Args:
        config: Configuration dictionary
        number_of_entities: Number of entities in the knowledge graph
        number_of_relations: Number of relations 
    """
    match config['model_type']:
        case "arm_transformer":
            return ARMTransformer(
                config['num_blocks'],
                config['embedding_dimension'],
                config['dropout'],
                config['num_heads'],
                config['num_neurons'],
                number_of_entities,
                number_of_relations)

        case "arm_convolution":
            return ARMConvolution(
                config['kernel_size'],
                config['m'],
                config['dropout'],
                number_of_entities,
                number_of_relations,
                config['embedding_dimension'])
        case _:
            raise ValueError(f"Unknown ARM model type: {config['model_type']}")
