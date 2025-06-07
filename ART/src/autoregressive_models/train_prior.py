from src.config import get_cli_args
from src.engine import Engine
import os

def main():
    config = get_cli_args()
    
    # Check for pretrained model similar to experiment.py
    if not os.path.exists(config['model_path']):
        raise ValueError("No pretrained model found, train a model first")
    
    # Define prior path similar to model_path in config.py
    config['prior_path'] = f"./src/saved_models/{config['dataset']['class'].lower()}/{config['model_type']}_prior.model"
    
    engine = Engine(config)
    engine.train_prior()

if __name__ == '__main__':
    main()