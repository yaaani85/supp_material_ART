from experiments.experiment_runner import ExperimentRunner
from experiments.plot import PlotRunner
from src.config import get_cli_args
import torch
import os

def main():
    args = get_cli_args()
    config = vars(args)
    
    # For evaluation, always use pretrained model
    config['use_pretrained_model'] = True
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        if 'plot' in config:
            runner = PlotRunner(config)
            runner.run_plots()
        elif 'experiment' in config:
            runner = ExperimentRunner(config)
            runner.run_experiment()
        else:
            raise ValueError("Neither 'plot' nor 'experiment' specified in config")

if __name__ == '__main__':
    main()
