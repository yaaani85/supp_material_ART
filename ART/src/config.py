from __future__ import annotations
import argparse
import os
import torch
from src.utils import add_defaults_and_override_from_cli, get_model_path_suffix, load_config_yaml


def get_cli_args():
    # Add at the beginning of the function
    EXTERNAL_MODELS = ['complex', 'complex2','complex*', 'nbf']  # Add other external models as needed
    
    # Define CLI to config mapping at the start
    cli_to_config = {
        'prior_init_weights': ('prior', 'init_weights'),
        'prior_optimize_temperature': ('prior', 'optimize_temperature'),
        'prior_alpha': ('prior', 'alpha'),
    }

    parser = argparse.ArgumentParser(
        description='Generative modelling of knowledge graphs')

    # General setup
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training ')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='input batch size for testing/validation ')
    parser.add_argument('--seed', type=int, default=17,
                        metavar='S', help='random seed ')
    parser.add_argument('--model_type', type=str,
                      help='Name of the model type')
    parser.add_argument('--model_path', type=str, help='Name of pretrained model (optional)')
    parser.add_argument('--use_pretrained_model', action='store_true', 
                       help='Load pretrained model from save_path_mrr')
    parser.add_argument('--eval', action='store_true', help='Eval a pretrained model on validation set')
    parser.add_argument('--wandb', action='store_true',
                        help='Log results to wandb (please set up your own server)')
    parser.add_argument('--save_model', action='store_true',
                        help='save model parameters to model_path')

    parser.add_argument('--config', type=str, help='Path to config file')
    # Learning
    parser.add_argument('--max_patience', type=int, help='patience before early stopping')
    parser.add_argument('--lr_patience', type=int,  help='patience before early stopping')

    parser.add_argument('--epochs', type=int, 
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float,
                        help='learning rate ')
    parser.add_argument('--embedding_dimension', type=int,
                        help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--dropout', type=float,

                        help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--factor', type=float)
    parser.add_argument('--prediction_smoothing', type=float)
    parser.add_argument('--label_smoothing', type=float)
    parser.add_argument('--weight_decay', type=float,
                        help='Decay the learning rate by this factor every epoch. Default: 0.995')


    # Transformer
    parser.add_argument('--num_blocks', type=int,
                        help='num blocks transformer (default:1)')
    parser.add_argument('--num_heads', type=int,
                        help='num blocks transformer (default:1)')
    parser.add_argument('--num_neurons', type=int,
                        help='num neurons transformer (default:1)')

    # Convolution
    parser.add_argument('--kernel_size', type=int,
                        help='kernel size of the convolution')

    parser.add_argument('--m', type=int,
                        help='hidden dimension after convolution')

    # TODO number of conv layers

    # Experiments
    parser.add_argument('--experiment', type=str, help='Name of the model type')
    parser.add_argument('--plot', type=str, help='Name of the model type')

    # Add model directory and save path arguments
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                       help='Directory for model checkpoints')


    # Prior 
    parser.add_argument('--prior_alpha', type=int, choices=[0, 1],
                       help='Whether to include p(s) in generative process (1=True, 0=False)')
    parser.add_argument('--prior_init_weights', type=str, default=None, choices=['None', 'uniform', 'frequencies'])
    parser.add_argument('--prior_optimize_temperature', type=bool, default=False)
    parser.add_argument('--prior_fixed_uniform', type=bool, default=True)
    # target
    parser.add_argument('--optimize', type=str, 
                       choices=['validation_loss', 'train_loss', 'train_and_validation_loss', 'mrr'])

    # Parse args and get default args
    args = parser.parse_args()
    default_args = parser.parse_args([])

  
    config = load_config_yaml(args.config)

    # Add defaults and override with CLI args
    config = add_defaults_and_override_from_cli(config, args, default_args, cli_to_config)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {config['device']}")
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])

    
    model_subdir = os.path.join(
            config['model_dir'],
            config['dataset']['class'].lower(),
            config['model_type'],
        )
    os.makedirs(model_subdir, exist_ok=True)


    if not config['model_type'] in EXTERNAL_MODELS:
        suffix = get_model_path_suffix(config)
        print(f"suffix: {suffix}")
        model_subdir = os.path.join(
            model_subdir,
            suffix
        )
    config['model_dir'] = model_subdir

    if config.get('model_path') is None:
        config['model_path'] = os.path.join(model_subdir, 'best_filtered_mrr.pt')

    print(f"Model path: {config['model_path']}")
        
    print(f"config: {config}")
    return config
