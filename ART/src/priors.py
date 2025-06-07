from torch import Tensor
from torch.nn import Module, Parameter
import torch
import torch.nn.functional as F

class Prior(Module):
    def __init__(self, 
                 n_entities: int,
                 raw_frequencies: Tensor,
                 init_weights: str = None,
                 optimize_temperature: bool = False,
                 alpha: float = 1.0):  # Add alpha parameter
        """
        Unified prior that can be uniform or frequency-based with optional temperature optimization.
        
        Args:
            n_entities: Number of entities
            raw_frequencies: Entity frequencies
            init_weights: How to initialize weights:
                - None: use fixed frequency-based prior
                - 'uniform': learn from uniform initialization
                - 'frequencies': learn from frequency initialization
            optimize_temperature: Whether to optimize temperature parameter
            alpha: Prior weight in loss function. If 0, uses uniform distribution during evaluation
        """
        super(Prior, self).__init__()
        
        # Set up logits
        if init_weights == 'uniform':
            self.logits = Parameter(torch.ones(n_entities))
        elif init_weights == 'frequencies':
            self.logits = Parameter(torch.log(raw_frequencies.float() + 1e-10))
        else:
            # Always initialize with frequencies when fixed
            self.register_buffer('logits', torch.log(raw_frequencies.float() + 1e-10))
        
        if optimize_temperature:
            self.temperature = Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('temperature', torch.tensor(1.0))
            
        self.alpha = bool(alpha)
        self.register_buffer('uniform_logits', torch.ones(n_entities))

    def forward(self, batch_size: int) -> Tensor:
        # Use uniform distribution if alpha is 0, otherwise use frequency-based prior
        logits = self.logits if self.alpha else self.uniform_logits
        scaled_logits = logits / self.temperature.abs()
        probabilities = F.softmax(scaled_logits, dim=0)
        return probabilities.unsqueeze(0).repeat(batch_size, 1)
