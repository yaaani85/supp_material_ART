from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.autoregressive_models.arm import AutoRegressiveModel


class Conv1DPadded(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout_probability):

        super().__init__()
        # attributes:
        self.kernel_size = kernel_size
        dilation = 1
        self.padding = (kernel_size - 1) * dilation

        # module:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=0)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        x = self.conv1d(x)
        x = self.dropout(x)
        return x


class ARMConvolution(AutoRegressiveModel):

    def __init__(self, kernel_size, hidden_dimension, dropout_probability, n_entities, n_relations, embedding_dimension):
        super().__init__()

        print('ARM by JT.')
        self.kernel_size = kernel_size
        M = hidden_dimension

        # Combined network for both relations and entities
        self.combined_net = nn.Sequential(
            Conv1DPadded(in_channels=2, out_channels=M,
                        kernel_size=self.kernel_size, 
                        dropout_probability=dropout_probability),
            nn.LeakyReLU(),
            Conv1DPadded(in_channels=M, out_channels=2,
                        kernel_size=self.kernel_size, 
                        dropout_probability=dropout_probability),
            nn.LeakyReLU())

        self.logits_relations = nn.Sequential(
            nn.Linear(embedding_dimension, n_relations),
            nn.Softmax(dim=-1),
        )

        self.logits_entities = nn.Sequential(
            nn.Linear(embedding_dimension, n_entities),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x_tuple: tuple[torch.Tensor, torch.Tensor]):
        rel_embedding, e1_embedding = x_tuple

        # Combine embeddings as channels
        x = torch.cat((rel_embedding, e1_embedding), dim=1)
        x = self.dropout(x)
        
        # Process both simultaneously
        x = self.combined_net(x)
        
        # Split the output back into relations and entities
        x_relations, x_entities = torch.chunk(x, 2, dim=1)

        # Get predictions
        logits_relations = self.logits_relations(x_relations.squeeze(1))
        logits_entities = self.logits_entities(x_entities.squeeze(1))

        return logits_relations, logits_entities
