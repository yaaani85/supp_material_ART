from abc import ABC, abstractmethod
import torch
from torch import nn

class AutoRegressiveModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()  # type:ignore

    @abstractmethod
    def forward(self, x_tuple: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: document

        Args:
            x (torch.Tensor): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """
        pass