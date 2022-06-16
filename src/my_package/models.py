"""A module for models."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A simple multi-layer perceptron."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize the MLP.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): The input.

        Returns:
            torch.Tensor: The output.
        """
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()
