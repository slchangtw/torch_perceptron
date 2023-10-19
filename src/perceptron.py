import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in: torch.tensor) -> torch.tensor:
        return torch.sigmoid(self.fc1(x_in)).squeeze()
