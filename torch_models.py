import torch
from torch import nn
import numpy as np

class LinearRegressionModel(nn.Module):
    """Simple Neural Network using PyTorch for Linear Regression problems"""
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

class ClassificationModel(nn.Module):
    """Simple Neural Network using PyTorch for non-linear classification"""
    def __init__(self, input_features, output_features, hidden_units=8) -> None:
        """Initializes multi-class classification model

        Args:
        input_features (int): Number of input features to the model
        output_features (int): Number of output classes from the model
        hidden_units (int): Number of hidden units between layers, default 8

        """
        super().__init__()
        self.model_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model_stack(X)