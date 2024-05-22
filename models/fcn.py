import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):

    def __init__(
        self, input_dim, out_dim, bias=False
    ):
        """
        Args:
            input_dim: The input dimension.
            out_dim: The output dimension.
            bias: True for adding bias.
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn( out_dim, input_dim)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).
        
        Returns:
            An affine transformation of x, tensor of size (batch_size, *, out_dim)
        """
        x = F.linear( x, self.weight, self.bias) / x.size(-1)**.5 # standard scaling
        return x

class Perceptron(nn.Module):
    def __init__(
        self, input_dim, out_dim, norm
    ):
        """
        Perceptron

        Args:
            input_dim: The input dimension.
            out_dim: The output dimension.
            norm: The output normalisation.
        """
        super().__init__()
        self.readout = nn.Parameter(
            torch.randn(input_dim, out_dim)
        )
        self.norm = norm

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).
        
        Returns:
            Output of a perceptron, tensor of size (batch_size, *, out_dim)
        """
        x = x @ self.readout / self.norm
        return x

class MLP(nn.Module):
    def __init__(
        self, input_dim, nn_dim, out_dim, num_layers, bias=False, norm='std'
    ):
        """
        MultiLayer Perceptron

        Args:
            input_dim: The input dimension.
            nn_dim: The number of hidden neurons per layer.
            out_dim: The output dimension.
            num_layers: The number of layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Sequential(
                MyLinear(
                    input_dim, nn_dim, bias
                ),
                nn.ReLU(),
            ),
            *[nn.Sequential(
                    MyLinear(
                        nn_dim, nn_dim, bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.readout = nn.Parameter(
            torch.randn(nn_dim, out_dim)
        )
        if norm=='std':
            self.norm = nn_dim**.5 # standard NTK scaling
        elif norm=='mf':
            self.norm = nn_dim # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).
        
        Returns:
            Output of a multilayer perceptron, tensor of size (batch_size, *, out_dim)
        """
        x = self.hidden(x)
        x = x @ self.readout / self.norm
        return x
