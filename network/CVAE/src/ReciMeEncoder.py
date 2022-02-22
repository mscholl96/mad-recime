import torch
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
import numpy as np

from typing import Tuple
from dataclasses import dataclass

class _ReLUBatchNormLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_ReLUBatchNormLinear, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


@dataclass
class RmeParameters:
    inputDimension: int
    inputLayer: int = 4096
    reductionLayer_1: int = 2048
    reductionLayer_2: int = 1024
    latentDimension: int = 512

class ReciMeEncoder(nn.Module):
    """ ReciMeEncoder
    Variational Autoencoder specifically adopted for the ReciMe Dataset.

    Provides additional helper functions to preprocess a given dataset or to iterate over the batches.
    """
    def __init__(self, parameters: RmeParameters) -> None:
        super(ReciMeEncoder, self).__init__()
 
        self.encoderStack = nn.Sequential(
            _ReLUBatchNormLinear(parameters.inputDimension, parameters.inputLayer),
            _ReLUBatchNormLinear(parameters.inputLayer, parameters.reductionLayer_1),
            _ReLUBatchNormLinear(parameters.reductionLayer_1, parameters.reductionLayer_2),
            _ReLUBatchNormLinear(parameters.reductionLayer_2, parameters.latentDimension)
        )

        self.muStack = nn.Linear(parameters.latentDimension, parameters.latentDimension)

        self.logvarStack = nn.Linear(parameters.latentDimension, parameters.latentDimension)

        self.decoderStack = nn.Sequential(
            _ReLUBatchNormLinear(parameters.latentDimension, parameters.reductionLayer_2),
            _ReLUBatchNormLinear(parameters.reductionLayer_2, parameters.reductionLayer_1),
            _ReLUBatchNormLinear(parameters.reductionLayer_1, parameters.inputLayer),
            nn.Linear(parameters.inputLayer, parameters.inputDimension),
            nn.Sigmoid()
        )


    def __encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_space = self.encoderStack(x)
        mu = self.muStack(latent_space)
        logvar = self.logvarStack(latent_space)
        return mu, logvar

    def __reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> Variable:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z: Variable) -> torch.Tensor:
        return self.decoderStack(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.__encode(x)
        z = self.__reparametrize(mu, logvar)
        y = self.decode(z)

        return y, mu, logvar