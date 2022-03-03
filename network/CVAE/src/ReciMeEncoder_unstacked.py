import torch
import torch.nn as nn
from torch.autograd import Variable
from src.ReciMeEncoder import RmeParameters

import pandas as pd
import numpy as np

from typing import Tuple

class ReciMeEncoder_unstacked(nn.Module):
    """ ReciMeEncoder
    Variational Autoencoder specifically adopted for the ReciMe Dataset.

    Provides additional helper functions to preprocess a given dataset or to iterate over the batches.
    """
    def __init__(self, parameters: RmeParameters) -> None:
        super(ReciMeEncoder_unstacked, self).__init__()
 
        self.encoderStack = nn.Sequential(
            nn.Linear(parameters.inputDimension, parameters.inputLayer),
            nn.ReLU(),
            nn.Linear(parameters.inputLayer, parameters.reductionLayer_1),
            nn.ReLU(),
            nn.Linear(parameters.reductionLayer_1, parameters.latentDimension),
            nn.ReLU()
        )

        self.muStack = nn.Linear(parameters.latentDimension, parameters.latentDimension)

        self.logvarStack = nn.Linear(parameters.latentDimension, parameters.latentDimension)

        self.decoderStack = nn.Sequential(
            nn.Linear(parameters.latentDimension, parameters.reductionLayer_1),
            nn.ReLU(),
            nn.Linear(parameters.reductionLayer_1, parameters.inputLayer),
            nn.ReLU(),
            nn.Linear(parameters.inputLayer, parameters.inputDimension),
            nn.Sigmoid()
        )


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_space = self.encoderStack(x)
        mu = self.muStack(latent_space)
        logvar = self.logvarStack(latent_space)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoderStack(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        y = self.decode(z)

        return y, mu, logvar