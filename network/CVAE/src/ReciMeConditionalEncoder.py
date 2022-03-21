import torch
import torch.nn as nn
from torch.autograd import Variable

from typing import List, Tuple

from src.BatchNormLinear import _BatchNormLinear

class ReciMeConditionalEncoder(nn.Module):
    """ ReciMeConditionalEncoder
    Conditional Variational Autoencoder specifically adopted for the ReciMe Dataset.

    Provides additional helper functions to preprocess a given dataset or to iterate over the batches.
    """
    def __init__(self, parameters: List[int], conditionalSize: int, useBatchNorm = True, actFunc = nn.ReLU(), outFunc = nn.Tanh()) -> None:
        super(ReciMeConditionalEncoder, self).__init__()

        encoderList = []
        for index in range(1,len(parameters)):
            if index == 1:
                encoderList.append(_BatchNormLinear(parameters[index-1]+conditionalSize, parameters[index], useBatchNorm, actFunc))
            else:
                encoderList.append(_BatchNormLinear(parameters[index-1], parameters[index], useBatchNorm, actFunc))
        self.encoderStack = nn.Sequential(*encoderList) 

        self.muStack = nn.Linear(parameters[-1], parameters[-1])

        self.logvarStack = nn.Linear(parameters[-1], parameters[-1])

        decoderList = []
        index = 0
        for index in range(len(parameters)-1, 0, -1):
            if index == len(parameters)-1:
                decoderList.append(_BatchNormLinear(parameters[index]+conditionalSize, parameters[index-1], useBatchNorm, actFunc))
            elif index > 1:
                decoderList.append(_BatchNormLinear(parameters[index], parameters[index-1], useBatchNorm, actFunc))
            else:
                decoderList.append(_BatchNormLinear(parameters[index], parameters[index-1], useBatchNorm, outFunc))
        self.decoderStack = nn.Sequential(*decoderList)


    def __encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cond = torch.concat((x,c), dim=1)
        latent_space = self.encoderStack(x_cond)
        mu = self.muStack(latent_space)
        logvar = self.logvarStack(latent_space)
        return mu, logvar

    def __reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z: Variable, c: torch.Tensor) -> torch.Tensor:
        z_cond = torch.concat((z,c), dim=1)
        return self.decoderStack(z_cond)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.__encode(x, c)
        z = self.__reparametrize(mu, logvar)
        y = self.decode(z, c)

        return y, mu, logvar