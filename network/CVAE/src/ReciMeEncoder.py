import torch
import torch.nn as nn
from torch.autograd import Variable

from typing import List, Tuple

from src.BatchNormLinear import _BatchNormLinear

class ReciMeEncoder(nn.Module):
    """ ReciMeEncoder
    Variational Autoencoder specifically adopted for the ReciMe Dataset.
    """
    def __init__(self, parameters: List[int], useBatchNorm = True, actFunc = nn.ReLU(), outFunc = nn.Tanh()) -> None:
        super(ReciMeEncoder, self).__init__()

        encoderList = []
        for index in range(1,len(parameters)):
            encoderList.append(_BatchNormLinear(parameters[index-1], parameters[index], useBatchNorm, actFunc))
        self.encoderStack = nn.Sequential(*encoderList) 

        self.muStack = nn.Linear(parameters[-1], parameters[-1])
        self.logvarStack = nn.Linear(parameters[-1], parameters[-1])

        decoderList = []
        index = 0
        for index in range(len(parameters)-1, 0, -1):
            if index > 1:
                decoderList.append(_BatchNormLinear(parameters[index], parameters[index-1], useBatchNorm, actFunc))
            else:
                decoderList.append(_BatchNormLinear(parameters[index], parameters[index-1], useBatchNorm, outFunc))
        self.decoderStack = nn.Sequential(*decoderList)


    def __encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_space = self.encoderStack(x)
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

    def decode(self, z: Variable) -> torch.Tensor:
        return self.decoderStack(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.__encode(x)
        z = self.__reparametrize(mu, logvar)
        y = self.decode(z)

        return y, mu, logvar