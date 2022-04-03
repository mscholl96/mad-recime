import torch.nn as nn

class _BatchNormLinear(nn.Module):
    def __init__(self, input_dim, output_dim, useBatchNorm = True, actFunc = nn.ReLU()):
        super(_BatchNormLinear, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        if useBatchNorm:
            layers.append(nn.BatchNorm1d(output_dim))
        if actFunc:
            layers.append(actFunc)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)