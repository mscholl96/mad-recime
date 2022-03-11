import imp
import torch
import torch.nn as nn
import pandas as pd
from ReciMePreprocessor import ReciMePreprocessor

class DataBuilder(torch.utils.data.Dataset):
    def __init__(self, dataset, preProcessor: ReciMePreprocessor) -> None:
        super().__init__()
        self.data = dataset
        self.len = len(dataset)
        self.preProcessor = preProcessor

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        processedData = self.preProcessor.preProcessInput(pd.Series(self.data.iloc[index]))
        normalizedData = self.preProcessor.normalizeData(processedData)
        return normalizedData[0]


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x).sum(axis=1)
        loss_MSE = loss_MSE.mean()
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), axis=1)
        loss_KLD = loss_KLD.mean()

        return loss_MSE, loss_KLD, loss_MSE + loss_KLD