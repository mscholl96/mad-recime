from typing import Tuple
import torch
import torch.nn as nn
import pandas as pd

class DataBuilder(torch.utils.data.Dataset):
    def __init__(self, dataset, preProcessor) -> None:
        super().__init__()
        self.data = dataset
        self.len = len(dataset)
        self.preProcessor = preProcessor

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data.iloc[index]

    def customCollate(self, batch):
        unzippedBatch = []
        for elem in batch:
            unzippedBatch.append(elem)
        seriesConcat = pd.Series(unzippedBatch, name="ingredients")
        processedData = self.preProcessor.preProcessInput(seriesConcat)
        normalizedData = self.preProcessor.normalizeData(processedData)
        return torch.FloatTensor(normalizedData)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x).sum(axis=1)
        loss_MSE = loss_MSE.mean()
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), axis=1)
        loss_KLD = loss_KLD.mean()

        return loss_MSE, loss_KLD, loss_MSE + loss_KLD


def train(
    epoch: int,
    batch_size: int,
    log_interval: int,
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.DeviceObjType,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    conditional:bool = False,
) -> Tuple[float, float, float]:
    model.train()
    train_loss = 0
    train_loss_MSE = 0
    train_loss_KLD = 0
    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad()
        if conditional:
            cond = trainloader.dataset.preProcessor.getConditional(data).to(device)
            data = data.to(device)
            recon_batch, mu, logvar = model(data, cond)
        else:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
        loss_MSE, loss_KLD, loss = loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        loss_itm = loss.item()
        train_loss += loss_itm
        train_loss_MSE += loss_MSE.item()
        train_loss_KLD += loss_KLD.item()
        optimizer.step()
    if epoch % log_interval == 0:
        print(
            "====> Epoch: {} Average training loss: {:.5f}, MSE: {:.5f}, KLD: {:.5f}".format(
                epoch,
                batch_size * train_loss / len(trainloader.dataset),
                batch_size * train_loss_MSE / len(trainloader.dataset),
                batch_size * train_loss_KLD / len(trainloader.dataset),
            )
        )
    train_losses = batch_size * train_loss / len(trainloader.dataset)
    train_losses_MSE = batch_size * train_loss_MSE / len(trainloader.dataset)
    train_losses_KLD = batch_size * train_loss_KLD / len(trainloader.dataset)
    return (train_losses, train_losses_MSE, train_losses_KLD)


def test(
    epoch: int,
    batch_size: int,
    log_interval: int,
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.DeviceObjType,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    conditional: bool = False,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        test_loss = 0
        test_loss_MSE = 0
        test_loss_KLD = 0
        for batch_idx, data in enumerate(testloader):
            optimizer.zero_grad()
            if conditional:
                cond = testloader.dataset.preProcessor.getConditional(data).to(device)
                data = data.to(device)
                recon_batch, mu, logvar = model(data, cond)
            else:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
            loss_MSE, loss_KLD, loss = loss_fn(recon_batch, data, mu, logvar)
            loss_itm = loss.item()
            test_loss += loss_itm
            test_loss_MSE += loss_MSE.item()
            test_loss_KLD += loss_KLD.item()
        if epoch % log_interval == 0:        
            print('====> Epoch: {} Average test loss: {:.4f}, MSE: {:.4f}, KLD: {:.4f}'.format(
                epoch, batch_size*test_loss / len(testloader.dataset), 
                batch_size*test_loss_MSE / len(testloader.dataset), 
                batch_size*test_loss_KLD / len(testloader.dataset)))
        test_losses = (batch_size*test_loss / len(testloader.dataset))
        test_losses_MSE = (batch_size*test_loss_MSE / len(testloader.dataset))
        test_losses_KLD = (batch_size*test_loss_KLD / len(testloader.dataset))
        return (test_losses, test_losses_MSE, test_losses_KLD)