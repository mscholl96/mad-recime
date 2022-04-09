import os
import re
import subprocess

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from numpy import average
from sklearn.metrics import accuracy_score
from torch import nn
from transformers import BertModel, DistilBertModel

from .config import create_config

class EmbedLSTM(pl.LightningModule):
    def __init__(self, config={}):
        super().__init__()

        self.config = create_config(config)
        self.save_hyperparameters(self.config)

        self.vocab_size = self.config['vocabSize']
        
        self.batchSize = self.config['batchSize']
        self.hiddenDim = self.config['hiddenDim']
        self.numLayers = self.config['numLayers']
        self.lr        = self.config['lr']

        self.hidden    = None

        self.embed = nn.Embedding(
            self.vocab_size, self.config['embeddingDim'], padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.config['embeddingDim'],
                            hidden_size=self.hiddenDim, num_layers=self.numLayers, batch_first=True)
        self.linear = nn.Linear(self.hiddenDim, self.vocab_size)


        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        embeds = self.embed(x)

        lstm_out, self.hidden = self.lstm(
            embeds, (self.hidden[0].detach(), self.hidden[1].detach()))

        out = self.linear(lstm_out.reshape(-1, self.hiddenDim))

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) #TODO: hyperparams
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def on_train_epoch_start(self):
        self.hidden = self.init_hidden(self.batchSize)

    def on_validation_epoch_start(self):
        self.hidden = self.init_hidden(self.batchSize)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        labels = labels.long().view(-1)

        # loss computation
        loss = self.criterion(outputs, labels)

        self.log('train_loss', loss, on_epoch=True)

        smOut = F.softmax(outputs, dim=1)
        self.train_acc = accuracy_score(torch.argmax(smOut, dim=1).cpu().data.numpy(), labels.cpu().data.numpy())

        self.log('train_acc', self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        labels = labels.long().view(-1)

        # loss computation
        loss = self.criterion(outputs, labels)

        self.log('val_loss', loss, on_epoch=True)

        smOut = F.softmax(outputs, dim=1)
        self.val_acc = accuracy_score(torch.argmax(smOut, dim=1).cpu().data.numpy(), labels.cpu().data.numpy())

        self.log('val_acc', self.val_acc, on_epoch=True)

    def predict(self):
        prediction = 1
        return prediction

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.numLayers, batch_size,
                             self.hiddenDim, device=self.device)
        cell = torch.zeros(self.numLayers, batch_size,
                           self.hiddenDim, device=self.device)
        return hidden, cell