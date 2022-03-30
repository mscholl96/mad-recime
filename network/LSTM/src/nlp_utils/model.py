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

from nlp_utils.config import create_config


class EmbedLSTM(pl.LightningModule):
    def __init__(self, config={}): #hyperParams, dataset):

        self.vocab_size = len(dataset.tokenizer.word_index)
        
        self.config = create_config(config)
        self.save_hyperparameters(self.config)

        # hyperparams
        self.batchSize = self.config['batchSize']
        self.hiddenDim = self.config['hiddenDim']
        self.numLayers = self.config['numLayers']
        self.lr        = self.config['lr']

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()


        self.word_embeddings = nn.Embedding(
            self.vocab_size, hyperParams.embeddingDim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=hyperParams.embeddingDim,
                            hidden_size=self.hiddenDim, num_layers=self.numLayers, batch_first=True)
        self.linear = nn.Linear(self.hiddenDim, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embeds = self.word_embeddings(x)

        hidden, cell = self.hidden

        lstm_out, (hidden, cell) = self.lstm(
            embeds, (hidden.detach(), cell.detach()))

        out = self.linear(lstm_out.reshape(-1, self.hiddenDim))

        out = self.softmax(out)

        self.hidden = (hidden, cell)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) #TODO: hyperparams
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def on_epoch_start(self):
        self.hidden = self.init_hidden(self.batchSize)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        labels = labels.long().view(-1)

        # loss computation
        loss = F.cross_entropy(outputs, labels)

        self.log('train_loss', loss, on_epoch=True)

        self.train_acc(torch.topk(outputs, 1)[1].view(-1), labels)
        # acc += accuracy_score(predict(outputs).numpy(),
        #                         labels.numpy())

        self.log('train_acc', self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        labels = labels.long().view(-1)

        # loss computation
        loss = F.cross_entropy(outputs, labels)

        self.log('val_loss', loss, on_epoch=True)

        self.val_acc(torch.topk(outputs, 1)[1].view(-1), labels)
        # acc += accuracy_score(predict(outputs).numpy(),
        #                         labels.numpy())

        self.log('val_acc', self.val_acc, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
    
    def train_dataloader(self):
        return DataLoader(...)

    def val_dataloader(self):
        return DataLoader(...)

    def test_dataloader(self):
        return DataLoader(...)

    def prepare_data(self):
        return 1

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.numLayers, batch_size,
                             self.hiddenDim)
        cell = torch.zeros(self.numLayers, batch_size,
                           self.hiddenDim)
        return hidden, cell