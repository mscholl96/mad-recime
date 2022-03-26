### base: https://github.com/yuchenlin/lstm_sentence_classifier/blob/master/LSTM_sentence_classifier.py

### base: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstms-in-pytorch

### base: https://www.analyticsvidhya.com/blog/2020/08/build-a-natural-language-generation-nlg-system-using-pytorch/


import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# hyperparameter tuning
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

from .preProc import getPreProcData


class HyperParams():
    def __init__(self,
                 epochs=10,
                 batchSize=10,
                 lr=1e-3,
                 ratio=[0.7, 0.3],
                 hiddenDim=256,
                 numLayers=1,
                 embeddingDim=300):
        self.epochs = epochs
        self.batchSize = batchSize
        self.lr = lr
        self.ratio = ratio

        self.hiddenDim = hiddenDim  # number of features in hidden state
        self.numLayers = numLayers  # number of stacked lstm layers
        self.embeddingDim = embeddingDim  # embedding dimension

    def __str__(self):
        return('epochs ' + str(self.epochs) + '\n' +
               'batchSize ' + str(self.batchSize) + '\n' +
               'lr ' + str(self.lr) + '\n' +
               'ratio train|val ' + str(self.ratio) + '\n' +
               'hiddenDim ' + str(self.hiddenDim) + '\n' +
               'numLayers ' + str(self.numLayers) + '\n' +
               'embeddingDim ' + str(self.embeddingDim) + '\n')


class TitleDataset(Dataset):
    def __init__(self, datapath):

      data = getPreProcData(datapath, inpRange=range(8))

      self.tokenizer = Tokenizer(oov_token='OOV')

      # dataset split into word sequences required for training
      self.wordSeq = np.vectorize(self.getTitleSequence, otypes=[np.ndarray])(
          data['title'], data['ingredient'])

      # training requires same length sequences -->  padding
      self.maxSequenceLength = max([len(seq['ings']) for seq in self.wordSeq])

      # list of all words in dataset
      self.words = np.concatenate(np.vectorize(self.getCorpus, otypes=[
                                  np.ndarray])(data['title'], data['ingredient']))

      # tokenization corpus
      self.tokenizer.fit_on_texts(self.words)

      # indexed wordSequences (could be calculated in getter but very slow, preprocessing better)
      self.idxWords = np.vectorize(self.getIndexedSeqs, otypes=[
                                   np.ndarray])(self.wordSeq)

      # n gram sequences
      self.movWindSeq = pd.Series(np.vectorize(self.getMovWindSeq, otypes=[
                                  np.ndarray])(self.idxWords)).explode()
      self.movWindSeq.dropna(inplace=True)
      self.movWindSeq = self.movWindSeq.to_numpy()

    def getCorpus(self, title, ingredient):
      titleTok = text_to_word_sequence(title)
      ingTok = text_to_word_sequence(','.join(ingredient))
      return np.array(ingTok + titleTok)

    def getTitleSequence(self, title, ingredient):
      titleTok = text_to_word_sequence(title)
      ingTok = text_to_word_sequence(','.join(ingredient))
      return {'ings': ingTok, 'title': titleTok}

    def getIndexedSeqs(self, seq):
      ingTok = self.tokenizer.texts_to_sequences([seq['ings']])[0]
      ingTok = pad_sequences([ingTok], maxlen=self.maxSequenceLength, padding='pre', value=1)[
          0]
      titleTok = self.tokenizer.texts_to_sequences([seq['title']])[0]

      return {'ings': ingTok, 'title': titleTok}

    def getMovWindSeq(self, seq):
      # input needs to be pre padded
      idxShift = len(seq['title'])
      ingLen = len(seq['ings'])

      fullSeq = np.append(seq['ings'], seq['title'])
      retSeq = np.empty((0, ingLen + 1), dtype=np.int32)

      for i_shift in range(idxShift):
        retSeq = np.vstack(
            [retSeq, np.array(fullSeq[i_shift:ingLen+i_shift+1])])
      return retSeq

    def __len__(self):
        return len(self.idxWords)

    def __getitem__(self, index):
      # tuple of input (ingredients) and label (title)
        return (
            torch.tensor(self.movWindSeq[index][:-1]),
            torch.tensor(self.movWindSeq[index][1:])
        )


## Models
# LSTM Net: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# Embedding Net: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
# Init state: https://stats.stackexchange.com/questions/224737/best-way-to-initialize-lstm-state

class EmbedLSTM(nn.Module):

    def __init__(self, hyperParams, dataset, device):
        super(EmbedLSTM, self).__init__()

        # initialize vital params
        self.vocab_size = len(dataset.tokenizer.word_index)
        self.batchSize = hyperParams.batchSize
        self.hiddenDim = hyperParams.hiddenDim
        self.numLayers = hyperParams.numLayers
        self.device = device

        # embedding definition
        self.word_embeddings = nn.Embedding(
            self.vocab_size, hyperParams.embeddingDim)

        # lstm definition
        self.lstm = nn.LSTM(input_size=hyperParams.embeddingDim,
                            hidden_size=self.hiddenDim, num_layers=self.numLayers, batch_first=True)

        # definition fully connected layer
        self.linear = nn.Linear(self.hiddenDim, self.vocab_size)

    def forward(self, x, hidden):
        embeds = self.word_embeddings(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        out = self.linear(lstm_out.reshape(-1, self.hiddenDim))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return out, hidden

    def init_hidden(self, batchSize=None):
        ''' initializes hidden state '''
        # Create two new tensors with sizes numLayers x batchSize x hiddenDim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        batchSize = self.batchSize if batchSize == None else batchSize

        hidden = (weight.new(self.numLayers, batchSize, self.hiddenDim).zero_().to(self.device),
                  weight.new(self.numLayers, batchSize, self.hiddenDim).zero_().to(self.device))

        return hidden


def train_epoch(epoch, model, criterion, optimizer, train_loader, device, writer):
  running_loss = 0.
  correct = 0
  total = 0

  h = model.init_hidden()

  model.train()

  for batch, (input, target) in enumerate(train_loader):
    if epoch == 0 and batch == 0:
      writer.add_graph(model, input_to_model=(
          input.to(device), h), verbose=False)

    # assign input and target to device
    input, target = input.to(device), target.to(device)

    # detach hidden states
    h = tuple([each.data for each in h])

    # clear gradients
    optimizer.zero_grad()

    # batch prediction (alternative: forward)
    outputs, h = model(input, h)
    target = target.long()

    # loss computation
    loss = criterion(outputs, target.view(-1))

    # calc backward gradients
    loss.backward()

    # run optimizer
    optimizer.step()

    # print statistics
    running_loss += loss.item()

    # _, predicted = outputs.max(1)
    # print(outputs.shape)
    # print(predicted.shape)
    # total += target.size(0)
    # correct += predicted.eq(target).sum().item()

  print("Epoch: %d, loss: %1.5f" % (epoch+1, running_loss / len(train_loader)))
  return(running_loss / len(train_loader))


def val_epoch(epoch, model, criterion, optimizer, val_loader, device, writer):
  # Validation Loss
  correct = 0
  total = 0
  running_loss = 0.0

  h = model.init_hidden()

  model.eval()  # what does it do
  with torch.no_grad():  # what does it do
    for batch, (input, target) in enumerate(val_loader):
      # assign input and target to device
      input, target = input.to(device), target.to(device)

      # detach hidden states
      h = tuple([each.data for each in h])

      # batch prediction (alternative: forward)
      outputs, h = model(input, h)
      target = target.long()

      # loss computation
      loss = criterion(outputs, target.view(-1))

      # _, predicted = torch.max(outputs.data, 1)
      # total += target.size(0)
      # correct += (predicted == target).sum().item()

      running_loss += loss.item()
  # # mean_val_accuracy = (100 * correct / total)
  mean_val_loss = (running_loss)
  # # print('Validation Accuracy: %d %%' % (mean_val_accuracy))
  # print('Validation Loss:'  ,mean_val_loss )
  return(running_loss / len(val_loader))


def train(dataset, model, hyperparams, device):
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  trainWriter = SummaryWriter(
      '/content/drive/MyDrive/runs/titleTrainer/train'.format(timestamp))
  valWriter = SummaryWriter(
      '/content/drive/MyDrive/runs/titleTrainer/validation'.format(timestamp))

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)

  # split data
  train_set, val_set = dataset['train'], dataset['val']

  train_loader = DataLoader(
      train_set, batch_size=hyperparams.batchSize, drop_last=True)
  val_loader = DataLoader(
      val_set, batch_size=hyperparams.batchSize, drop_last=True)
  # further options: shuffle, num_workers

  for epoch in range(hyperparams.epochs):
    trainLoss = train_epoch(epoch, model, criterion,
                            optimizer, train_loader, device, trainWriter)
    valLoss = val_epoch(epoch, model, criterion, optimizer,
                        val_loader, device, valWriter)

    trainWriter.add_scalar('loss', trainLoss, epoch)
    valWriter.add_scalar('loss', valLoss, epoch)

  trainWriter.flush()
  valWriter.flush()


def predict(model, dataset, tkn, h=None):

  # tensor inputs
  x = np.array([[dataset.tokenizer.word_index[tkn]]])
  inputs = torch.from_numpy(x)

  # push to GPU
  inputs = inputs.to(model.device)

  # detach hidden state from history
  h = tuple([each.data for each in h])

  # get the output of the model
  out, h = model(inputs, h)

  # get the token probabilities
  p = F.softmax(out, dim=1).data

  p = p.cpu()

  p = p.numpy()
  p = p.reshape(p.shape[1],)

  # get indices of top 3 values
  top_n_idx = p.argsort()[-3:][::-1]

  # randomly select one of the three indices
  sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]

  # return the encoded value of the predicted char and the hidden state
  return dataset.tokenizer.index_word[sampled_token_index], h


# function to generate text
def sample(model, dataset, size, device, initial):

    model.eval()

    # batch size is 1
    h = model.init_hidden(batchSize=1)

    toks = initial
    title = []

    # predict next token
    for t in initial:
      token, h = predict(model, dataset, t, h)

    toks.append(token)

    title.append(token)

    # predict subsequent tokens
    for i in range(size-1):
        token, h = predict(model, dataset, toks[-1], h)
        toks.append(token)
        title.append(token)

    return ' '.join(title)
