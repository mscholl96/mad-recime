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

from sklearn.metrics import accuracy_score


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
    def __init__(self, datapath, setSize=8):

      data = getPreProcData(datapath, inpRange=range(setSize))

      self.tokenizer = Tokenizer(oov_token='OOV')

      # dataset split into word sequences required for training
      self.wordSeq = np.vectorize(self.getSequence, otypes=[np.ndarray])(
          data['ingredient'], data['title'])

      # training requires same length sequences -->  padding
      self.maxSequenceLength = max([len(seq['ings']) for seq in self.wordSeq])

      # list of all words in dataset
      self.words = np.concatenate(np.vectorize(self.getCorpus, otypes=[
                                  np.ndarray])(data['ingredient'], data['title']))

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

    def getCorpus(self, ingredient, title):
      ingTok = text_to_word_sequence(' '.join(ingredient))
      titleTok = text_to_word_sequence(title)
      return np.array(ingTok + titleTok)

    def getSequence(self, ingredient, title):
      ingTok = text_to_word_sequence(' '.join(ingredient))
      titleTok = text_to_word_sequence(title)
      return {'ings': ingTok, 'title': titleTok}

    def getIndexedSeqs(self, seq):
      ingTok = self.tokenizer.texts_to_sequences([seq['ings']])[0]
      ingTok = pad_sequences([ingTok], maxlen=self.maxSequenceLength, padding='pre')[
          0] # optional value=1
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
            self.vocab_size, hyperParams.embeddingDim, padding_idx=0)
        # lstm definition
        self.lstm = nn.LSTM(input_size=hyperParams.embeddingDim,
                            hidden_size=self.hiddenDim, num_layers=self.numLayers, batch_first=True)

        # definition fully connected layer
        self.linear = nn.Linear(self.hiddenDim, self.vocab_size)

    def forward(self, x, hidden, cell):
        embeds = self.word_embeddings(x)

        lstm_out, (hidden, cell) = self.lstm(
            embeds, (hidden.detach(), cell.detach()))

        out = self.linear(lstm_out.reshape(-1, self.hiddenDim))

        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.numLayers, batch_size,
                             self.hiddenDim).to(self.device)
        cell = torch.zeros(self.numLayers, batch_size,
                           self.hiddenDim).to(self.device)
        return hidden, cell


class EmbedGRU(nn.Module):

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
        self.gru = nn.GRU(input_size=hyperParams.embeddingDim,
                            hidden_size=self.hiddenDim, num_layers=self.numLayers, batch_first=True)

        # definition fully connected layer
        self.linear = nn.Linear(self.hiddenDim, self.vocab_size)

    def forward(self, x, hidden):
        embeds = self.word_embeddings(x)

        gru_out, hidden = self.lstm(embeds, hidden)

        out = self.linear(gru_out.reshape(-1, self.hiddenDim))

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.numLayers, batch_size,
                             self.hiddenDim).to(self.device)
        cell = torch.zeros(self.numLayers, batch_size,
                           self.hiddenDim).to(self.device)
        return hidden, cell


def train_epoch(epoch, batchSize, model, criterion, optimizer, train_loader, device, writer):
  running_loss = 0.
  accuracy = 0.

  h, c = model.init_hidden(batchSize)

  model.train()

  for batch, (inputs, labels) in enumerate(train_loader):
    if epoch == 0 and batch == 0:
      writer.add_graph(model, input_to_model=(
          inputs.to(device), h, c), verbose=False)

    # assign inputs and labels to device
    inputs, labels = inputs.to(device), labels.to(device)

    # clear gradients
    optimizer.zero_grad()

    # batch prediction
    outputs, (h, c) = model(inputs, h, c)
    labels = labels.long()
    labels = labels.view(-1)

    # loss computation
    loss = criterion(outputs, labels)

    outputPred = outPredict(outputs)
    # print('Out {}, OutPred {}, Lable {}'.format(outputs.shape, outputPred.shape, labels.shape))

    accuracy += accuracy_score(outputPred.cpu().data.numpy(), labels.cpu().data.numpy())

    # calc backward gradients
    loss.backward()

    # run optimizer
    optimizer.step()

    # print statistics
    running_loss += loss.item()

  return(running_loss / len(train_loader), accuracy / len(train_loader))

def val_epoch(epoch, batchSize, model, criterion, optimizer, val_loader, device, writer):
  # Validation Loss
  running_loss = 0.
  accuracy = 0.

  h, c = model.init_hidden(batchSize)

  model.eval()  # what does it do
  with torch.no_grad():  # what does it do
    for batch, (inputs, labels) in enumerate(val_loader):
      # assign inputs and labels to device
      inputs, labels = inputs.to(device), labels.to(device)

      # batch prediction (alternative: forward)
      outputs, (h, c) = model(inputs, h, c)
      labels = labels.long()
      labels = labels.view(-1) # flatten labels to batchSize * seqLength

      # loss computation
      loss = criterion(outputs, labels)

      outputPred = outPredict(outputs)

      accuracy += accuracy_score(outputPred.cpu().data.numpy(), labels.cpu().data.numpy())


      running_loss += loss.item()
  return(running_loss / len(val_loader), accuracy / len(val_loader))


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

  for epoch in range(hyperparams.epochs):
    trainLoss, trainAcc = train_epoch(epoch, hyperparams.batchSize, model,
                            criterion, optimizer, train_loader, device, trainWriter)
    valLoss, valAcc = val_epoch(epoch, hyperparams.batchSize, model,
                        criterion, optimizer, val_loader, device, valWriter)

    print("Epoch: {}, loss: {}, acc: {}".format(epoch+1, trainLoss, trainAcc))
    trainWriter.add_scalar('loss', trainLoss, epoch)
    valWriter.add_scalar('loss', valLoss, epoch)

  trainWriter.flush()
  valWriter.flush()



def predict(model, token, h, c):

  # tensor inputs
  x = np.array([[token]])
  inputs = torch.from_numpy(x)

  # push to GPU
  inputs = inputs.to(model.device)

  # get the output of the model
  out, (h, c) = model(inputs, h, c)

  sampledIdx = outPredict(out, 1).item()
  print(sampledIdx)

  # return the encoded value of the predicted char and the hidden state
  return sampledIdx, (h, c)


def outPredict(modelOutput, n=1):
  # modelOutput is supposed to be a tensor of size [batchSize * seqLength, vocabSize]

  # token probabilities
  smOutput = F.softmax(modelOutput, dim=1).data

  # return idx with highest probability
  idxPred = torch.topk(smOutput, n)[1]

  # reshape tensor to [batchSize * seqLength]
  idxPred = idxPred.view(-1)

  return idxPred
  # predict = predict.cpu()
  # predict = predict.numpy()
  # predict = predict.squeeze()

  # print('out shape {}, predict shape {}'.format(modelOutput.shape, predict.shape))

  # randomly select out of highest n probabilities
  # idxPredict = np.random.choice(np.argpartition(predict, -n)[-n:])
  # print('Top1Idx {} TopIdx {}'.format(np.argmax(predict), idxPredict))

  # return idxPredict



# function to generate text
def sample(model, dataset, size, device, initial):

    # push to GPU
    model.to(device)

    model.eval()

    # batch size is 1
    h, c = model.init_hidden(1)

    toks = initial
    title = []

    # predict next token
    for t in initial:
      token, (h, c) = predict(model, t, h, c)

    toks.append(token)
    title.append(token)

    # predict subsequent tokens
    for i in range(size-1):
        token, (h, c) = predict(model, toks[-1], h, c)
        toks.append(token)
        title.append(token)

    return dataset.tokenizer.sequences_to_texts([title])[0]

def testInputPrep(ingList, tokenizer):
  ingList = tokenizer.texts_to_sequences([' '.join(ingList)])[0]

  return(ingList)
