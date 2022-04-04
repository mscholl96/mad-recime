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

# Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from .preProc import getPreProcData

from sklearn.metrics import accuracy_score

import yaml

#%%
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

    def setHyperParams(self, yamlPath):
      with open(yamlPath, 'r') as ymlFile:
        data = yaml.safe_load(ymlFile)

        self.epochs = data['epochs']
        self.batchSize = data['batchSize']
        self.lr = data['lr']
        self.ratio = data['ratio']
        self.hiddenDim = data['hiddenDim']
        self.numLayers = data['numLayers']
        self.embeddingDim = data['embeddingDim'] 

    def __str__(self):
        return('epochs ' + str(self.epochs) + '\n' +
               'batchSize ' + str(self.batchSize) + '\n' +
               'lr ' + str(self.lr) + '\n' +
               'ratio train|val ' + str(self.ratio) + '\n' +
               'hiddenDim ' + str(self.hiddenDim) + '\n' +
               'numLayers ' + str(self.numLayers) + '\n' +
               'embeddingDim ' + str(self.embeddingDim) + '\n')
#%%

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

class EmbedLSTM(nn.Module):

    def __init__(self, hyperParams, tokenizer, device):
        super(EmbedLSTM, self).__init__()

        # initialize vital params
        self.vocab_size = len(tokenizer.word_index)
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

    def forward(self, x, hidden):
        embeds = self.word_embeddings(x)

        lstm_out, hidden = self.lstm(
            embeds, (hidden[0].detach(), hidden[1].detach()))

        out = self.linear(lstm_out.reshape(-1, self.hiddenDim))

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.numLayers, batch_size,
                             self.hiddenDim).to(self.device)
        cell = torch.zeros(self.numLayers, batch_size,
                           self.hiddenDim).to(self.device)
        return (hidden, cell)


def train_epoch(epoch, batchSize, model, optimizer, train_loader, device, writer):
  running_loss = 0.
  accuracy = 0.

  hidden = model.init_hidden(batchSize)

  model.train()

  for batch, (inputs, labels) in enumerate(train_loader):
    if epoch == 0 and batch == 0:
      writer.add_graph(model, input_to_model=(
          inputs.to(device), hidden), verbose=False)

    # assign inputs and labels to device
    inputs, labels = inputs.to(device), labels.to(device)

    # clear gradients
    optimizer.zero_grad()

    # batch prediction
    outputs, hidden = model(inputs, hidden)
    labels = labels.long().view(-1)

    # loss computation
    criterion =  nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    outputs = F.softmax(outputs,dim=1)
    outputPred = torch.argmax(outputs, dim=1)

    accuracy += accuracy_score(outputPred.cpu().data.numpy(), labels.cpu().data.numpy())

    # calc backward gradients
    loss.backward()

    # run optimizer
    optimizer.step()

    # print statistics
    running_loss += loss.item()

  return(running_loss / len(train_loader), accuracy / len(train_loader))

def val_epoch(batchSize, model, val_loader, device):
  # Validation Loss
  running_loss = 0.
  accuracy = 0.

  hidden = model.init_hidden(batchSize)

  model.eval()  # what does it do
  with torch.no_grad():  # what does it do
    for batch, (inputs, labels) in enumerate(val_loader):
      # assign inputs and labels to device
      inputs, labels = inputs.to(device), labels.to(device)

      outputs, hidden = model(inputs, hidden)
      labels = labels.long().view(-1) # flatten labels to batchSize * seqLength

      # loss computation
      criterion =  nn.CrossEntropyLoss()
      loss = criterion(outputs, labels)
      
      outputs = F.softmax(outputs,dim=1)
      outputPred = torch.argmax(outputs, dim=1)

      accuracy += accuracy_score(outputPred.cpu().data.numpy(), labels.cpu().data.numpy())

      running_loss += loss.item()
  return(running_loss / len(val_loader), accuracy / len(val_loader))


def train(dataset, model, hyperparams, device, logDir):
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  trainWriter = SummaryWriter(
      logDir + 'train'.format(timestamp))
  valWriter = SummaryWriter(
      logDir + 'validation'.format(timestamp))

  optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)

  # split data set
  trainNum = int(hyperparams.ratio[0] * len(dataset))
  valNum = len(dataset) - trainNum

  train_set, val_set = random_split(
        dataset, [trainNum, valNum], generator=torch.Generator().manual_seed(0))

  train_loader = DataLoader(
      train_set, batch_size=hyperparams.batchSize, drop_last=True)
  val_loader = DataLoader(
      val_set, batch_size=hyperparams.batchSize, drop_last=True)

  for epoch in range(hyperparams.epochs):
    trainLoss, trainAcc = train_epoch(epoch, hyperparams.batchSize, model,
                            optimizer, train_loader, device, trainWriter)
    valLoss, valAcc = val_epoch(hyperparams.batchSize, model,
                        val_loader, device)

    print("Epoch: {}, loss: {:10.5f}, acc: {:10.5f}".format(epoch+1, trainLoss, trainAcc))
    trainWriter.add_scalar('loss', trainLoss, epoch)
    trainWriter.add_scalar('acc', trainAcc, epoch)
    valWriter.add_scalar('loss', valLoss, epoch)
    valWriter.add_scalar('acc', valAcc, epoch)

  trainWriter.flush()
  valWriter.flush()



def predict(model, token, hidden):

  # tensor inputs
  x = np.array([[token]])
  inputs = torch.from_numpy(x)

  # push to GPU
  inputs = inputs.to(model.device)

  # get the output of the model
  out, hidden = model(inputs, hidden)

  out = F.softmax(out, dim=1)
  sampledIdx = torch.argmax(out, dim=1).item()

  # return the encoded value of the predicted char and the hidden state
  return sampledIdx, hidden


# function to generate text
def sample(model, tokenizer, size, device, initial):

    # push to GPU
    model.to(device)

    model.eval()

    # batch size is 1
    hidden = model.init_hidden(1)

    toks = initial
    title = []

    # predict next token
    for t in initial:
      token, hidden = predict(model, t, hidden)

    toks.append(token)
    title.append(token)

    # predict subsequent tokens
    for i in range(size-1):
        token, hidden = predict(model, toks[-1], hidden)
        toks.append(token)
        title.append(token)

    return tokenizer.sequences_to_texts([title])[0]

def testInputPrep(ingList, tokenizer):
  ingList = tokenizer.texts_to_sequences([' '.join(ingList)])[0]

  return(ingList)
