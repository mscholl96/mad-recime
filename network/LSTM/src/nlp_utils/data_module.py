import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from config import create_config

from transformers import AutoTokenizer, DistilBertTokenizer

from ..preProc import getPreProcData

# Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence


class TitleSet(Dataset):
    def __init__(self, datapath, setSize=8):

      data = getPreProcData(datapath, inpRange=range(setSize))

      self.tokenizer = Tokenizer(oov_token='OOV')

      # dataset split into word sequences required for training
      self.wordSeq = np.vectorize(self.getSequence, otypes=[np.ndarray])(
          data['ingredient'], data['title'])

      # training requires same length sequences -->  padding
      self.maxSequenceLength = max([len(seq['ings']) for seq in self.wordSeq])

      # list of all words in dataset
      self.vocab = np.concatenate(np.vectorize(self.getCorpus, otypes=[
                                  np.ndarray])(data['ingredient'], data['title']))

      # tokenization corpus
      self.tokenizer.fit_on_texts(self.vocab)

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
          0]  # optional value=1
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


class TitleDataModule(pl.LightningDataModule):
    """
    This is the DistilBERT based model. It's called "BaseModel" as I thought it would
    be the only NN model we would make...
    """

    def __init__(self, data_dir: str = "path/to/dir", num_workers=4, config={}):
        super().__init__()
        self.config = create_config(config)
        self.data_dir = data_dir
        self.batch_size = self.config['batchSize']
        self.num_workers = num_workers

        self.trainSet = None
        self.valSet = None
        self.testSet = None
        self.predictSet = None

        self.vocab_size = None

    def setup(self, stage = None):
        self.testSet = getPreProcData(self.data_dir, range(-1, 0))
        titleSet = TitleSet(self.data_dir, setSize=8)

        self.vocab_size = len(titleSet.tokenizer.word_index)

        # split data set
        trainNum = int(self.config['ratio'][0] * len(titleSet))
        valNum = len(titleSet) - trainNum

        self.trainSet, self.valSet = random_split(
            titleSet, [trainNum, valNum], generator=torch.Generator().manual_seed(0))

    def train_dataloader(self):
        return DataLoader(self.trainSet, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valSet, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testSet, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.predictSet, batch_size=self.batch_size)



class TitleTokenizer:
    """
    class to prepare tokenizer to be used
    """
    def __init__(self) -> None:
        self.tokenizer = Tokenizer(oov_token='OOV')
        self.tokenizer.fit_on_texts(TitleSet.words)

    def get_tokenizer(self):
        return self.tokenizer
