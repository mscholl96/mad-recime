import torch
from torch.utils.data import Dataset

# Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

import pandas as pd
import numpy as np
import glob


def preProc(dataPath, tarPath):

  dataSetSplits = glob.glob(dataPath  +  '/recipes_valid_*.pkl')

  for split in range(len(dataSetSplits)):
    baseFrame = pd.read_pickle(dataSetSplits[split])

    def getAmount(row):
      return row['amount'].tolist()
    def getUnit(row):
      return row['unit'].tolist()
    def getIng(row):
      return row['ingredient'].tolist()

    baseFrame['amount'] = np.vectorize(getAmount, otypes=[np.ndarray])(baseFrame['ingredients'])
    baseFrame['unit'] = np.vectorize(getUnit, otypes=[np.ndarray])(baseFrame['ingredients'])
    baseFrame['ingredient'] = np.vectorize(getIng, otypes=[np.ndarray])(baseFrame['ingredients'])
    baseFrame = baseFrame.drop(columns=['ingredients'])

    baseFrame.to_pickle(tarPath  + '/recipePreProc_' + str(split) + '.pkl')

def getPreProcData(dataPath, inpRange=None):
  baseFrame = pd.DataFrame()

  dataSetSplits = glob.glob(dataPath  +  '/recipePreProc_*.pkl')

  iterRange = range(len(dataSetSplits))
  
  if inpRange != None and len(inpRange) <= len(iterRange):
    iterRange = inpRange

  for split in iterRange:
    baseFrame = baseFrame.append(pd.read_pickle(dataSetSplits[split]))

  return baseFrame


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



class InstructionSet(Dataset):
    def __init__(self, datapath, setSize=8):

      data = getPreProcData(datapath, inpRange=range(setSize))

      self.delimiter = '\n'
      self.filters = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'.replace(
          self.delimiter, '').replace('-', '').replace('°', '')
      self.tokenizer = Tokenizer(oov_token='OOV', filters=self.filters)

      # dataset split into word sequences required for training
      self.wordSeq = np.vectorize(self.getSequence, otypes=[np.ndarray])(
          data['ingredient'], data['title'],  data['instructions'])

      # training requires same length sequences -->  padding
      self.maxSequenceLength = max(
          [len(seq['ingTitle']) for seq in self.wordSeq])

      # list of all words in dataset
      self.words = np.concatenate(np.vectorize(self.getCorpus, otypes=[np.ndarray])(
          data['ingredient'], data['title'], data['instructions']))

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

    def getCorpus(self, ingredient, title, instructions):
      ingTok = text_to_word_sequence(' '.join(ingredient))
      titleTok = text_to_word_sequence(title)
      instTok = text_to_word_sequence(
          str(' ' + self.delimiter + ' ').join(instructions), filters=self.filters)
      return np.array(ingTok + titleTok + instTok)

    def getSequence(self, ingredient, title, instructions):
      ingTok = text_to_word_sequence(' '.join(ingredient))
      titleTok = text_to_word_sequence(title)
      instTok = text_to_word_sequence(
          str(' ' + self.delimiter + ' ').join(instructions), filters=self.filters)
      return {'ingTitle': ingTok + titleTok, 'instructions': instTok}

    def getIndexedSeqs(self, seq):
      ingTok = self.tokenizer.texts_to_sequences([seq['ingTitle']])[0]
      ingTok = pad_sequences([ingTok], maxlen=self.maxSequenceLength, padding='pre')[
          0]
      instTok = self.tokenizer.texts_to_sequences([seq['instructions']])[0]

      return {'ingTitle': ingTok, 'instructions': instTok}

    def getMovWindSeq(self, seq):
      # input needs to be pre padded
      idxShift = len(seq['instructions'])
      ingLen = len(seq['ingTitle'])

      fullSeq = np.append(seq['ingTitle'], seq['instructions'])
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
