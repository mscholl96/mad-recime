import pandas as pd
import numpy as np
import os
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