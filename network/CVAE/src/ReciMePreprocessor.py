import string
import word2vec
import numpy as np
import pandas as pd
import torch
from typing import List

class ReciMePreprocessor():
    def __init__(self, path: str) -> None:
        self.w2v_model = word2vec.load(path)
        self.ingredientDict = {}
        self.unitDict = {}

        for voc in self.w2v_model.vocab:
            # Offset by 1 so empty fields can be 0
            self.ingredientDict.setdefault(voc, len(self.ingredientDict)+1)

        weights = torch.FloatTensor(self.w2v_model.vectors)
        self.emb = torch.nn.Embedding.from_pretrained(weights)

    def oneHotEncoding(self, values: List[str], dictionary: dict) -> np.ndarray:
        embedding = np.zeros((20, len(dictionary)))

        for index in range(0, len(values)):
            embedding[index][dictionary[values[index]]-1] = 1

        return embedding.flatten()

    def preProcessInput(self, ds: pd.Series):
        assert('ingredients' == ds.name)
        assert 'amount' in ds.iloc[0].columns, "Column Amount not found in Dataframe"
        assert 'unit' in ds.iloc[0].columns, "Column Unit not found in Dataframe"
        assert 'ingredient' in ds.iloc[0].columns, "Column Ingredient not found in Dataframe"

        # Unit dict will still be necessary for the one hot encoding
        for df in ds:
            for unit in df['unit']:
                # Offset by 1 so empty fields can be 0
                self.unitDict.setdefault(unit, len(self.unitDict)+1)

        outputList = []

        for df in ds:
            # Split amounts, units, ingredients and convert to single row
            amounts = np.zeros((20,))
            amounts[:df['amount'].shape[0]] = df['amount'].tolist()

            # Convert units to one hot encoded single row
            units = self.oneHotEncoding(df['unit'].tolist(), self.unitDict)

            # Maybe embedd first and convert to single row afterwards
            ingredients = df['ingredient'].tolist()

            # Do embedding for integer encoded ingredients convert to single row
            # Replace spaces with underscores else no mapping can be found in the dict
            # Load vocab.txt (or vocab.bin) + create dict --> integer encode ingredients of the current recipe
            ingr = np.zeros((20,), dtype=int)
            for index in range(0,len(ingredients)):
                # preferably use last part of the ingredient, some are not available in the dict
                if ingredients[index]:
                    name_words = ingredients[index].lower().split(' ')
                    for i in range(len(name_words)):
                        name_ind = self.ingredientDict.get('_'.join(name_words[i:]))
                        if name_ind:
                            ingr[index] = name_ind
                            break

            ingr_emb = self.emb(torch.LongTensor(ingr))

            ingrOut = ingr_emb.numpy().flatten()

            # Sort + concat amounts, units, ingredients (as df with corrent row names -> If embedded this is not possible + useless)
            # Sorting is ignored for now

            out = np.concatenate((amounts, units, ingrOut))
            # Append output df
            outputList.append(out)

        return outputList


    def decodeOutput(self, output: np.ndarray):
        pass