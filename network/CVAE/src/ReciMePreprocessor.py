import word2vec
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import List
import pickle


class ReciMePreprocessor:
    def __init__(self, embPath: str, unitDictPath: str) -> None:
        self.w2v_model = word2vec.load(embPath)
        self.ingredientDict = {}
        with open(unitDictPath, "rb") as file:
            self.unitDict = pickle.load(file)

        for voc in self.w2v_model.vocab:
            # Offset by 1 so empty fields can be 0
            self.ingredientDict.setdefault(voc, len(self.ingredientDict) + 1)

        weights = torch.FloatTensor(self.w2v_model.vectors)
        self.emb = torch.nn.Embedding.from_pretrained(weights)

        self.normalizer_amounts = MinMaxScaler((-1, 1))
        self.normalizer_units = MinMaxScaler((-1, 1))
        self.normalizer_ingredients = MinMaxScaler((-1, 1))

    def oneHotEncoding(self, values: List[str], dictionary: dict) -> np.ndarray:
        embedding = np.zeros((20, len(dictionary)))

        for index in range(0, len(values)):
            embedding[index][dictionary[values[index]]] = 1

        return embedding.flatten()

    def normalizeData(self, data: List[np.ndarray]):
        data = np.array(data)
        # Those values were determined by iterating the whole dataset
        # without conversion and getting the min/max over the whole dataset
        self.normalizer_amounts.data_min_ = np.array([1e-12] * 20)
        self.normalizer_amounts.data_max_ = np.array([10000.0] * 20)
        self.normalizer_amounts.scale_ = (
            self.normalizer_amounts.feature_range[1]
            - self.normalizer_amounts.feature_range[0]
        ) / (self.normalizer_amounts.data_max_ - self.normalizer_amounts.data_min_)
        self.normalizer_amounts.min_ = (
            self.normalizer_amounts.feature_range[0]
            - self.normalizer_amounts.data_min_ * self.normalizer_amounts.scale_
        )
        amounts = self.normalizer_amounts.transform(data[:, :20])
        # Units do not need a feature range since they are one hot encoded (range 0:1)
        units = self.normalizer_units.fit_transform(
            data[:, 20 : len(self.unitDict) * 20 + 20]
        )
        # Those values were determined by iterating the whole dataset
        # without conversion and getting the min/max over the whole dataset
        self.normalizer_ingredients.data_min_ = np.array([-0.2968] * 6000)
        self.normalizer_ingredients.data_max_ = np.array([0.3292] * 6000)
        self.normalizer_ingredients.scale_ = (
            self.normalizer_ingredients.feature_range[1]
            - self.normalizer_ingredients.feature_range[0]
        ) / (
            self.normalizer_ingredients.data_max_
            - self.normalizer_ingredients.data_min_
        )
        self.normalizer_ingredients.min_ = (
            self.normalizer_ingredients.feature_range[0]
            - self.normalizer_ingredients.data_min_ * self.normalizer_ingredients.scale_
        )
        ingredients = self.normalizer_ingredients.transform(
            data[:, len(self.unitDict) * 20 + 20 :]
        )

        return np.concatenate((amounts, units, ingredients), axis=1)

    def preProcessInput(self, ds: pd.Series):
        assert "ingredients" == ds.name
        assert "amount" in ds.iloc[0].columns, "Column Amount not found in Dataframe"
        assert "unit" in ds.iloc[0].columns, "Column Unit not found in Dataframe"
        assert (
            "ingredient" in ds.iloc[0].columns
        ), "Column Ingredient not found in Dataframe"

        outputList = []

        for df in ds:
            # Split amounts, units, ingredients and convert to single row
            amounts = np.zeros((20,))
            amounts[: df["amount"].shape[0]] = df["amount"].tolist()
            # amounts = self.normalizer_amounts.transform(amounts)

            # Convert units to one hot encoded single row
            units = self.oneHotEncoding(df["unit"].tolist(), self.unitDict)
            # units = self.normalizer_units.transform(units)

            # Maybe embedd first and convert to single row afterwards
            ingredients = df["ingredient"].tolist()

            # Do embedding for integer encoded ingredients convert to single row
            # Replace spaces with underscores else no mapping can be found in the dict
            # Load vocab.txt (or vocab.bin) + create dict --> integer encode ingredients of the current recipe
            ingr = np.zeros((20,), dtype=int)
            for index in range(0, len(ingredients)):
                # preferably use last part of the ingredient, some are not available in the dict
                if ingredients[index]:
                    name_words = ingredients[index].lower().split(" ")
                    for i in range(len(name_words)):
                        name_ind = self.ingredientDict.get("_".join(name_words[i:]))
                        if name_ind:
                            ingr[index] = name_ind
                            break

            ingr_emb = self.emb(torch.LongTensor(ingr))

            ingrOut = ingr_emb.numpy().flatten()
            # ingrOut = self.normalizer_ingredients.transform(ingrOut)

            # Sort + concat amounts, units, ingredients (as df with corrent row names -> If embedded this is not possible + useless)
            # Sorting is ignored for now

            out = np.concatenate((amounts, units, ingrOut))
            # Append output df
            outputList.append(out)

        return outputList

    def inverseOneHotEncoding(
        self, encoded: np.ndarray, dictionary: dict
    ) -> np.ndarray:
        output = []
        for row in encoded:
            rowTransformed = np.reshape(row, (20, -1))
            rowString = []
            indexTransformed = np.argmax(rowTransformed, axis=1)
            for index in indexTransformed:
                if index in dictionary.values():
                    rowString.append(
                        list(dictionary.keys())[index]
                    )
                else:
                    rowString.append("")
            output.append(rowString)
        return np.array(output)

    def inverseEmbedding(self, embedded: np.ndarray, dictionary: dict) -> np.ndarray:
        output = []
        for row in embedded:
            outputRows = []
            for rowTransformed in np.reshape(row, (20, -1)):
                rowTransformed = torch.Tensor(rowTransformed)
                distance = torch.norm(self.emb.weight.data - rowTransformed, dim=1)
                nearest = torch.argmin(distance)
                index = nearest.item()
                if index:
                    outputRows.append(
                        list(dictionary.keys())[list(dictionary.values()).index(index)].replace("_", " ")
                    )
                else:
                    outputRows.append("")
            output.append(outputRows)

        return np.array(output)

    def decodeOutput(self, output: np.ndarray):
        # Split output into amounts, units and ingredients
        amounts = self.normalizer_amounts.inverse_transform(output[:, :20])
        units = self.normalizer_units.inverse_transform(
            output[:, 20 : len(self.unitDict) * 20 + 20]
        )
        ingredients = self.normalizer_ingredients.inverse_transform(
            output[:, len(self.unitDict) * 20 + 20 :]
        )
        unitsDecoded = self.inverseOneHotEncoding(units, self.unitDict)
        ingredientsDecoded = self.inverseEmbedding(ingredients, self.ingredientDict)
        outputList = []
        for index in range(len(amounts)):
            array = np.stack(
                (amounts[index], unitsDecoded[index], ingredientsDecoded[index]), axis=1
            )
            outputList.append(
                pd.DataFrame(array, columns=["amount", "unit", "ingredient"])
            )
        return outputList

    def getConditional(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return data[:, len(self.unitDict) * 20 + 20 : len(self.unitDict) * 20 + 20 + 300]
