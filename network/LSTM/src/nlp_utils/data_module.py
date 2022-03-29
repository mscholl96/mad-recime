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
from torch.utils.data import DataLoader, Dataset, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from nlp_utils.config import create_config

from transformers import AutoTokenizer, DistilBertTokenizer





class TitleSet(pl.LightningDataModule):



def inverse_transform(vector, config=None, encoder=None):
    """
    returns the actual categories from a one-hot-encoding.
    Based on the config, some categories might be missing (e.g. group_id for rtnews)
    """
    config = create_config(config)
    result = {}
    decoded = list(encoder.inverse_transform(vector)[0])

    if config["category_group_id"]:
        result["group_id"] = decoded.pop(0)

    if config["category_type"]:
        result["post_type"] = decoded.pop(0)

    if config["category_tld"]:
        result["domain"] = decoded.pop(0)

    return result


class Collator:
    """
    helper class to transform a batch so it can be fed into the DistilBERT based model
    """

    def __init__(self, tokenizer, class_encoder, config):
        self.config = create_config(config)
        self.tokenizer = tokenizer
        self.class_encoder = class_encoder

    def collate(self, batch):
        labels, features = zip(*batch)

        encoded_texts = self.tokenizer(
            [row["Text"] for row in features],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded_classes = self.class_encoder.transform(
            list(get_classes_per_row(features, self.config))
        )

        return (
            torch.FloatTensor(labels),
            encoded_texts,
            torch.FloatTensor(encoded_classes),
            features,
        )


class CrowdTangleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index], self.texts[index]

    def __len__(self):
        return len(self.texts)


def create_datasets(class_encoder, config={}):
    """
    creates a train, validation and test dataset.
    The trainset will be shuffeled. 42 is used as random seed to get deterministic results.
    """
    config = create_config(config)

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, config["dataset_path"])
    df = pd.read_csv(filename, low_memory=False)

    df["Text"] = (
        df["Message"].fillna("").astype(str)
        + " "
        + df["Image Text"].fillna("").astype(str)
        + " "
        + df["Link Text"].fillna("").astype(str)
        + " "
        + df["Description"].fillna("").astype(str)
    )

    raw_records = df.to_dict("records")
    raw_labels = df[config["column_goldlabel"]].to_list()

    # Split train / val_test date
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        raw_records, raw_labels, test_size=0.4, random_state=42
    )
    # maybe zse torch.utils.data.random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    # initialize encoder using training data
    class_encoder.fit(list(get_classes_per_row(X_train, config)))

    # Split validation / test set
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42
    )

    return (
        CrowdTangleDataset(X_train, y_train),
        CrowdTangleDataset(X_val, y_val),
        CrowdTangleDataset(X_test, y_test),
    )


class CrowdTangleDataModule(pl.LightningDataModule):
    """
    Data module for the DistilBERT based model + linear model
    """

    def __init__(self, num_workers=4, config={}):
        super().__init__()
        config = create_config(config)
        self.batch_size = config["batch_size"]
        self.num_workers = num_workers

        self.collator = None
        self.trainset = None
        self.valset = None
        self.testset = None
        self.vocab = []
        self.config = config
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.class_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def setup(self, stage = None):
        if self.trainset is None:
            self.collator = Collator(self.tokenizer, self.class_encoder, self.config)
            self.trainset, self.valset, self.testset = create_datasets(
                self.class_encoder, self.config
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def get_tokenizer(self):
        return self.tokenizer


class PlainCollator:
    """
    helper class to transform a batch so it can be fed into the BiLSTM model
    """

    def __init__(self, vocab, tokenizer, class_encoder, config):
        self.config = create_config(config)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.class_encoder = class_encoder

    def collate(self, batch):
        labels, features = zip(*batch)

        encoded_texts = []
        for row in features:
            encoded_texts.append(
                torch.tensor(
                    [self.vocab[token] for token in self.tokenizer(row["Text"])]
                )
            )

        encoded_classes = self.class_encoder.transform(
            list(get_classes_per_row(features, self.config))
        )

        return (
            torch.FloatTensor(labels),
            pad_sequence(
                encoded_texts, batch_first=True, padding_value=self.vocab["<pad>"]
            ),
            torch.FloatTensor(encoded_classes),
            features,
        )


class PlainCrowdTangleDataModule(pl.LightningDataModule):
    """
    Data module for the BiLSTM model
    """

    def __init__(self, num_workers=4, config={}):
        super().__init__()
        config = create_config(config)
        self.batch_size = config["batch_size"]
        self.num_workers = num_workers

        self.collator = None
        self.trainset = None
        self.valset = None
        self.testset = None
        self.vocab = None
        self.config = config
        self.tokenizer = get_tokenizer("basic_english")
        self.class_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def setup(self, stage):
        if self.trainset is None:
            self.trainset, self.valset, self.testset = create_datasets(
                self.class_encoder, self.config
            )
            self._build_vocab()
            self.collator = PlainCollator(
                self.vocab, self.tokenizer, self.class_encoder, self.config
            )

    def _build_vocab(self):
        counter = Counter()
        for (label, features) in iter(self.trainset):
            counter.update(self.tokenizer(features["Text"]))
        self.vocab = Vocab(counter, min_freq=self.config["vocab_min_freq"])

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def get_tokenizer(self):
        return self.tokenizer


def create_SemEval_datasets(config={}):
    """
    creates a train, validation and test dataset.
    The trainset will be shuffeled. 42 is used as random seed to get deterministic results.
    """

    def clean_ascii(text):
        # function to remove non-ASCII chars from data
        return "".join(i for i in text if ord(i) < 128)

    # default data directory
    dirname = os.path.dirname(__file__)
    data_dir = "../../data/raw/SemEval/"

    if config["dataset_path"]:
        data_dir = os.path.join(dirname, config["dataset_path"])
    else:
        data_dir = os.path.join(dirname, data_dir)

    path = Path(data_dir)
    trainfile = "semeval2016-task6-trainingdata.txt"
    testfile = "SemEval2016-Task6-subtaskA-testdata.txt"

    # create train, test data
    df_train = pd.read_csv(
        path / trainfile, delimiter="\t", header=0, encoding="latin-1"
    )
    df_test = pd.read_csv(path / testfile, delimiter="\t", header=0, encoding="latin-1")
    df_train["Tweet"] = df_train["Tweet"].apply(clean_ascii)
    df_test["Tweet"] = df_test["Tweet"].apply(clean_ascii)

    # encode classes
    stances = ["AGAINST", "FAVOR", "NONE", "UNKNOWN"]
    stance_dict = {s: i for i, s in enumerate(stances)}
    inv_stance_dict = {i: s for i, s in enumerate(stances)}
    targets = df_train.Target.unique()
    target_dict = {s: i for i, s in enumerate(targets)}
    inv_target_dict = {i: s for i, s in enumerate(targets)}

    # select data
    X = df_train.Tweet.values
    y_stance = df_train.Stance.values
    y_target = df_train.Target.values

    # convert into int
    y_stance = np.array([stance_dict[s] for s in df_train.Stance])
    y_target = np.array([target_dict[s] for s in df_train.Target])

    X_test = df_test.Tweet.values
    y_stance_test = np.array([stance_dict[s] for s in df_test.Stance])
    y_target_test = np.array([target_dict[s] for s in df_test.Target])
    y_test = (y_stance_test, y_target_test)

    # create train and val split
    (
        X_train,
        X_val,
        y_train_stance,
        y_val_stance,
        y_train_target,
        y_val_target,
    ) = train_test_split(X, y_stance, y_target, test_size=0.2, random_state=42)

    # put in tuple
    y_train = (y_train_stance, y_train_target)
    y_val = (y_val_stance, y_val_target)

    return (
        inv_stance_dict,
        inv_target_dict,
        SemEvalDataset(X_train, y_train),
        SemEvalDataset(X_val, y_val),
        SemEvalDataset(X_test, y_test),
    )


class SemEvalCollator:
    """
    helper class to transform a batch so it can be fed into the CustomDistilBERT based model
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch):
        labels, features = zip(*batch)

        encoded_texts = self.tokenizer(
            [row for row in features],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return (
            torch.LongTensor(labels),
            encoded_texts,
        )


class SemEvalDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return (self.labels[0][index], self.labels[1][index]), self.texts[index]

    def __len__(self):
        return len(self.texts)


class SemEvalDataModule(pl.LightningDataModule):
    """
    Data module for the CustomDistilBERT based model + linear model
    """

    def __init__(self, num_workers=4, config={}):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.num_workers = num_workers

        self.collator = None
        self.trainset = None
        self.valset = None
        self.testset = None
        self.vocab = []
        self.config = config
        self.categories = []
        self.stance_encoding = None
        self.target_encoding = None
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def setup(self, stage: str = None):
        if self.trainset is None:
            self.collator = SemEvalCollator(self.tokenizer)
            (
                self.stance_encoding,
                self.target_encoding,
                self.trainset,
                self.valset,
                self.testset,
            ) = create_SemEval_datasets(self.config)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate,
            num_workers=self.num_workers,
        )

    def get_tokenizer(self):
        return self.tokenizer
