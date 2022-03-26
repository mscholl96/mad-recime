from numpy import average
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import DistilBertModel, BertModel
from nlp_utils.config import create_config
import torchmetrics
import os, subprocess
import numpy as np
import pandas as pd
import re

class BasePartyModel(pl.LightningModule):
    """
    This is the DistilBERT based model. It only contains the party classifier and it is used for explanations.
    """

    def __init__(self, config={}):

        super().__init__()
        config = create_config(config)
        self.config = config
        self.save_hyperparameters(self.config)
        self.learning_rate = config["learning_rate"]

        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # setup layers
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if self.config["vocab_size"] == 0:
            self.config["vocab_size"] = self.bert.config.vocab_size

        # freeze the encode, head layer will still be trainable
        for param in self.bert.parameters():
            param.requires_grad = False

        # Model design
        self.distilbert_tail_party = nn.Sequential(
            nn.Linear(self.bert.config.dim, self.bert.config.dim),
            nn.ReLU(),
            nn.Dropout(self.bert.config.seq_classif_dropout),
        )

        # 768 bert hidden state shape + category_encoder_out
        self.party = nn.Linear(
            self.bert.config.hidden_size, 1
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids, attention_mask
        )

        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.distilbert_tail_party(pooled_output)


        out = self.party(pooled_output)
        sig = nn.Sigmoid()
        return sig(out)

    def training_step(self, batch, batch_idx):
        y, encoded_texts, category_vectors, _ = batch
        y, encoded_texts, category_vectors = (
            y.to(self.device),
            encoded_texts.to(self.device),
            category_vectors.to(self.device),
        )

        y_hat = self(encoded_texts["input_ids"], encoded_texts["attention_mask"])

        loss = F.binary_cross_entropy(y_hat.view(-1), y.view(-1))
        self.train_metric(y_hat, y.unsqueeze(1))

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def training_epoch_end(self, outs):
        self.log(
            "train_epoch" + type(self.train_metric).__name__,
            self.train_metric.compute(),
        )

    def validation_step(self, batch, batch_idx):
        y, encoded_texts, category_vectors, _ = batch
        y, encoded_texts, category_vectors = (
            y.to(self.device),
            encoded_texts.to(self.device),
            category_vectors.to(self.device),
        )

        y_hat = self(encoded_texts["input_ids"], encoded_texts["attention_mask"])

        loss = F.binary_cross_entropy(y_hat.view(-1), y.view(-1))
        self.val_metric(y_hat, y.unsqueeze(1))

        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        self.log(
            "val_epoch_" + type(self.train_metric).__name__, self.val_metric.compute()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class BaseModel(pl.LightningModule):
    """
    This is the DistilBERT based model. It's called "BaseModel" as I thought it would
    be the only NN model we would make...
    """

    def __init__(self, config={}):

        super().__init__()
        config = create_config(config)
        self.config = config
        self.save_hyperparameters(self.config)
        self.learning_rate = config["learning_rate"]

        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # setup layers
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if self.config["vocab_size"] == 0:
            self.config["vocab_size"] = self.bert.config.vocab_size

        # freeze the encode, head layer will still be trainable
        for param in self.bert.parameters():
            param.requires_grad = False

        # Model design
        self.distilbert_tail = nn.Sequential(
            nn.Linear(self.bert.config.dim, self.bert.config.dim),
            nn.ReLU(),
            nn.Dropout(self.bert.config.seq_classif_dropout),
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(
                config["category_encoded_length"], config["category_encoder_out"]
            ),
            nn.ReLU(),
        )
        # 768 bert hidden state shape + category_encoder_out
        self.classifier = nn.Linear(
            self.bert.config.hidden_size + config["category_encoder_out"], 1
        )

    def forward(self, encoded_text, category_vectors):
        bert_output = self.bert(
            encoded_text["input_ids"], encoded_text["attention_mask"]
        )

        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.distilbert_tail(pooled_output)

        categories_encoded = self.category_encoder(category_vectors)
        # test = torch.cat((pooled_output, categories_encoded))
        concat = torch.cat((pooled_output, categories_encoded), 1)

        out = self.classifier(concat)
        # out = self.output_layer(bert_output['pooler_output'])
        return out

    def training_step(self, batch, batch_idx):
        y, encoded_texts, category_vectors, _ = batch
        y, encoded_texts, category_vectors = (
            y.to(self.device),
            encoded_texts.to(self.device),
            category_vectors.to(self.device),
        )

        y_hat = self(encoded_texts, category_vectors)

        loss = F.mse_loss(y_hat.view(-1), y.view(-1))
        self.train_metric(y_hat, y.unsqueeze(1))

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def training_epoch_end(self, outs):
        self.log(
            "train_epoch" + type(self.train_metric).__name__,
            self.train_metric.compute(),
        )

    def validation_step(self, batch, batch_idx):
        y, encoded_texts, category_vectors, _ = batch
        y, encoded_texts, category_vectors = (
            y.to(self.device),
            encoded_texts.to(self.device),
            category_vectors.to(self.device),
        )

        y_hat = self(encoded_texts, category_vectors)

        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.val_metric(y_hat, y.unsqueeze(1))

        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        self.log(
            "val_epoch_" + type(self.train_metric).__name__, self.val_metric.compute()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class BiLSTMModel(pl.LightningModule):
    """
    This is the second neural model. (BiLSTM)
    """

    def __init__(self, config={}):
        super().__init__()

        config = create_config(config)
        self.config = config
        self.save_hyperparameters(self.config)
        self.learning_rate = config["learning_rate"]

        self.train_metric = pl.metrics.MeanSquaredError()
        self.val_metric = pl.metrics.MeanSquaredError()

        # Network structure
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])

        self.category_encoder = nn.Sequential(
            nn.Linear(
                config["category_encoded_length"], config["category_encoder_out"]
            ),
            nn.ReLU(),
        )

        self.bilstm = nn.LSTM(
            input_size=config["embedding_dim"],
            hidden_size=config["bilstm_hidden_dim"],
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            config["bilstm_hidden_dim"] * 2 + config["category_encoder_out"], 1
        )  #

    def forward(self, encoded_texts, encoded_classes):
        embeddings = self.embedding(encoded_texts)
        lstm_out, _ = self.bilstm(embeddings)
        categories_encoded = self.category_encoder(encoded_classes)
        # print("categories_encoded", categories_encoded.shape)
        # concat = torch.cat((lstm_out, categories_encoded),1)
        forward = lstm_out[:, lstm_out.shape[1] - 1, :]
        # backward = lstm_out[:, 0, :]
        combined = torch.cat((forward, categories_encoded), 1)
        out = self.classifier(combined)
        return out

    def training_step(self, batch, batch_idx):
        y, encoded_texts, encoded_classes, _ = batch
        y, encoded_texts, encoded_classes = (
            y.to(self.device),
            encoded_texts.to(self.device),
            encoded_classes.to(self.device),
        )

        y_hat = self(encoded_texts, encoded_classes)

        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.train_metric(y_hat, y.unsqueeze(1))

        self.log("loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outs):
        self.log(
            "train_epoch" + type(self.train_metric).__name__,
            self.train_metric.compute(),
        )

    def validation_step(self, batch, batch_idx):
        y, encoded_texts, encoded_classes, _ = batch
        y, encoded_texts, encoded_classes = (
            y.to(self.device),
            encoded_texts.to(self.device),
            encoded_classes.to(self.device),
        )

        y_hat = self(encoded_texts, encoded_classes)

        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.val_metric(y_hat, y.unsqueeze(1))

        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        self.log(
            "val_epoch_" + type(self.train_metric).__name__, self.val_metric.compute()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CustomDistilBertModel(pl.LightningModule):
    """
    This is the DistilBERT based model for stance prediction
    """

    def __init__(self, config={}):

        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.learning_rate = config["learning_rate"]
        self.stance_encoding = config["stance_encoding"]
        self.target_encoding = config["target_encoding"]
        self.num_classes_stance = 3
        self.num_classes_target = 5

        # metric for stance
        self.train_metric_stance = torchmetrics.F1(
            num_classes=self.num_classes_stance, average="micro"
        )
        self.val_metric_stance = torchmetrics.F1(
            num_classes=self.num_classes_stance, average="micro"
        )
        self.test_metric_stance = torchmetrics.F1(
            num_classes=self.num_classes_stance, average="micro"
        )

        # metric for target
        self.train_metric_target = torchmetrics.F1(
            num_classes=self.num_classes_target, average="micro"
        )
        self.val_metric_target = torchmetrics.F1(
            num_classes=self.num_classes_target, average="micro"
        )
        self.test_metric_target = torchmetrics.F1(
            num_classes=self.num_classes_target, average="micro"
        )

        # save predictions from test_set - needed for test script
        self.pred = np.empty(0, dtype="int64")

        # setup layers
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if self.config["vocab_size"] == 0:
            self.config["vocab_size"] = self.bert.config.vocab_size

        # freeze the encoding, head layer will still be trainable
        for param in self.bert.parameters():
            param.requires_grad = False

        # Model design - two separate bert tails for the tasks for best performance
        self.distilbert_tail_stance = nn.Sequential(
            nn.Linear(self.bert.config.dim, self.bert.config.dim),
            nn.ReLU(),
            nn.Dropout(self.bert.config.seq_classif_dropout),
        )

        self.distilbert_tail_target = nn.Sequential(
            nn.Linear(self.bert.config.dim, self.bert.config.dim),
            nn.ReLU(),
            nn.Dropout(self.bert.config.seq_classif_dropout),
        )

        # 768 bert hidden state shape
        self.classifier_stance = nn.Linear(
            self.bert.config.hidden_size,
            self.num_classes_stance,
        )
        self.classifier_target = nn.Linear(
            self.bert.config.hidden_size,
            self.num_classes_target,
        )

    def forward(self, encoded_text):
        with torch.no_grad():
            bert_output = self.bert(
                encoded_text["input_ids"], encoded_text["attention_mask"]
            )

        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output_stance = self.distilbert_tail_stance(pooled_output)
        pooled_output_target = self.distilbert_tail_stance(pooled_output)

        out_stance = self.classifier_stance(pooled_output_stance)
        out_target = self.classifier_target(pooled_output_target)
        return (out_stance, out_target)

    def training_step(self, batch, batch_idx):
        y, encoded_texts = batch
        y, encoded_texts = (y.to(self.device), encoded_texts.to(self.device))

        y_hat = self(encoded_texts)
        pred_stance = torch.argmax(y_hat[0], axis=1)
        pred_target = torch.argmax(y_hat[1], axis=1)

        loss_stance = F.cross_entropy(y_hat[0], y[:, 0])
        loss_target = F.cross_entropy(y_hat[1], y[:, 1])
        loss = loss_stance + loss_target
        self.train_metric_stance(pred_stance, y[:, 0])
        self.train_metric_target(pred_target, y[:, 1])

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def training_epoch_end(self, outs):
        self.log(
            "train_epoch_stance_" + type(self.train_metric_stance).__name__,
            self.train_metric_stance.compute(),
        )
        self.log(
            "train_epoch_target_" + type(self.train_metric_target).__name__,
            self.train_metric_target.compute(),
        )
        self.log(
            "train_epoch_" + type(self.val_metric_target).__name__,
            (self.train_metric_target.compute() + self.train_metric_stance.compute())
            / 2,
        )

    def validation_step(self, batch, batch_idx):
        y, encoded_texts = batch
        y, encoded_texts = (y.to(self.device), encoded_texts.to(self.device))

        y_hat = self(encoded_texts)
        pred_stance = torch.argmax(y_hat[0], axis=1)
        pred_target = torch.argmax(y_hat[1], axis=1)

        loss_stance = F.cross_entropy(y_hat[0], y[:, 0])
        loss_target = F.cross_entropy(y_hat[1], y[:, 1])
        loss = loss_stance + loss_target
        self.val_metric_stance(pred_stance, y[:, 0])
        self.val_metric_target(pred_target, y[:, 1])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        self.log(
            "val_epoch_stance_" + type(self.val_metric_stance).__name__,
            self.val_metric_stance.compute(),
        )
        self.log(
            "val_epoch_target_" + type(self.val_metric_target).__name__,
            self.val_metric_target.compute(),
        )
        self.log(
            "val_epoch_" + type(self.val_metric_target).__name__,
            (self.val_metric_target.compute() + self.val_metric_stance.compute()) / 2,
        )

    def test_step(self, batch, batch_idx):
        y, encoded_texts = batch
        y, encoded_texts = (y.to(self.device), encoded_texts.to(self.device))

        y_hat = self(encoded_texts)
        pred_stance = torch.argmax(y_hat[0], axis=1)
        pred_target = torch.argmax(y_hat[1], axis=1)

        # create predictions array
        pred2 = pred_stance.detach().cpu().numpy()
        self.pred = np.concatenate((self.pred, pred2), axis=None)

        # stance loss can't be calculated because test set has all stances set to 3, which is not a valid stance the model can predict
        loss_stance = 0 # F.cross_entropy(y_hat[0], y[:, 0])
        loss_target = F.cross_entropy(y_hat[1], y[:, 1])
        loss = loss_stance + loss_target
        self.test_metric_target(pred_target, y[:, 1])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        # save predictions to file and reset self.pred
        predictions = self.pred
        self.pred = pd.DataFrame()
        path = os.path.dirname(__file__)
        save_dir = "../../logs/StancePrediction_SemEval/predictions/"
        log_dir = "../../logs/StancePrediction_SemEval/lightning_logs/"
        log_path = os.path.join(path, save_dir)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # get number of latest version
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split("(\d+)", text)]

        ver = os.listdir(os.path.join(path, log_dir))
        ver.sort(key=natural_keys)
        if ver:
            version = int(ver[-1].split("_", 2)[-1])
        else:
            version = 0

        # get save directory and write predictions to file
        save_dir = save_dir + "version_" + str(version)
        filename = "bert_stance.tsv"
        pred_path = os.path.join(path, save_dir, filename)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w") as f:
            f.write("{}\t{}\n".format("index", "prediction"))
            for i, prediction in enumerate(predictions):
                f.write("{}\t{}\n".format(i, prediction))

        # read in test data and save to csv
        test_path = "../../data/raw/SemEval/SemEval2016-Task6-subtaskA-testdata.txt"
        test_path = os.path.join(path, test_path)
        test = pd.read_csv(test_path, delimiter="\t", header=0, encoding="latin-1")

        def clean_ascii(text):
            # function to remove non-ASCII chars from data
            return "".join(i for i in text if ord(i) < 128)

        test["Tweet"] = test["Tweet"].apply(clean_ascii)

        pred = pd.DataFrame(data=predictions, columns=["prediction"])
        pred.reset_index(level=0, inplace=True)

        test["Stance"] = np.array([self.stance_encoding[s] for s in pred.prediction])
        test["index"] = test.index
        df = test[["index", "Target", "Tweet", "Stance"]]
        file = "bert_predictions.txt"
        out_path = os.path.join(path, save_dir, file)

        df.to_csv(
            out_path, sep="\t", index=False, header=["ID", "Target", "Tweet", "Stance"]
        )

        base_dir = os.path.join(path, "../..")
        os.chdir(base_dir)

        # execute test pearl script
        subprocess.call(
            [
                "perl",
                "data/raw/SemEval/eval/eval.pl",
                "data/raw/SemEval/eval/gold.txt",
                "logs/StancePrediction_SemEval/predictions/version_"
                + str(version)
                + "/bert_predictions.txt",
            ]
        )
        os.chdir(path)

        self.log(
            "test_epoch_target_" + type(self.test_metric_target).__name__,
            self.test_metric_target.compute(),
        )
        self.log(
            "test_epoch_" + type(self.test_metric_target).__name__,
            self.test_metric_target.compute(),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
