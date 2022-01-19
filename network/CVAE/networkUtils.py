import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def standardize_data(df: pd.DataFrame):
    # replace nan with -99
    df = df.fillna(-99)
    # make all values float
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    # split randomly
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, df, train=True):
        self.X_train, self.X_test, self.standardizer = standardize_data(df)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD