import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def standardize_data(matrix: sparse.csr_matrix):
    # replace nan with -99
    #matrix = np.array(matrix.toarray())
    #matrix[np.isnan(matrix)] = -99
    # make all values float
    #matrix = matrix.reshape(-1, matrix.shape[1]).astype('float32')
    # split randomly
    X_train, X_test = train_test_split(matrix, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, matrix: sparse.csr_matrix, standardizer):
        self.x = matrix
        self.standardizer = standardizer
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

def sparse_coo_to_tensor(coo:sparse.coo_matrix) -> torch.FloatTensor:
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s).to_dense()

def sparse_batch_collate(batch:list) -> torch.FloatTensor:
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    if type(batch[0]) == sparse.csr_matrix:
        batch = sparse.vstack(batch).tocoo()
        batch = sparse_coo_to_tensor(batch)
    else:
        batch = torch.FloatTensor(batch)

    batch[batch!=batch] = 0

    return batch

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE, loss_KLD, loss_MSE + loss_KLD