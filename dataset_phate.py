import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import torch
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Fnirs134EEGDataset(Dataset):   
    def __init__(self, X, Y, P, eeg_chunk_length = 1, fnirs_chunk_length = 1, n_samples = 1):
        self.X = torch.tensor(X, dtype=torch.double)
        self.Y = torch.tensor(Y, dtype=torch.double)
        self.P = torch.tensor(P, dtype=torch.double)
        self.n_samples = n_samples
        self.eeg_chunk_length = eeg_chunk_length
        self.fnirs_chunk_length = fnirs_chunk_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        index1 = index * math.ceil(self.eeg_chunk_length)
        index2 = index * math.ceil(self.fnirs_chunk_length)
        
        return (self.X[index1 : index1 + self.eeg_chunk_length, 0:31], self.Y[index2 : index2 + self.fnirs_chunk_length, 0:134], self.P[index1 : index1 + self.eeg_chunk_length, :])

