import torch
from torch import nn
import torch.nn.functional as F
import random


def downsample_multivariate_time_series(input_tensor, dec_seq_len, interpolation_method='linear'):
    """
    Downsample a multivariate time series using torch interpolation.
    - interpolation_method (str): The interpolation method to use ('linear', 'nearest', etc.).
    Returns:
    - torch.Tensor: The downsampled tensor.
    """
    downsampled_tensor = F.interpolate(input_tensor.permute(0,2,1), size = dec_seq_len, mode=interpolation_method).permute(0,2,1)

    return downsampled_tensor#.squeeze(2)

class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim, dropout):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
    self.dropout = dropout

    
    self.rnn = nn.LSTM(
      input_size=self.n_features,
      hidden_size=self.embedding_dim, 
      num_layers=3,
      dropout = self.dropout,
      batch_first=True
    )

  def forward(self, x):
    size = len(x)
    x = x.reshape((size, self.seq_len, self.n_features))   # (batch_size, seq_len, num of features)

    x, (hidden_n, cell_n) = self.rnn(x)
    x = x.reshape((size, self.seq_len, self.embedding_dim))

    return (x, hidden_n, cell_n)


class Decoder(nn.Module):

  def __init__(self, seq_len, embedding_dim, n_features, dropout):
    super(Decoder, self).__init__()

    self.seq_len, self.embedding_dim = seq_len, embedding_dim
    self.n_features =  n_features
    self.dropout = dropout

    self.rnn = nn.LSTM(
      input_size=self.n_features,
      hidden_size=self.embedding_dim,
      num_layers=3,
      dropout = self.dropout,
      batch_first=True
    )

    # outputting multivariate time series 
    self.output_layer1 = nn.Linear(self.embedding_dim, 100)  # dense output layer; 
    self.output_layer2 = nn.Linear(100, self.n_features)  # dense output layer; 
    self.relu = nn.ReLU()

  def forward(self, x, h_0, c_0):
    size = len(x)
    x, (hidden_n, cell_n) = self.rnn(x, (h_0, c_0))
    x = x.reshape((size, 1, self.embedding_dim))
    x = self.relu(self.output_layer1(x))
    x = self.output_layer2(x)
    x = x.reshape((size, 1, self.n_features))

    return (x, hidden_n, cell_n)

class RecurrentAutoencoder(nn.Module):

  def __init__(self, enc_seq_len, dec_seq_len, enc_features, dec_features, enc_embedding_dim, dec_embedding_dim, dropout, device):
    super(RecurrentAutoencoder, self).__init__()

    self.device = device

    self.encoder = Encoder(enc_seq_len, enc_features, enc_embedding_dim, dropout).to(self.device)
    self.decoder = Decoder(dec_seq_len, dec_embedding_dim, dec_features, dropout).to(self.device)

    self.dec_seq_len = dec_seq_len
    self.dec_features = dec_features

    self.linear_model = nn.Sequential(
    nn.Linear(128 * self.dec_seq_len, 4096),
    nn.Tanh(),
    nn.Linear(4096, 1024),
    nn.Tanh(),
    nn.Linear(1024, self.dec_seq_len * self.dec_features),
    nn.Tanh()
    # nn.Sigmoid()
    )

  
  
  def forward(self, x, y, teacher_forcing_ratio):
    batch_size = y.shape[0]
    trg_len = y.shape[1]
    trg_dim = y.shape[2]
    x_enc, hidden, cell = self.encoder(x)

    downsampled_x_enc = downsample_multivariate_time_series(x_enc, self.dec_seq_len, interpolation_method='linear')

    downsampled_x_enc = downsampled_x_enc.reshape(downsampled_x_enc.shape[0], -1)

    outputs = self.linear_model(downsampled_x_enc)

    outputs = outputs.reshape((batch_size, self.dec_seq_len, self.dec_features))

    return (x_enc, outputs)



