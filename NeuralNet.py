import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
  def __init__(self, embedding_dim):
    super(NN, self).__init__()
    self.linear1 = nn.Linear(embedding_dim, 128, bias=True)
    self.linear2 = nn.Linear(128, 256, bias=True)
    self.linear3 = nn.Linear(256, 64, bias=True)
    self.linear4 = nn.Linear(64, 1)

  def forward(self, input):
    out = torch.tanh(self.linear1(input))
    out = torch.tanh(self.linear2(out))
    out = torch.tanh(self.linear3(out))
    out = torch.sigmoid(self.linear4(out))
    return out
