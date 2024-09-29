import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch import nn

class NCF(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=100, n_hidden=30,y_range=(0,5.5),layers=None):
        super(NCF, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.movie_factors.weight.data.uniform_(0, 0.05)
        self.y_range  = y_range
        if layers == None:
            self.layers = nn.Sequential(nn.ReLU(),nn.Dropout(0.1),nn.Linear(n_factors*2, n_hidden),nn.ReLU(),nn.Linear(n_hidden, 1))
        else:
            self.layers  = layers
 
        
    def forward(self, x):
        U = self.user_factors(x[:,0])
        V = self.movie_factors(x[:,1])
        x = torch.cat([U, V], dim=1)
        for l in self.layers: x = l(x)
        high,low =  self.y_range
        out =  (torch.sigmoid(x) * (high - low) + low).squeeze(1)
        return out
        