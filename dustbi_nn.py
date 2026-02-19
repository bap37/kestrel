import torch
import torch.nn.functional as F
import torch.nn as nn
from sbi import analysis as analysis
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

class PopulationEmbeddingFull(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.phi(x)                              # (batch, N, hidden)
        w = torch.softmax(self.attention(h), dim=1)   # (batch, N, 1)
        h = (h * w).sum(dim=1)                        # (batch, hidden)
        return self.rho(h)
