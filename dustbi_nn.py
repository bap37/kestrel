import torch
import torch.nn as nn
import torch.nn.functional as F

class PopulationEmbeddingFull(nn.Module):
    #input_dim is set to 2 by default, but will need to be scaled appropriately when mass-splitting.
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, N, 2)
        h = self.phi(x)           # (batch_size, N, hidden_dim)
        h = h.mean(dim=1)         # mean over N samples -> (batch_size, hidden_dim)
        return self.rho(h)        # (batch_size, output_dim)
