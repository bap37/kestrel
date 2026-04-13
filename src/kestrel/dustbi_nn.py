import torch
import torch.nn.functional as F
import torch.nn as nn
from sbi import analysis as analysis
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

class PopulationEmbeddingFull(nn.Module):
    """Attention-pooled embedding network for a variable-size SN Ia population.

    Maps a set of N supernovae (each described by input_dim observables) to a
    fixed-length summary vector via per-supernova encoding followed by
    attention-weighted pooling.

    Architecture:
        phi:       per-SN encoder  (input_dim → hidden_dim)
        attention: scalar attention weight per SN  (hidden_dim → 1)
        rho:       output projection  (hidden_dim → output_dim)

    Args:
        input_dim:  Number of observables per supernova (matches
                    len(parameters_to_condition_on) in the config).
        hidden_dim: Width of the internal representation layers.
        output_dim: Dimensionality of the population summary vector passed
                    to the density estimator.
    """
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
        """Embed a batch of SN populations.

        Args:
            x: Tensor of shape (batch, N, input_dim) — a batch of SN populations,
               each containing N supernovae with input_dim observables.

        Returns:
            Tensor of shape (batch, output_dim) — one summary vector per population.
        """
        h = self.phi(x)                              # (batch, N, hidden)
        w = torch.softmax(self.attention(h), dim=1)   # (batch, N, 1)
        h = (h * w).sum(dim=1)                        # (batch, hidden)
        return self.rho(h)
