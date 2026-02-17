from dustbi_simulator import *
from Functions import *
######################

#Params to fit 
param_names = ['SIM_c', 'SIM_RV', 'SIM_beta', 'SIM_EBV']

###### load in and prep big sim and real data
import numpy as np
import pandas as pd
df = pd.read_csv("INPUT_DES5YR_D2D.FITRES", comment="#", sep='\s+')

df['SIM_EBV'] = df.SIM_AV/df.SIM_RV


dfdata = pd.read_csv("/Users/bpopovic/Documents/INVERSE_H0/D5YR_DATA/FITOPT000_MUOPT000.FITRES.gz", comment="#", sep=r'\s+')

##################
# Can't be arsed to do the yaml right now; here's the dictionaries

bounds_dict = {
    "SIM_c"   : (-0.5, 0.5),
    "SIM_RV"  : (0.5, 5),
    "SIM_EBV" : (0,1),
    "SIM_beta": (0.5,4),
}

function_dict = {
    "SIM_c"   : DistGaussian,
    "SIM_RV"  : DistGaussian,
    "SIM_EBV" : DistExponential,
    "SIM_beta": DistGaussian,
}

split_dict = {
    "SIM_RV":["HOST_LOGMASS", 10],
}

priors_dict = {    
    "SIM_c"   : [(-0.05, 0.05), (0.07, 0.07), False],
    "SIM_RV"  : [(3,1), (1,0.5), True],
    "SIM_EBV" : [.3, True],
    "SIM_beta": [(2,1), (0.5,1), True],

}

    
dicts = [bounds_dict, function_dict, split_dict, priors_dict]
##################
#Start up the band, etc 

params_to_fit = parameter_generation(param_names, dicts)
priors = prior_generator(param_names, dicts)
layout = build_layout(params_to_fit, dicts)

simulatinator = make_simulator(layout, df, param_names, dicts, dfdata)

def batched_simulator(theta_batch):
    return torch.stack([simulatinator(theta) for theta in theta_batch])

######


################# Start the inference down here

import torch.nn as nn
import torch.nn.functional as F

class PopulationEmbedding(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super().__init__()
        # phi maps each sample to hidden space
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # rho maps aggregated representation to final embedding
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, N, 2)
        """
        h = self.phi(x)        # (batch_size, N, hidden_dim)
        h = h.mean(dim=1)      # mean over N samples -> (batch_size, hidden_dim)
        return self.rho(h)     # (batch_size, output_dim)

def batched_simulator(theta_batch):
    return torch.stack([simulatinator(theta) for theta in theta_batch])

from sbi.inference import SNPE
from sbi.utils import MultipleIndependent

from sbi.neural_nets import posterior_nn

density_estimator = posterior_nn(
    model="maf",
    embedding_net=PopulationEmbedding()
)

inference = SNPE(
    prior=priors,
    density_estimator=density_estimator
)


data = torch.load("simulations_v1.pt")
theta_batch = data["theta"]
x_batch = data["x"]

print("starting the inference")

inference.append_simulations(theta_batch, x_batch)

density_estimator = inference.train()

print("\n inferred successfully")

posterior = inference.build_posterior(density_estimator)

torch.save(posterior, "posterior.pt")
