from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
import yaml, os, argparse

#cosmology imports
from astropy.cosmology import Planck18
import astropy.units as u

# sbi and torch imports
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi import analysis as analysis
from sbi.utils import MultipleIndependent


simfilename = 'INPUT_DES5YR_D2D.FITRES'
datfilename = 'SIMS_FOR_TESTING/FITOPT000.FITRES.gz'

infos = load_kestrel("KESTREL.yml")

parameters_to_condition_on = infos['parameters_to_condition_on']
n_sim = infos['sim_parameters']['n_sim']
n_batch = infos['sim_parameters']['n_batch']
sims_savename = infos['sim_parameters']['simname']
posterior_savename = infos['sim_parameters']['posteriorname']

function_dict = {
    "SIM_c"   : DistGaussian,
    "SIM_RV"  : DistGaussian,
    "SIM_EBV" : DistExponential,
    "SIM_beta": DistGaussian,
}

infos['Splits'] = {}
    
dicts = [infos['Boundaries'], function_dict, infos['Splits'], infos['Priors']]

##############################
# Load information and setup

param_names = infos['param_names']

params_to_fit = parameter_generation(param_names, dicts)
priors = prior_generator(param_names, dicts)

layout = build_layout(params_to_fit, dicts)

ndim = len(parameters_to_condition_on)
if any(p in infos['Splits'] for p in param_names): #check early to see if we need to split anything. 
    ndim *= 2
print(f"The NN will be trained on a {ndim}-dimensional space.")

###############
#Do some quick checks and establish density estimator and inference pipelines.
prior, num_parameters, prior_returns_numpy = process_prior(priors)

density_estimator = posterior_nn(
    model="nsf", #switch to nsf if interested 
    embedding_net=PopulationEmbeddingFull(input_dim=ndim)
)

inference = SNPE(
    prior=priors,
    density_estimator=density_estimator, 
)

def get_args():
    
    
    parser = argparse.ArgumentParser()

    msg = """
    Enables importance sample simulator to generate simulations for the training of the network. \n
    Is boolean. \n
    Please configure the specific batch sizes and total number of simulations in KESTREL.yml"""
    parser.add_argument("--SIMULATE", help=msg, type=bool, default=False)
    
    msg = """
    Set to toggle training process for the SBI network. \n
    Is boolean. \n
    Specifics on the network are in this file."""
    parser.add_argument("--TRAIN", help=msg, type=bool, default=False)
    
    msg = "Default False. Prints a nice bird :)"
    parser.add_argument("--BIRD", help = msg, type=bool, default=False)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.BIRD:
        print("I'm very sorry but the kestrel hasn't taken flight yet!")
        quit()

    df, dfdata = load_data(simfilename, datfilename)

    simulatinator = make_simulator(layout, df, param_names, 
                                   parameters_to_condition_on, dicts, dfdata, is_split=True)

    simulation_wrapper = process_simulator(simulatinator, prior, prior_returns_numpy)
    check_sbi_inputs(simulation_wrapper, prior)

    if args.SIMULATE:
        print(f"Training {n_sim} simulations and saving to {sims_savename}")
        train_model(n_sim, n_batch, sims_savename, priors, simulatinator, inference)
        print("Quitting after simulation stage.")
        quit()
    ################
    if os.path.exists(sims_savename):
        data = torch.load(sims_savename)
        theta_batch = data["theta"]
        x_batch = data["x"]

    else:
        print(f"I've not detected {sims_savename} anywhere. Is this intentional?")
        quit()
    ################
    if args.TRAIN:
        print(f"Not enabled yet")
        inference.append_simulations(theta_batch, x_batch)
        density_estimator = inference.train(validation_fraction=0.1)
        print("\n inferred successfully")
        posterior = inference.build_posterior(density_estimator)
        torch.save(posterior, posterior_savename)
