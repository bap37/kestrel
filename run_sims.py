from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
from dustbi_plotting import plot_loss, plot_surviving_priors
import yaml, os, argparse
import shutil
import pickle

#cosmology imports
from astropy.cosmology import Planck18
#import astropy.units as u

# sbi and torch imports
#from sbi import utils as utils
#from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import torch
#import torch.nn as nn
#import torch.nn.functional as F
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
#from sbi import analysis as analysis
from sbi.utils import MultipleIndependent

def add_distance(df_tensor):
    
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    
    beta = 3.1 ; alpha = 0.16 ; M0 = -19.3
    
    correction = alpha * x1_obs - beta * c_obs + M0 + mB_obs
        
    MURES = df_tensor['MU'] - correction
    
    return  MURES


def get_args():
    
    
    parser = argparse.ArgumentParser()

    msg = """
    Enables importance sample simulator to generate simulations for the training of the network. \n
    Is boolean. \n
    Please configure the specific batch sizes and total number of simulations in KESTREL.yml"""
    parser.add_argument("--SIMULATE", help=msg, action="store_true")
    
    msg = """
    Set to toggle training process for the SBI network. \n
    Is boolean. \n
    Specifics on the network are in this file."""
    parser.add_argument("--TRAIN", help=msg, action="store_true")
    
    msg = """
    Configuration yaml for use with simulation and training.
    """
    parser.add_argument("--CONFIG", help=msg, type=str)

    msg = "Default False. Prints a nice bird :)"
    parser.add_argument("--BIRD", help=msg, action="store_true")


    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = get_args()

    if not args.CONFIG:
        print("No configuration file provided via --CONFIG. Quitting.")

    infos = load_kestrel(args.CONFIG)

    datfilename = infos['Data_File'][0]
    simfilename = infos['Simbank_File'][0]

    #Error Trap
    if not os.path.exists(datfilename):
        AssertionError(f"{datfilename} doesn't exist; quitting.")
    if not os.path.exists(simfilename):
        AssertionError(f"{simfilename} doesn't exist; quitting.")

    parameters_to_condition_on = infos['parameters_to_condition_on']
    n_sim = infos['sim_parameters']['n_sim']
    n_batch = infos['sim_parameters']['n_batch']
    sims_savename = infos['sim_parameters']['simname']
    posterior_savename = infos['sim_parameters']['posteriorname']

    
    dicts = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations']]

    ##############################
    # Load information and setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    param_names = infos['param_names']

    params_to_fit = parameter_generation(param_names, dicts)
    priors = prior_generator(param_names, dicts, device=device)

    layout = build_layout(params_to_fit, dicts)


    ndim = len(parameters_to_condition_on)
    print(f"The NN will be trained on a {ndim}-dimensional space, on {param_names}")

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
        device=device,
    )

    if args.BIRD:
        print("I'm very sorry but the kestrel hasn't taken flight yet!")
        quit()

    #A quick hack to avoid the painful loading of a bunch of unnecessary features...
    if args.SIMULATE:
        df, dfdata = load_data(simfilename, datfilename)

        print("Adding 'broad' MURES now. ")
    
        output_distribution = preprocess_input_distribution(
            df, parameters_to_condition_on[:-1]+['x0', 'x0ERR', 'MU'])

        MURES_SIMS = add_distance(output_distribution)
        df['MURES'] = MURES_SIMS

        output_distribution = preprocess_input_distribution(
            dfdata, parameters_to_condition_on[:-1]+['x0', 'x0ERR', 'MU'])

        MURES_DATA = add_distance(output_distribution)
        dfdata['MURES'] = MURES_DATA

        print("We are temporarily not standardising data.")
        #df, dfdata = standardise_data(df, dfdata, parameters_to_condition_on)


        sim_for_training = make_batched_simulator(layout, df,
                                param_names,parameters_to_condition_on,
                                dicts, dfdata, device=device)
        batched = True

    if args.SIMULATE:
        print(f"Training {n_sim} simulations and saving to {sims_savename}")
        theta, priors = simulate_model(n_sim, n_batch, sims_savename, priors, sim_for_training, inference, device=device, batched=batched)
        labels = unspool_labels(param_names, dicts, infos['Latex_Names'], infos['Functions'])
        plot_surviving_priors(theta,priors,labels,sims_savename.replace("h5","survivng_priors.pdf"))
        print("Quitting after simulation stage.")
        shutil.copy(args.CONFIG, sims_savename.replace(".h5", ".yml.bk"))
        quit()
    ################
    if args.TRAIN:
    ################
        if os.path.exists(sims_savename) & args.TRAIN:
            pass
        else:
            print(f"I've not detected {sims_savename} anywhere. Is this intentional?")
            quit()

        import h5py

        with h5py.File(sims_savename, "r") as f:
            theta_total = f["theta"]
            x_total = f["x"]
            n = theta_total.shape[0]
            chunk_size = 10_000

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                print(f"Processing chunk {start}:{end}")
                
                try:
                # Load chunk
                    theta_batch = torch.tensor(theta_total[start:end]).cuda()
                    x_batch = torch.tensor(x_total[start:end]).cuda()
                except AssertionError:
                    theta_batch = torch.tensor(theta_total[start:end])
                    x_batch = torch.tensor(x_total[start:end])

                # Append simulations
                inference.append_simulations(theta_batch, x_batch)

                # Train only on this chunk
                density_estimator = inference.train(
                    validation_fraction=0.1,
                    force_first_round_loss=True,  # prior samples; keeps training consistent
                    training_batch_size=64,
                )

                # Build posterior from final density estimator
                posterior = inference.build_posterior(density_estimator)

                with open(posterior_savename, "wb") as handle:
                    pickle.dump(posterior, handle)

                print(f"Chunk {start}:{end} trained and cleared from memory.")
                plot_loss(inference, posterior_savename.replace(".pt", "_loss.pdf"))

                # Clear simulations from inference to save memory
                inference._theta = []
                inference._x = []

