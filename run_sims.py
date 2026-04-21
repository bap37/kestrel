from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
from dustbi_calibration import *
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

    msg = """
    Default posterior calibration performance suite.  \n
    Is boolean. \n
    Does SBC, rank calibration, and TARP"""
    parser.add_argument("--CAL1", help=msg, action="store_true")
    
    msg = """
    More in-depth calibration performance \n
    Is boolean. \n
    Requires much more hard-coded values that can be found after the if args.CAL2 block. \n
    Will check cosmology dependence of Rv inference and SNANA-simulation based calibration.    """
    parser.add_argument("--CAL2", help=msg, action="store_true")
    

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
    layout = build_layout(params_to_fit, dicts)

    mixture = 'Population_B' in infos
    if mixture:
        dicts_B = [infos['Functions'], infos['Splits'],
                   infos['Population_B']['Priors'], infos['Correlations']]
        priors_A = build_distribution_priors(param_names, dicts, device=device)
        priors_B = build_distribution_priors(param_names, dicts_B, device=device)
        mix = infos['Population_B']['mixing_prior']
        f_prior = BoxUniform(
            low=torch.tensor([mix[0]], dtype=torch.float32, device=device),
            high=torch.tensor([mix[1]], dtype=torch.float32, device=device))
        special = build_special_priors(param_names, dicts, device=device)
        priors = MultipleIndependent(priors_A + priors_B + [f_prior] + special, device=device)
        print(f"Mixture mode: {len(priors_A)} pop A + {len(priors_B)} pop B + 1 mixing + {len(special)} special = {len(priors_A)+len(priors_B)+1+len(special)} total priors")
    else:
        priors = prior_generator(param_names, dicts, device=device)


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
                                dicts, dfdata, device=device, mixture=mixture)
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
                print(f"Appending chunk {start}:{end}")

                theta_batch = torch.tensor(theta_total[start:end])
                x_batch = torch.tensor(x_total[start:end])

                inference.append_simulations(theta_batch, x_batch, data_device="cpu")

        # Train once on all accumulated data
        density_estimator = inference.train(
            validation_fraction=0.1,
            force_first_round_loss=True,
            training_batch_size=64,
        )

        posterior = inference.build_posterior(density_estimator)

        with open(posterior_savename, "wb") as handle:
            pickle.dump(posterior, handle)

        print(f"Posterior saved to {posterior_savename}")
        plot_loss(inference, posterior_savename.replace(".pt", "_loss.pdf"))


    ################
    if args.CAL1:
    ################

        #Neeeeeed to clean this up something fierce

        posterior = load_posterior(posterior_savename, device)

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

        #truly hateful
        labels = unspool_labels(param_names, dicts, infos['Latex_Names'], infos['Functions'])
        truth = priors.sample()
        ws = [-0.9, -0.95, -1, -1.05, -1.1]
        #cosmology_dependence(df, ws, posterior, truth, device,
        #    parameters_to_condition_on, make_batched_simulator,
        #        layout, param_names, dicts, dfdata, priors, labels)





    ################
    if args.CAL2:
    ################

        #This section is not really for users. 
        #it is for some hardcore tseting for publication purposes.

        #Requires data to be stored in format of 
        # CALIB_DATA/output/PIP_BP-DUST-SAMPLES_DATADESSIM_IA-{SIM}/FITOPT000.FITRES.gz"
        #And 
        # CALIB_GENPDF/{GENPDF}.DAT"
        #And 
        # "ASSIGNMENTS" as a text file that maps the GENPDF to the simulation. 


        posterior = load_posterior(posterior_savename, device)

        GENPDFS, SIMS = load_assignments('CALIB_DATA/output/ASSIGNMENTS')
        ranks = run_sbc(SIMS, GENPDFS, posterior, Nsamples=5000, 
            timeout=40, 
            parameters_to_condition_on=parameters_to_condition_on)
        plot_sbc_ranks(ranks)
            