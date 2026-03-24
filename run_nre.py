from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
from dustbi_plotting import plot_loss
import yaml, os, argparse
import shutil
import pickle
from sbi import inference as sbi_inference

from astropy.cosmology import Planck18

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import torch

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

from sbi.utils import MultipleIndependent

def add_distance(df_tensor):
    
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    
    beta = 3.1 ; alpha = 0.16 ; M0 = -19.3
    
    correction = alpha * x1_obs - beta * c_obs + M0 + mB_obs
        
    MURES = df_tensor['MU'] - correction
    
    return  MURES

simfilename = 'SIM_BANK_SB.FITRES.gz'
datfilename = 'SIMS_FOR_TESTING/FITOPT000.FITRES.gz'


def get_args():
    
    parser = argparse.ArgumentParser()

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

    parameters_to_condition_on = infos['parameters_to_condition_on']

    df, dfdata = load_data(simfilename, datfilename)

    num_simulations = 3000
    ##############################
    # Load information and setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parameters_to_condition_on = infos['parameters_to_condition_on']

    output_distribution = preprocess_input_distribution(
        df, parameters_to_condition_on[:-1]+['x0', 'x0ERR', 'MU'])
    MURES_sims = add_distance(output_distribution)
    df['MURES'] = MURES_sims

    output_distribution = preprocess_input_distribution(
        dfdata, parameters_to_condition_on[:-1]+['x0', 'x0ERR', 'MU'])
    MURES_sims = add_distance(output_distribution)
    dfdata['MURES'] = MURES_sims

    x_obs = preprocess_data(parameters_to_condition_on, dfdata)

    #Load in nominal information and simulate it 
    dicts = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations']]
    param_names = infos['param_names']
    params_to_fit = parameter_generation(param_names, dicts)
    priors_1 = prior_generator(param_names, dicts)
    layout = build_layout(params_to_fit, dicts)
    parameters_to_condition_on = infos['parameters_to_condition_on']
    nominal_model = make_batched_simulator(layout, df,
                            param_names,parameters_to_condition_on,
                            dicts, dfdata, sub_batch=500, device='cpu')
    batched = True
    theta_1 = priors_1.sample((num_simulations,))
    x1 = nominal_model(theta_1)  # (batch_size, N, input_dim)

    for model in infos['Models_Comparison']:
        infos = load_kestrel(model)
        dicts = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations']]
        param_names = infos['param_names']
        params_to_fit = parameter_generation(param_names, dicts)
        priors_2 = prior_generator(param_names, dicts)
        layout = build_layout(params_to_fit, dicts)
        parameters_to_condition_on = infos['parameters_to_condition_on']
        log_model = make_batched_simulator(layout, df,
                                param_names,parameters_to_condition_on,
                                dicts, dfdata, sub_batch=500, device='cpu')
        batched = True
        theta_2 = priors_2.sample((num_simulations,))
        x2 = log_model(theta_2)  # (batch_size, N, input_dim)

        # Mask improper simulations
        mask1 = torch.isfinite(x1).all(dim=(1, 2))  # True if all entries along (seq_len, dim_x) are finite
        x1_clean = x1[mask1]         # shape: (num_valid_sims, 5720, 10)
        theta1_clean = theta_1[mask1] # shape: (num_valid_sims, dim_theta)
        print(f"{args.CONFIG}: {x1_clean.shape[0]} valid simulations out of {x1.shape[0]}")


        mask2 = torch.isfinite(x2).all(dim=(1, 2))  # True if all entries along (seq_len, dim_x) are finite
        x2_clean = x2[mask2]         # shape: (num_valid_sims, 5720, 10)
        theta2_clean = theta_2[mask2] # shape: (num_valid_sims, dim_theta)
        print(f"{model}: {x2_clean.shape[0]} valid simulations out of {x2.shape[0]}")

        num_sims = x1_clean.shape[0]
        x1_flat = x1_clean.reshape(num_sims, -1)  # shape: (num_sims, 5720*10)
        theta1_flat = theta1_clean  # leave theta as is

        num_sims2 = x2_clean.shape[0]
        x2_flat = x2_clean.reshape(num_sims2, -1)
        theta2_flat = theta2_clean

        # Set up inference objects
        inference1 = sbi_inference.SNRE_A(priors_1)
        inference2 = sbi_inference.SNRE_A(priors_2)

        # Append simulations (can append more than observed for robustness)
        inference1.append_simulations(theta1_flat, x1_flat)
        inference2.append_simulations(theta2_flat, x2_flat)

        # Train the ratio estimators
        ratio_estimator1 = inference1.train()
        ratio_estimator2 = inference2.train()

        num_mc_samples = 5000

        # Suppose x1_clean had shape (num_sims, 5720, 10) and was flattened during training:
        x_obs_flat = x_obs.reshape(1, -1)  # 1 x 57200

        # x_obs_flat: shape (1, features)
        x_obs_repeated = x_obs_flat.repeat(num_mc_samples, 1)  # repeat for all theta samples

        theta_samples1 = priors_1.sample((num_mc_samples,))
        theta_samples2 = priors_2.sample((num_mc_samples,))

        # Evaluate ratio estimator directly
        eps = 1e-12  # small epsilon to avoid log(0)
        r1 = ratio_estimator1(theta_samples1, x_obs_repeated).clamp(min=eps)
        r2 = ratio_estimator2(theta_samples2, x_obs_repeated).clamp(min=eps)

        log_r1 = torch.log(r1 + 1e-10).squeeze()
        log_r2 = torch.log(r2 + 1e-10).squeeze()

        log_marginal1 = torch.logsumexp(log_r1, dim=0) - torch.log(torch.tensor(num_mc_samples, dtype=torch.float))
        log_marginal2 = torch.logsumexp(log_r2, dim=0) - torch.log(torch.tensor(num_mc_samples, dtype=torch.float))

        BF_12 = torch.exp(log_marginal1 - log_marginal2)
        print("\n")
        print(f"Robust Bayes Factor {args.CONFIG}/{model}:", BF_12.item())