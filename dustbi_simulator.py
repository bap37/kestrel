from dataclasses import dataclass
import torch
from torch.distributions import Normal, LogNormal, Exponential
from sbi.utils import MultipleIndependent
import torch
import numpy as np
from sbi.utils import BoxUniform
from Functions import *
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_median

#Create the layout function to describe input parameters theta

@dataclass
class ThetaLayout:
    #Add_Function_Key
    gauss: slice
    exp: slice
    lognormal: slice
    double_gaussian: slice
    n_gauss: int
    n_lognormal: int
    n_double_gaussian: int

def build_theta_layout(n_gauss: int, n_exp: int, n_lognormal: int, n_double_gaussian: int) -> ThetaLayout:
    
    #Add_Function_Key
    
    idx = 0

    gauss_slice = slice(idx, idx + 2 * n_gauss)
    idx += 2 * n_gauss

    exp_slice = slice(idx, idx + n_exp)
    idx += n_exp
    
    lognormal_slice = slice(idx, idx + 2 * n_lognormal)
    idx += 2 * n_lognormal
    
    double_gaussian_slice = slice(idx, idx + 5 * n_double_gaussian)
    idx += 5 * n_double_gaussian
    
    return ThetaLayout(
        gauss=gauss_slice,
        exp=exp_slice,
        lognormal=lognormal_slice,
        double_gaussian=double_gaussian_slice,
        n_gauss=n_gauss,
        n_lognormal=n_lognormal,
        n_double_gaussian=n_double_gaussian
    )


def build_layout(param_names, dicts):
            
    #Add_Function_Key 
    n_gauss = 0 ; n_exp = 0 ; n_lognormal = 0; n_double_gaussian = 0
    bounds_dict, function_dict, split_dict, priors_dict = dicts

    
    #need to include the splitting ability 
    
    for name in param_names:
        
        if "HIGH" in name: name = name.split("_HIGH_")[0]
        funcname = function_dict[name].__name__    
        
        if   "DistGaussian"    == str(funcname): n_gauss += 1 ; 
        elif "Exponential"     in str(funcname): n_exp   += 1 ;
        elif "LogNormal"       in str(funcname): n_lognormal += 1;
        elif "Double"          in str(funcname): n_double_gaussian = 0
        #Add_Function_Key
    
    layout = build_theta_layout(n_gauss=n_gauss, n_exp=n_exp, 
                                n_lognormal = n_lognormal, 
                                n_double_gaussian = n_double_gaussian)
    
    return layout 
    

#######################
## BEGIN SIMULATOR
#######################

def simulator(theta: torch.Tensor, layout, param_names, parameters_to_condition_on,
               df, df_tensor, dicts, dfdata, is_split, debug=False):

    #Unravel a bunch of necessary information
    bounds_dict, function_dict, split_dict, priors_dict = dicts
    high_flag = True #Set split flag early in case we need to keep track of parameters in split_dict
    splits = list({v[0] for v in split_dict.values()}) #spools out split_dict parameters.

    salt_mcmc = start_distance()
    
    #torch dimensionality nonsense 
    if theta.ndim == 1: theta = theta.unsqueeze(0)
    batch_size = theta.shape[0] ; device = theta.device

    #set up error catching for simulator
    ndim = len(parameters_to_condition_on)
    if is_split: ndim *= 2
    BAD_SIMULATION = torch.full((len(dfdata), ndim), float('nan'), device=device)

    #Initialise weights
    N = len(df)
    joint_weights = torch.ones(batch_size, N, device=device)

    # --------------------------------------------------
    # Gaussian Parameters
    # --------------------------------------------------

    gauss_theta = theta[:, layout.gauss]
    gauss_theta = gauss_theta.view(batch_size, layout.n_gauss, 2)

    #Loops over all the Gaussian parameters to sample from.
    for i in range(layout.n_gauss):
        idx = layout.gauss.start + 2*i ; name = param_names[i]
    
        if "_HIGH_" in name:
            name = name.split("_HIGH_")[0] ; high_flag = True

        bounds = bounds_dict[name] ; shape = function_dict[name]
        
        x = df_tensor[name]  
        theta_g = gauss_theta[:, i, :]

        ########################
        #Start Split Logic

        if name in split_dict:

            split_param, split_val = split_dict[name]
            split_tensor = df_tensor[split_param]

            if high_flag:
                mask = split_tensor >= split_val
                high_flag = False
            else:
                mask = split_tensor < split_val

            weights = torch.ones(batch_size, N, device=device)

            x_sub = x[mask]
            density_sub = shape(x_sub, theta_g)
            density_sub = torch.clamp(density_sub, min=0.0)

            if density_sub.ndim == 1:
                density_sub = density_sub.unsqueeze(0)

            weights[:, mask] = density_sub



        #If no split, treat as-normal.
        else:
            density = shape(x, theta_g)  # expect (N,) or broadcastable
            weights = density 
            weights = torch.clamp(weights, min=0.0)
        
        ########################
        #End Split Logic

        if weights.ndim == 1: weights = weights.unsqueeze(0)  # shape = (1, N)
        joint_weights *= weights #Apply weight

        
    # --------------------------------------------------
    # Exponential Parameters
    # --------------------------------------------------

    #Same Logic as Gaussian, basically.
    
    exp_theta = theta[:, layout.exp]

    for i in range(exp_theta.shape[1]):
        name = param_names[layout.n_gauss + i]

        if "_HIGH_" in name:
            name = name.split("_HIGH_")[0]
            high_flag = True

        theta_e = exp_theta[:, i:i+1]

        bounds = bounds_dict[name]
        shape = function_dict[name]

        x = df_tensor[name]
        theta_e = exp_theta[:, i:i+1]

       ########################
        #Start Split Logic
        
        if name in split_dict:

            split_param, split_val = split_dict[name]
            split_tensor = df_tensor[split_param]

            if high_flag:
                mask = split_tensor >= split_val
                high_flag = False
            else:
                mask = split_tensor < split_val

            weights = torch.ones(batch_size, N, device=device)

            x_sub = x[mask]
            density_sub = shape(x_sub, theta_e)
            density_sub = torch.clamp(density_sub, min=0.0)

            if density_sub.ndim == 1:
                density_sub = density_sub.unsqueeze(0)

            weights[:, mask] = density_sub

                
        else:
            density = shape(x, theta_e)  # expect (N,) or broadcastable
            weights = density 
            weights = torch.clamp(weights, min=0.0)
        


        if weights.ndim == 1: weights = weights.unsqueeze(0)  # shape = (1, N)

        joint_weights *= weights


        
    # --------------------------------------------------
    # Additional Parameters
    # --------------------------------------------------

    #Will need to be added as more functions are included. Please follow the same logic as Gaussian parameters.
        

    # --------------------------------------------------
    # Normalise + Resample Once
    # --------------------------------------------------

    #The final weighted sum.
    weight_sum = joint_weights.sum(dim=1, keepdim=True)

    #Catch any shizwizz and return a bad simulation
    if weight_sum == 0:
        print("ERROR: weight_sum = 0 for", theta)
        return BAD_SIMULATION

    normalized_weights = joint_weights / weight_sum

    ess = 1.0 / torch.sum(normalized_weights ** 2, dim=1)
    n_samples = int(torch.ceil(ess).item())

    #The re-sampling occurs here; grabs all the desired indices that we've built up from our importance sampler.
    resampled_idx = torch.multinomial(
        normalized_weights,
        num_samples=n_samples,
        replacement=True
    )

    #Then dimension down and sample from the input data frame.
    indices = resampled_idx.squeeze(0).cpu().numpy()
    dft = df.iloc[indices]

    try:
        dft = dft.sample(n=len(dfdata))
    except ValueError:
        print("ERROR: not enough SNe for", theta)
        return BAD_SIMULATION #If there's not enough samples left, it's a bad simulation. 

    # --------------------------------------------------
    # Add distance into here
    # --------------------------------------------------

    output_distribution = preprocess_input_distribution(
        dft, parameters_to_condition_on+['x0', 'x0ERR', 'MU']
    )
    
    salt_mcmc.run(
        output_distribution['x0'],
        output_distribution['x0ERR'],
        output_distribution['x1'],
        output_distribution['x1ERR'],
        output_distribution['c'],
        output_distribution['cERR'],
        output_distribution['MU']
        )

    MURES = add_distance(salt_mcmc, output_distribution)
    output_distribution['MURES'] = MURES
    
    
    # --------------------------------------------------
    # Final Processing 
    # --------------------------------------------------


    #Check if any of the model parameters are split; if so, proceed to offer chopped distributions. 
    if is_split:

        matching = [p for p in param_names if p in split_dict]
        name = matching[0]
        
        split_param = split_dict[name][0]
        split_val   = split_dict[name][1]
    
    
        split_tensor = torch.tensor(
        dft[split_param].to_numpy(),
        dtype=torch.float32,
        device=device
    )
    
        x = split_outputs(
            output_distribution,
            split_tensor,
            split_val,
            parameters_to_condition_on
        )
        
    else:
        #And stack the conditioned parameters
        x = torch.stack(
            [output_distribution[p] for p in parameters_to_condition_on],
            dim=-1
        )
    
    #debug flag will helpfully return a pandas dataframe containing your desired distribution
    if debug:
        dft['MURES'] = MURES
        return dft
    
    return x


#####################
### BEGIN SUPPORT FUNCTIONS
###########################

def preprocess_input_distribution(df, cols):
    return {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32)
        for col in cols
    }

def make_simulator(layout, df, param_names, parameters_to_condition_on, dicts, dfdata, is_split, debug=False):

    _, function_dict, split_dict, priors_dict = dicts
    
    validate_order(param_names, function_dict) #force correct parameter order

    is_split = False
    splits = list({v[0] for v in split_dict.values()}) #spools out split_dict parameters.
    if any(p in split_dict for p in param_names): #check early to see if we need to split anything. 
        is_split = True
        print(f"Found a split in {split_dict.keys()}")
        
    params_to_fit = parameter_generation(param_names, dicts)

    df_tensor = {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32, )
        for col in list(priors_dict.keys())+splits+parameters_to_condition_on
    }
    
    def simulator_with_input(theta):
        return simulator(theta, layout, params_to_fit, parameters_to_condition_on, df, df_tensor, dicts, dfdata, is_split, debug)

    return simulator_with_input

def parameter_generation(list_of_parameter_names, dicts):
    
    empty_list = []
    bounds_dict, function_dict, split_dict, priors_dict = dicts
    
    for name in list_of_parameter_names:
        if name in split_dict.keys():
            split = (split_dict[name][0])
            empty_list.append(name+"_HIGH_"+split)
        empty_list.append(name)
    return empty_list


def validate_order(param_names, function_dict): 
    """
    Function that ensures the proper ordering of parameters; all exponential functions must come after all Gaussian functions.
    """
    
    #Will need to add other functions in as they are implemented; make sure that Exponential is always last 
    order_priority = {
        DistDoubleGaussian: 3,
        DistGaussian: 4,
        DistExponential: 5,
    }

    priorities = [order_priority[function_dict[p]] for p in param_names]

    if priorities != sorted(priorities):
        raise ValueError("Please ensure that any Exponential distribution strictly comes after all Gaussian distributions.")

    return True


def unspool_labels(list_of_parameter_names, dicts, latex_dict, function_dict):
    """
    Creates a set of labels for the parameters. Takes in your regular parameter names; outputs latex labels for everything. 
    """
    empty_list = []
    high_flag = False
    
    params_to_fit = parameter_generation(list_of_parameter_names, dicts)
    
    for name in params_to_fit:
        if "_HIGH_" in name:
            name = name.split("_HIGH_")[0]
            high_flag = True
   
        func_name = (function_dict[name].__name__)
        func_params = latex_dict[func_name] ; pname = latex_dict[name]
        
        for _ in func_params:
            if high_flag:
                latex_string =f'{pname} Hi {_}'
            else:
                latex_string =f'{pname} {_}'
            
            empty_list.append(latex_string)
        high_flag = False
    return empty_list


def split_outputs(output_distribution, split_tensor, split_val, param_list):
    """
    Splits the output tensors to return 4 parameters instead of 2; assumes that split_dict always splits on the same parameter.
    """
    mask_high = split_tensor >= split_val
    mask_low  = ~mask_high

    out = []
    for p in param_list:
        v = output_distribution[p]
        out.extend([
            torch.where(mask_low,  v, torch.zeros_like(v)),
            torch.where(mask_high, v, torch.zeros_like(v))
        ])

    return torch.stack(out, dim=-1)


def preprocess_data(param_names, parameters_to_condition_on, split_dict, dfdata, ):
    
    output_distribution = preprocess_input_distribution(dfdata, parameters_to_condition_on)

    if any(p in split_dict for p in param_names): #check early to see if we need to split anything. 
    
        matching = [p for p in param_names if p in split_dict]
        name = matching[0]

        split_param = split_dict[name][0]
        split_val   = split_dict[name][1]

        split_tensor = torch.tensor(
            dfdata[split_param].to_numpy(),
            dtype=torch.float32,
            )

        x = split_outputs(
            output_distribution,
            split_tensor,
            split_val,
            parameters_to_condition_on
            )

    else:
        x = torch.stack(
            [output_distribution[p] for p in parameters_to_condition_on],
            dim=-1
        )
        
    return x


def load_kestrel(filename):
    
    import ast
    import yaml
    
    with open(filename, 'r') as file:
        raw_yaml = yaml.safe_load(file)

    raw_yaml['param_names'] = raw_yaml['param_names'].split(" ")
        
    for entry in raw_yaml['Boundaries']:
        raw_yaml['Boundaries'][entry] = tuple(raw_yaml['Boundaries'][entry])
        
    _priors = raw_yaml['Priors']
    for entry in _priors:
        for n in range(len(_priors[entry])):
            raw_yaml['Priors'][entry][n] = ast.literal_eval(raw_yaml['Priors'][entry][n])
            
    return raw_yaml

def load_data(simfilename, datfilename):

    import numpy as np
    import pandas as pd
    df = pd.read_csv(simfilename, comment="#", sep='\s+')

    df['SIM_EBV'] = df.SIM_AV/df.SIM_RV


    dfdata = pd.read_csv(datfilename, 
                             comment="#", sep=r'\s+')

    dfdata = dfdata.loc[dfdata.IDSURVEY == 10]
    dfdata = dfdata.loc[dfdata.PROB_SNNV19 >= 0.5]

    return df, dfdata

def train_model(n_sim, n_batch, sims_savename, priors, simulatinator, inference):
    import os
    from tqdm import tqdm

    def batched_simulator(theta_batch):
        return torch.stack([simulatinator(theta) for theta in theta_batch])
    
    batch_size = n_batch
    num_simulations = n_sim
    save_path = sims_savename

    # If the file already exists, start fresh
    if os.path.exists(save_path):
        os.remove(save_path)

    with tqdm(total=num_simulations, desc="Running simulations", unit="sim") as pbar:
        for start in range(0, num_simulations, batch_size):
            current_bs = min(batch_size, num_simulations - start)

            theta_batch = priors.sample((current_bs,))
            x_batch = batched_simulator(theta_batch)
            inference.append_simulations(theta_batch, x_batch)

            if start == 0:
                # First batch, create the file
                torch.save({'theta': theta_batch, 'x': x_batch}, save_path)
            else:
                # Load existing data
                data = torch.load(save_path)
                data['theta'] = torch.cat([data['theta'], theta_batch], dim=0)
                data['x'] = torch.cat([data['x'], x_batch], dim=0)
                torch.save(data, save_path)

            # Update progress bar
            pbar.update(current_bs)
            pbar.set_postfix(saved=start + current_bs)

    print(f"All simulations saved incrementally to '{save_path}'")

def add_distance(mcmc, df_tensor):
    
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    
    nuisance = mcmc.get_samples()
    beta = nuisance['beta'].mean() ; alpha = nuisance['alpha'].mean() ; M0 = nuisance['M'].mean()
    
    correction = alpha * x1_obs - beta * c_obs - M0
        
    MURES = df_tensor['MU'] - correction
    
    return  MURES
    
def start_distance(NUM_WARMUP = 50, NUM_SAMPLES = 150, NUM_CHAINS = 1):

    nuts_kernel = NUTS(
        distancinator,
        jit_compile=True,
        init_strategy=init_to_median(),
        max_tree_depth=10
        )
    
    salt_mcmc = MCMC(
        nuts_kernel,
        warmup_steps=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS
    )
    
    return salt_mcmc

def distancinator(x0_obs, x0_err, x1_obs, x1_err, c_obs, c_err, dist_mod):

    n = dist_mod.shape[0]

    alpha = pyro.sample("alpha", dist.Normal(0.1, 1.0))
    beta = pyro.sample("beta", dist.Normal(2.0, 3.0))
    M = pyro.sample("M", dist.Uniform(-21.5, -17.0))
    sigma_int = pyro.sample("sigma_int", dist.HalfNormal(0.3))

    with pyro.plate("sne", n):
        log10_x0 = pyro.sample("log10_x0", dist.Normal(-3.0, 2.0))
        x0_true = torch.pow(10.0, log10_x0)

        pyro.sample("x0_obs",
                    dist.Normal(x0_true, x0_err),
                    obs=x0_obs)

        correction = alpha * x1_obs - beta * c_obs - M

        mag_err = (2.5 / torch.log(torch.tensor(10.0))) * (x0_err / x0_true)
        total_err = torch.sqrt(mag_err**2 + sigma_int**2 + x1_err**2 + c_err**2)

        mean_mag = -2.5 * torch.log10(x0_true) + 10.635 + correction
        
        pyro.sample("cosmo",
                    dist.Normal(mean_mag, total_err),
                    obs=dist_mod)

####################
## BEGIN PRIORS
######################

def prior_generator(param_names, dicts):

    bounds_dict, function_dict, split_dict, priors_dict = dicts
    list_o_priors = []
    
    for name in param_names:
        if "_HIGH_" in name:
            name = name.split("_HIGH_")[0]

        func_name = function_dict[name].__name__        
        if func_name == "DistGaussian":
            mu0, sigma0 = priors_dict[name]

            mu_prior = BoxUniform(
                low= torch.tensor([mu0[0]], dtype=torch.float32), 
                high=torch.tensor([mu0[1]], dtype=torch.float32)
                )

            sigma_prior = BoxUniform(
                low= torch.tensor([sigma0[0]], dtype=torch.float32),
                high=torch.tensor([sigma0[1]], dtype=torch.float32)
                )

            list_o_priors.extend([mu_prior, sigma_prior])

            if name in split_dict:
                list_o_priors.extend([mu_prior, sigma_prior])

                
        elif func_name == "DistExponential":
            tau0 = priors_dict[name][0]

            tau_prior = BoxUniform(
                low= torch.tensor([tau0[0]], dtype=torch.float32),
                high=torch.tensor([tau0[1]], dtype=torch.float32)
                    )

            list_o_priors.append(tau_prior)

            if name in split_dict:
                list_o_priors.append(tau_prior)


        if func_name == "DistDoubleGaussian":
            mu1, sigma1, mu2, sigma2, a, need_positive = priors_dict[name]

            mu1_prior = BoxUniform(
                low= torch.tensor([mu1[0]], dtype=torch.float32), 
                high=torch.tensor([mu1[1]], dtype=torch.float32)
                )

            sigma1_prior = BoxUniform(
                low= torch.tensor([sigma1[0]], dtype=torch.float32),
                high=torch.tensor([sigma1[1]], dtype=torch.float32)
                )

            mu2_prior = BoxUniform(
                low= torch.tensor([mu2[0]], dtype=torch.float32), 
                high=torch.tensor([mu2[1]], dtype=torch.float32)
                )

            sigma2_prior = BoxUniform(
                low= torch.tensor([sigma2[0]], dtype=torch.float32),
                high=torch.tensor([sigma2[1]], dtype=torch.float32)
                )

            a_prior = BoxUniform(
                low= torch.tensor([a[0]], dtype=torch.float32),
                high=torch.tensor([a[1]], dtype=torch.float32)
                )
            
            
            list_o_priors.extend([mu1_prior, sigma1_prior, mu2_prior, sigma2_prior, a_prior])

            if name in split_dict:
                list_o_priors.extend([mu1_prior, sigma1_prior, mu2_prior, sigma2_prior, a_prior])

                
    print("Total priors added:", len(list_o_priors))
    for i, p in enumerate(list_o_priors):
        print([i], type(p))

                
    return MultipleIndependent(list_o_priors)
