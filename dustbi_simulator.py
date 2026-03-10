from dataclasses import dataclass
import torch
from torch.distributions import Normal, LogNormal, Exponential, HalfNormal
from sbi.utils import MultipleIndependent
import numpy as np
from sbi.utils import BoxUniform
from Functions import *
#import pyro
#import pyro.distributions as dist
#from pyro.infer import MCMC, NUTS
#from pyro.infer.autoguide.initialization import init_to_median

#Create the layout function to describe input parameters theta
from dataclasses import dataclass

@dataclass
class ThetaLayout:
    slices: dict
    idx: dict
    counts: dict
    n_params: dict
    order: dict

def build_theta_layout(counts, n_params):

    slices = {}
    idx = 0

    for dist in counts:

        width = counts[dist] * n_params[dist]
        slices[dist] = slice(idx, idx + width)
        idx += width

    return slices

    
def build_layout(param_names, dicts):

    bounds_dict, function_dict, split_dict, priors_dict = dicts

    idx = {
        "gauss": [],
        "exp": [],
        "lognormal": [],
        "double_gaussian": [],
        "linear": [],
        "stepwise": [],
    }

    order = {
        "gauss": [],
        "exp": [],
        "lognormal": [],
        "double_gaussian": [],
        "linear": [],
        "stepwise": [],
    }

    for i, name in enumerate(param_names):

        base = name.split("_HIGH_")[0]
        funcname = function_dict[base].__name__

        if funcname == "DistGaussian":
            idx["gauss"].append(i)
            order["gauss"].append(name)

        elif "Exponential" in funcname:
            idx["exp"].append(i)
            order["exp"].append(name)

        elif "LogNormal" in funcname:
            idx["lognormal"].append(i)
            order["lognormal"].append(name)

        elif "Double" in funcname:
            idx["double_gaussian"].append(i)
            order["double_gaussian"].append(name)

        elif "Linear" in funcname:
            idx["linear"].append(i)
            order["linear"].append(name)

        elif "Stepwise" in funcname:
            idx["stepwise"].append(i)
            order["stepwise"].append(name)

    counts = {k: len(v) for k, v in idx.items()}

    n_params = {
        "gauss": 2,
        "exp": 1,
        "lognormal": 2,
        "double_gaussian": 5,
        "linear": 2,
        "stepwise": 1,
    }

    slices = build_theta_layout(counts, n_params)

    return ThetaLayout(
        slices=slices,
        idx=idx,
        counts=counts,
        n_params=n_params,
        order=order
    )    

#######################
## BEGIN SIMULATOR
#######################

def simulator(theta: torch.Tensor, layout, param_names, parameters_to_condition_on,
               df, df_tensor, dicts, dfdata, debug=False):

    #Unravel a bunch of necessary information
    bounds_dict, function_dict, split_dict, priors_dict = dicts
    high_flag = True #Set split flag early in case we need to keep track of parameters in split_dict

    #salt_mcmc = start_distance()
    
    #torch dimensionality nonsense 
    if theta.ndim == 1: theta = theta.unsqueeze(0)
    batch_size = theta.shape[0] ; device = theta.device

    #set up error catching for simulator
    numdim = len(parameters_to_condition_on)#+1 BRODIE 
    BAD_SIMULATION = torch.full((len(dfdata), numdim), float('nan'), device=device)

    #Initialise weights
    N = len(df)
    joint_weights = torch.ones(batch_size, N, device=device)

    #########################
    # Once without if/then statements
    #########################
    for dist in layout.slices:

        if layout.counts[dist] == 0:
            continue 
        
        theta_dist = theta[:, layout.slices[dist]]
        n_param = layout.n_params[dist]

        theta_dist = theta_dist.view(batch_size, layout.counts[dist], n_param)

        for i in range(layout.counts[dist]):

            param_index = layout.idx[dist][i]
            name = param_names[param_index]

            if "_HIGH_" in name:
                name = name.split("_HIGH_")[0]
                high_flag = True

            shape = function_dict[name]
            x = df_tensor[name]

            theta_i = theta_dist[:, i, :]

            ########################
            # Split Logic
            ########################

            if name in split_dict:

                _, split_param, split_val = split_dict[name]
                split_tensor = df_tensor[split_param]

                if high_flag:
                    mask = split_tensor >= split_val
                    high_flag = False
                else:
                    mask = split_tensor < split_val

                weights = torch.ones(batch_size, N, device=device)

                x_sub = x[mask]
                density_sub = shape(x_sub, theta_i)
                density_sub = torch.clamp(density_sub, min=0.0)

                if density_sub.ndim == 1:
                    density_sub = density_sub.unsqueeze(0)

                weights[:, mask] = density_sub

            else:

                density = shape(x, theta_i)
                weights = torch.clamp(density, min=0.0)

            if weights.ndim == 1:
                weights = weights.unsqueeze(0)

            joint_weights *= weights

    # --------------------------------------------------
    # Normalise + Resample Once
    # --------------------------------------------------

    #The final weighted sum.
    weight_sum = joint_weights.sum(dim=1, keepdim=True)

    #Catch any shizwizz and return a bad simulation
    if weight_sum == 0:
        if debug: print("ERROR: weight_sum = 0 for", theta)
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
        if debug: print("ERROR: not enough SNe for", theta)
        return BAD_SIMULATION #If there's not enough samples left, it's a bad simulation. 

    # --------------------------------------------------
    # Add distance into here
    # --------------------------------------------------

    output_distribution = preprocess_input_distribution(
        dft, parameters_to_condition_on#+['x0', 'x0ERR', 'MU']
    )
    #BRODIE
    #salt_mcmc.run(
    #    output_distribution['x0'],
    #    output_distribution['x0ERR'],
    #    output_distribution['x1'],
    #    output_distribution['x1ERR'],
    #    output_distribution['c'],
    #    output_distribution['cERR'],
    #    output_distribution['MU']
    #    )

    #MURES = add_distance(salt_mcmc, output_distribution)
    #output_distribution['MURES'] = MURES
    
    
    # --------------------------------------------------
    # Final Processing 
    # --------------------------------------------------

    #parameters_to_condition_on = parameters_to_condition_on+['MURES']
    #And stack the conditioned parameters
    x = torch.stack(
    [output_distribution[p] for p in parameters_to_condition_on],
    dim=-1
)
    
    #debug flag will helpfully return a pandas dataframe containing your desired distribution
    if debug:
        #dft['MURES'] = MURES
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

def make_simulator(layout, df, param_names, parameters_to_condition_on, dicts, dfdata, debug=False, device="cpu"):

    _, function_dict, split_dict, priors_dict = dicts
    
    validate_order(param_names, function_dict) #force correct parameter order

    splits = list({v[1] for v in split_dict.values()}) #spools out split_dict parameters.


    params_to_fit = parameter_generation(param_names, dicts)

    df_tensor = {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32, device=device)
        for col in list(priors_dict.keys())+splits+parameters_to_condition_on
    }

    def simulator_with_input(theta):
        return simulator(theta, layout, params_to_fit, parameters_to_condition_on, df, df_tensor, dicts, dfdata, debug)

    return simulator_with_input


def make_batched_simulator(layout, df, param_names, parameters_to_condition_on,
                           dicts, dfdata, sub_batch=10, device="cpu"):
    bounds_dict, function_dict, split_dict, priors_dict = dicts
    validate_order(param_names, function_dict)

    splits = list({v[1] for v in split_dict.values()}) #spools out split_dict parameters.

    # Pre-compute ALL tensor columns ONCE
    all_cols = list(set(list(priors_dict.keys()) + splits + parameters_to_condition_on))
    all_cols.remove("STEP")
    df_tensor = {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32, device=device)
        for col in all_cols
    }

    # Pre-stack output columns for fast batched indexing
    output_stack = torch.stack(
        [df_tensor[col] for col in parameters_to_condition_on], dim=-1
    )  # (N, n_features)

    #Calculate indices for things that need the "mass" step added to them. 
    steps_to_add = ['MU', 'MURES', 'mB']
    step_indices = torch.tensor(
        [parameters_to_condition_on.index(c) for c in steps_to_add],
        dtype=torch.long,
        device=device
    )

    y_idx = parameters_to_condition_on.index(split_dict["STEP"][1])
    step_threshold = split_dict["STEP"][2]


    N = len(df)
    n_target = len(dfdata)

    def _simulate_sub_batch(theta):
        """Fully batched: theta is (B, n_params), returns (B, n_target, n_features)."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        B = theta.shape[0]
        device = theta.device

        joint_weights = torch.ones(B, N, device=device)
        high_flag = False
        
        # --- Trying to replace Matt's stuff
        for dist in layout.slices:
            if layout.counts[dist] == 0:
                continue 
            theta_dist = theta[:, layout.slices[dist]]
            n_param = layout.n_params[dist]
            theta_dist = theta_dist.view(B, layout.counts[dist], n_param)

            for i in range(layout.counts[dist]):
                name = layout.order[dist][i]

                
                if "_HIGH_" in name:
                    name = name.split("_HIGH_")[0]
                    high_flag = True
                    
                theta_i = theta_dist[:, i, :]
                x = df_tensor[name]  # (N,)
                batch_size = B

                if name in split_dict:
                    _, split_param, split_val = split_dict[name]
                    split_tensor = df_tensor[split_param]

                    if high_flag:
                        mask = split_tensor >= split_val
                        high_flag = False
                    else:
                        mask = split_tensor < split_val
                    
                    weights = torch.ones(batch_size, N, device=device)

                    x_sub = x[mask]  # still on GPU
                    density_sub = function_dict[name](x_sub, theta_i)
                    density_sub = torch.clamp(density_sub, min=0.0)

                    if density_sub.ndim == 1:
                        density_sub = density_sub.unsqueeze(0)
                    weights[:, mask] = density_sub

                else:
                    density = function_dict[name](x, theta_i)
                    weights = torch.clamp(density, min=0.0)

                if weights.ndim == 1:
                    weights = weights.unsqueeze(0)

                joint_weights *= weights

        # --- Normalise ---
        weight_sum = joint_weights.sum(dim=1, keepdim=True)  # (B, 1)
        bad_mask = (weight_sum.squeeze(1) == 0)

        joint_weights[bad_mask] = 1.0
        weight_sum = torch.where(weight_sum == 0, torch.tensor(float(N)), weight_sum)
        normalized_weights = joint_weights / weight_sum

        # --- ESS check: mark low-ESS draws as bad ---
        ess = 1.0 / torch.sum(normalized_weights ** 2, dim=1)  # (B,)
        bad_mask = bad_mask | (ess < n_target)

        # --- Batched multinomial: draw n_target samples per theta ---
        resampled_idx = torch.multinomial(
            normalized_weights, num_samples=n_target, replacement=True
        )  # (B, n_target)

        # --- Batched output tensor indexing ---
        result = output_stack[resampled_idx]  # (B, n_target, n_features)

        #then add whatever proposed step is necessary.
        if "STEP" in param_names:
            gamma = theta[:, -1].unsqueeze(1)       # (B,1)
            y = result[:, :, y_idx]                 # (B, n_target)
            step = torch.where(y < step_threshold, -gamma, gamma)  # (B, n_target)
            result[:, :, step_indices] += step.unsqueeze(-1)

        # --- Fill bad simulations with NaN ---
        result[bad_mask] = float('nan')

        return result

    
    def batched_simulator(theta_batch):
        """Process theta_batch in sub-batches to control memory."""
        B = theta_batch.shape[0]
        if B <= sub_batch:
            return _simulate_sub_batch(theta_batch)

        results = []
        for start in range(0, B, sub_batch):
            end = min(start + sub_batch, B)
            results.append(_simulate_sub_batch(theta_batch[start:end]))
        return torch.cat(results, dim=0)

    return batched_simulator


    
def parameter_generation(list_of_parameter_names, dicts):
    
    empty_list = []
    bounds_dict, function_dict, split_dict, priors_dict = dicts
    
    for name in list_of_parameter_names:
        if name in split_dict.keys():
            evol_type = (split_dict[name][0])
            if evol_type == 'Stepwise':
                split = (split_dict[name][1])
                empty_list.append(name+"_HIGH_"+split)
            elif evol_type == 'Linear':
                empty_list.append(name+"_EVOL_"+split)
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

    #Temporarily strip "step" from param names, since it's implemented differently.
    new_list = param_names.copy()
    new_list.remove("STEP")

    priorities = [order_priority[function_dict[p]] for p in new_list]

    if priorities != sorted(priorities):
        raise ValueError("Please ensure that any Exponential distribution strictly comes after all Gaussian distributions.")
    elif param_names[-1] != "STEP":
        raise ValueError("Please ensure that the step parameter is the last entry in param_names.")
        
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
        raw_yaml['Boundaries'][entry] = ast.literal_eval(raw_yaml['Boundaries'][entry])
        
    _priors = raw_yaml['Priors']
    for entry in _priors:
        for n in range(len(_priors[entry])):
            raw_yaml['Priors'][entry][n] = ast.literal_eval(raw_yaml['Priors'][entry][n])
            
    return raw_yaml

def load_data(simfilename, datfilename):

    from astropy.cosmology import Planck18
    import numpy as np
    import pandas as pd
    df = pd.read_csv(simfilename, comment="#", sep='\s+')

    df['SIM_EBV'] = df.SIM_AV/df.SIM_RV
    df['MU'] = Planck18.distmod(df.zHD.values).value


    
    dfdata = pd.read_csv(datfilename, 
                             comment="#", sep=r'\s+')
    dfdata['MU'] = Planck18.distmod(dfdata.zHD.values).value

    
    dfdata = dfdata.loc[dfdata.IDSURVEY == 10]
    dfdata = dfdata.loc[dfdata.PROB_SNNV19 >= 0.5]

    return df, dfdata

def standardise_data(dft, dfdata, parameters_to_condition_on, param_names):

    return_dict = {}
    
    import numpy as np

    for param in parameters_to_condition_on:
        if "ERR" in param:
            dfdata[param] = np.log(dfdata[param].values)
            meanval = np.mean(dfdata[param].values)
            stdval  = np.std(dfdata[param].values)
            dfdata[param] = (dfdata[param] - meanval)/stdval

            return_dict[param+"_data"] = [meanval, stdval]
            
            dft[param] = np.log(dft[param].values)
            meanval = np.mean(dft[param].values)
            stdval  = np.std(dft[param].values)
            dft[param] = (dft[param] - meanval)/stdval

            return_dict[param+"_sim"] = [meanval, stdval]
        else:
            #Data
            meanval = np.mean(dfdata[param].values)
            stdval  = np.std(dfdata[param].values)
            dfdata[param] = (dfdata[param] - meanval)/stdval

            return_dict[param+"_data"] = [meanval, stdval]

            #sim
            meanval = np.mean(dft[param].values)
            stdval  = np.std(dft[param].values)
            dft[param] = (dft[param] - meanval)/stdval

            return_dict[param+"_sim"] = [meanval, stdval]

        #Repeat once to standardise input parameters; only applies to the sims
        for param in param_names:
            meanval = np.mean(dft[param].values)
            stdval  = np.std(dft[param].values)
            dft[param] = (dft[param] - meanval)/stdval

            return_dict[param+"_sim"] = [meanval, stdval]

    return dft, dfdata, return_dict



def simulate_model(n_sim, n_batch, sims_savename, priors, simulator, inference, device="cpu", batched=True):
    import h5py
    from tqdm import tqdm

    with h5py.File(sims_savename, "w") as f:
        
        theta_dim = priors.sample((1,)).shape[-1]
        x_example = simulator(priors.sample((1,)).to(device))
        x_shape = x_example.shape[1:] if x_example.ndim > 1 else (x_example.shape[-1],)

        theta_ds = f.create_dataset(
            "theta",
            shape=(0, theta_dim),
            maxshape=(None, theta_dim),
            dtype="float32",
            chunks=True,
        )

        x_ds = f.create_dataset(
            "x",
            shape=(0, *x_shape),
            maxshape=(None, *x_shape),
            dtype="float32",
            chunks=True,
        )

        cursor = 0

        with tqdm(total=n_sim, desc="Running simulations", unit="sim") as pbar:
            for start in range(0, n_sim, n_batch):

                current_bs = min(n_batch, n_sim - start)

                theta_batch = priors.sample((current_bs,)).to(device)

                if batched:
                    x_batch = simulator(theta_batch)
                else:
                    x_batch = torch.stack([simulator(t) for t in theta_batch])

                theta_np = theta_batch.cpu().numpy()
                x_np = x_batch.cpu().numpy()

                # resize datasets
                theta_ds.resize(cursor + current_bs, axis=0)
                x_ds.resize(cursor + current_bs, axis=0)

                # write batch
                theta_ds[cursor:cursor + current_bs] = theta_np
                x_ds[cursor:cursor + current_bs] = x_np

                cursor += current_bs
                pbar.update(current_bs)


def train_spne(prior, x):

    from sbi.inference import SNPE, simulate_for_sbi
    from sbi.utils import process_prior

    num_rounds = 5
    num_simulations = 500

    proposal = prior

    for round in range(num_rounds):

        theta, x_sim = simulate_for_sbi(
            batched_simulator,
            proposal,
            num_simulations=num_simulations,
            )

        inference = inference.append_simulations(theta, x_sim, proposal=proposal)

        density_estimator = inference.train()

        posterior = inference.build_posterior(density_estimator)

        proposal = posterior.set_default_x(x)
    
    return posterior
    

##########################
# MURES Calculation code
# (Currently unused)
##########################


def add_distance(mcmc, df_tensor):
    
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    
    nuisance = mcmc.get_samples()
    beta = nuisance['beta'].mean() ; alpha = nuisance['alpha'].mean() ; M0 = nuisance['M'].mean()
    
    correction = alpha * x1_obs - beta * c_obs + M0 + mB_obs
        
    MURES = df_tensor['MU'] - correction
    
    return  MURES
    
def start_distance(NUM_WARMUP = 50, NUM_SAMPLES = 150, NUM_CHAINS = 1):

    nuts_kernel = NUTS(
        distancinator,
        init_strategy=init_to_median(),
        max_tree_depth=10
        )
    
    salt_mcmc = MCMC(
        nuts_kernel,
        warmup_steps=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        disable_progbar=True,
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

def prior_generator(param_names, dicts, device='cpu'):

    bounds_dict, function_dict, split_dict, priors_dict = dicts
    list_o_priors = []
    
    for name in param_names:
        #We want to parse evolution parameters separately, so break the loop if we find one. 
        if "EVOL" in name:
            continue 
        if "STEP" in name:
            continue 

        if "_HIGH_" in name:
            name = name.split("_HIGH_")[0]

        func_name = function_dict[name].__name__        
        
        if func_name == "DistGaussian":
            mu0, sigma0 = priors_dict[name]
            mu_prior, sigma_prior = TwoDBoxPrior(mu0, sigma0, device=device)
            list_o_priors.extend([mu_prior, sigma_prior])
            #EVOL parameters are assessed here. 
            if name in split_dict:
                evol_type = (split_dict[name][0])
                if evol_type == "Stepwise":
                    list_o_priors.extend([mu_prior, sigma_prior])
                elif evol_type == "Linear":
                    offset0, slope0 = priors_dict[name+"_EVOL"]
                    offset_prior, slope_prior = TwoDBoxPrior(offset0, slope0)
                    list_o_priors.extend([offset_prior, slope_prior])

                
        elif func_name == "DistExponential":
            tau0 = priors_dict[name][0]

            tau_prior = BoxUniform(
                low= torch.tensor([tau0[0]], dtype=torch.float32, device=device),
                high=torch.tensor([tau0[1]], dtype=torch.float32, device=device)
                    )
            
            list_o_priors.append(tau_prior)

            if name in split_dict:
                evol_type = (split_dict[name][0])
                if evol_type == "Stepwise":
                    list_o_priors.extend([tau_prior])
                elif evol_type == "Linear":
                    offset0, slope0 = priors_dict[name+"_EVOL"]
                    offset_prior, slope_prior = TwoDBoxPrior(offset0, slope0)
                    list_o_priors.extend([offset_prior, slope_prior])


        if func_name == "DistDoubleGaussian":
            mu1, sigma1, mu2, sigma2, a, need_positive = priors_dict[name]

            mu1_prior = BoxUniform(
                low= torch.tensor([mu1[0]], dtype=torch.float32, device=device), 
                high=torch.tensor([mu1[1]], dtype=torch.float32, device=device)
                )

            sigma1_prior = BoxUniform(
                low= torch.tensor([sigma1[0]], dtype=torch.float32, device=device),
                high=torch.tensor([sigma1[1]], dtype=torch.float32, device=device)
                )

            mu2_prior = BoxUniform(
                low= torch.tensor([mu2[0]], dtype=torch.float32, device=device), 
                high=torch.tensor([mu2[1]], dtype=torch.float32, device=device)
                )

            sigma2_prior = BoxUniform(
                low= torch.tensor([sigma2[0]], dtype=torch.float32, device=device),
                high=torch.tensor([sigma2[1]], dtype=torch.float32, device=device)
                )

            a_prior = BoxUniform(
                low= torch.tensor([a[0]], dtype=torch.float32, device=device),
                high=torch.tensor([a[1]], dtype=torch.float32, device=device)
                )
            
            
            list_o_priors.extend([mu1_prior, sigma1_prior, mu2_prior, sigma2_prior, a_prior])

            if name in split_dict:
                evol_type = (split_dict[name][0])
                if evol_type == "Stepwise":
                    list_o_priors.extend([mu1_prior, sigma1_prior, mu2_prior, sigma2_prior, a_prior])
                elif evol_type == "Linear":
                    offset0, slope0 = priors_dict[name+"_EVOL"]
                    offset_prior, slope_prior = TwoDBoxPrior(offset0, slope0)
                    list_o_priors.extend([offset_prior, slope_prior])


    if "STEP" in param_names:
        step0 = priors_dict["STEP"][0]

        step_prior = BoxUniform(
            low= torch.tensor([step0[0]], dtype=torch.float32, device=device), 
            high=torch.tensor([step0[1]], dtype=torch.float32, device=device)
            )
        list_o_priors.extend([step_prior])


    print(f"Added {len(list_o_priors)} priors")
                    
    return MultipleIndependent(list_o_priors, device=device)

def TwoDBoxPrior(param_1, param_2, device="cpu"):
    """
    General wrapper for any distribution with two parameters.
    """

    p1_prior = BoxUniform(
        low= torch.tensor([param_1[0]], dtype=torch.float32, device=device),
        high=torch.tensor([param_1[1]], dtype=torch.float32, device=device)
        )

    p2_prior = BoxUniform(
        low= torch.tensor([param_2[0]], dtype=torch.float32, device=device),
        high=torch.tensor([param_2[1]], dtype=torch.float32, device=device)
        )

    return p1_prior, p2_prior


def TwoDGaussianPrior(param_1, param_2, device="cpu"):
    """
    theta[0] ~ mean value, error on mean value
    theta[1] ~ standard deviation of the Gaussian
    """

    p1_prior = Normal(
        loc=torch.tensor(param_1[0], dtype=torch.float32, device=device),
        scale=torch.tensor(param_1[1], dtype=torch.float32, device=device),
    )

    p2_prior = HalfNormal(
        scale=torch.tensor(param_2[0], dtype=torch.float32, device=device),
    )

    return p1_prior, p2_prior
