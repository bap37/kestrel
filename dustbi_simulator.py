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

    function_dict, split_dict, priors_dict, corr_dict = dicts

    idx = {
        "gauss": [],
        "delta": [],
        "gauss_EVOL": [],
        "exp": [],
        "exp_EVOL": [],
        "lognormal": [],
        "double_gaussian": [],
        "linear": [],
        "stepwise": [],
        "logistic": [],
    }

    order = {
        "gauss": [],
        "delta": [],
        "gauss_EVOL": [],
        "exp": [],
        "exp_EVOL": [],
        "lognormal": [],
        "double_gaussian": [],
        "linear": [],
        "stepwise": [],
        "logistic": [],
    }

    params_to_avoid = ['STEP', 'SCATTER']

    for i, name in enumerate(param_names):

        base = name.split("_HIGH_")[0]

        if base in params_to_avoid:
            continue

        if "Truncated" in base:
            base = base.replace("Truncated", "")
        
        try:
            funcname = function_dict[base].__name__
        except KeyError:
            AssertionError(f"I didn't understand how to parse {base}; please ensure it's parsed correctly")

        if ("Gaussian" in funcname) and ("EVOL" not in funcname):
            idx["gauss"].append(i)
            order["gauss"].append(name)

        elif funcname == "DistDelta":
            idx["delta"].append(i)
            order["delta"].append(name)

        elif funcname == "DistGaussian_EVOL":
            idx["gauss_EVOL"].append(i)
            order["gauss_EVOL"].append(name)

        elif funcname == "DistExponential":
            idx["exp"].append(i)
            order["exp"].append(name)

        elif funcname == "DistExponential_EVOL":
            idx["exp_EVOL"].append(i)
            order["exp_EVOL"].append(name)

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

        elif "Logistic" in funcname:
            idx["logistic"].append(i)
            order["logistic"].append(name)


        else:
            AssertionError(f"You have passed {funcname}, which I don't recognise")

    counts = {k: len(v) for k, v in idx.items()}

    n_params = {
        "gauss": 2,
        "gauss_EVOL":3,
        "delta":1,
        "exp": 1,
        "exp_EVOL": 2,
        "lognormal": 2,
        "double_gaussian": 5,
        "linear": 2,
        "stepwise": 1,
        "logistic": 3,
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



def make_batched_simulator(layout, df, param_names, parameters_to_condition_on,
                           dicts, dfdata, sub_batch=10, device="cpu", debug=False,
                           mixture=False):
    function_dict, split_dict, priors_dict, corr_dict = dicts
    validate_order(param_names, dicts)

    params_to_avoid = ['STEP', 'SCATTER', 'EVOL']
    
    splits = list({v[1] for v in split_dict.values()})
    
    # Pre-compute ALL tensor columns ONCE
    all_cols = list(set(list(priors_dict.keys()) + splits + parameters_to_condition_on))
    
    # Remove anything that contains any of the substrings
    all_cols = [
        col for col in all_cols
        if not any(substr in col for substr in params_to_avoid)
    ]

    #print(all_cols)
    
    df_tensor = {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32, device=device)
        for col in all_cols
    }
    
    # Pre-stack output columns for fast batched indexing
    output_stack = torch.stack(
        [df_tensor[col] for col in parameters_to_condition_on], dim=-1
    )  # (N, n_features)

    if "STEP" in param_names:
    #Calculate indices for things that need the "mass" step added to them. 
        steps_to_add = ['MURES']
        step_indices = torch.tensor(
            [parameters_to_condition_on.index(c) for c in steps_to_add],
            dtype=torch.long,
            device=device
        )

        y_idx = parameters_to_condition_on.index(split_dict["STEP"][1])
        step_threshold = split_dict["STEP"][2]

    if "SCATTER" in param_names:
        steps_to_add = ['MURES', 'mB']
        scatter_indices = torch.tensor(
            [parameters_to_condition_on.index(c) for c in steps_to_add],
            dtype=torch.long,
            device=device
        )


    N = len(df)
    n_target = len(dfdata)

    # Mixture mode: compute how many params per population
    if mixture:
        n_pop_params = sum(layout.counts[d] * layout.n_params[d] for d in layout.counts)
        f_idx = 2 * n_pop_params

    def _compute_joint_weights(theta_pop, B, dev, extra_tensors=None):
        """Compute importance weights for a single population's parameters."""
        joint_weights = torch.ones(B, N, device=dev)
        high_flag = False

        for dist in layout.slices:
            if layout.counts[dist] == 0:
                continue
            if dist == "delta":
                continue

            theta_dist = theta_pop[:, layout.slices[dist]]
            n_param = layout.n_params[dist]
            theta_dist = theta_dist.view(B, layout.counts[dist], n_param)

            for i in range(layout.counts[dist]):
                name = layout.order[dist][i]

                if "_EVOL" in name:
                    name = name.split("_EVOL_")[0]

                if "_HIGH_" in name:
                    name = name.split("_HIGH_")[0]
                    high_flag = True

                theta_i = theta_dist[:, i, :]
                if extra_tensors and name in extra_tensors:
                    x = extra_tensors[name]  # (B, N)
                else:
                    x = df_tensor[name]      # (N,)
                if x.ndim == 1:
                    x = x.unsqueeze(0)  # (1, N)
                if debug: print(name, theta_dist.shape, x.ndim)
                batch_size = B
                correlation = df_tensor.get(corr_dict.get(name)) if corr_dict.get(name) else None

                try:
                    steptype, split_param, split_val = split_dict[name]
                except KeyError:
                    steptype, split_param, split_val = None, None, None

                if (name in split_dict) & (steptype == "Stepwise"):
                    split_tensor = df_tensor[split_param]

                    if high_flag:
                        mask = split_tensor >= split_val
                        high_flag = False
                    else:
                        mask = split_tensor < split_val

                    weights = torch.ones(batch_size, N, device=dev)

                    x_sub = x[:,mask]
                    density_sub = function_dict[name](x_sub, theta_i, correlation)
                    density_sub = torch.clamp(density_sub, min=0.0)

                    if density_sub.ndim == 1:
                        density_sub = density_sub.unsqueeze(0)
                    weights[:, mask] = density_sub

                else:
                    density = function_dict[name](x, theta_i, correlation)
                    weights = torch.clamp(density, min=0.0)

                if weights.ndim == 1:
                    weights = weights.unsqueeze(0)

                joint_weights *= weights

        return joint_weights

    def _simulate_sub_batch(theta):
        """Fully batched: theta is (B, n_params), returns (B, n_target, n_features)."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        B = theta.shape[0]
        dev = theta.device

        mu = None
        # Identify delta parameters
        if layout.counts["delta"] > 0:
            theta_delta = theta[:, layout.slices["delta"]]
            theta_delta = theta_delta.view(B, layout.counts["delta"], layout.n_params["delta"])
            for i in range(layout.counts["delta"]):
                name = layout.order["delta"][i]
                theta_i = theta_delta[:, i, :]  # (B, 1)

                if name == "SIM_beta":  # or whatever your key is
                    mu = add_beta_distance(df_tensor, theta_i)  # (B, N)


        # --- Build kwargs cleanly ---
        extra_tensors = {}
        if mu is not None:
            extra_tensors["MU"] = mu
        
                
        if mixture:
            theta_A = theta[:, :n_pop_params]
            theta_B = theta[:, n_pop_params:2*n_pop_params]
            f = theta[:, f_idx].unsqueeze(1)  # (B, 1)
            joint_weights = f * _compute_joint_weights(theta_A, B, dev, extra_tensors=extra_tensors) \
                      + (1 - f) * _compute_joint_weights(theta_B, B, dev, extra_tensors=extra_tensors)
        else:
            joint_weights = _compute_joint_weights(theta, B, dev, extra_tensors=extra_tensors)

        # --- Normalise ---
        weight_sum = joint_weights.sum(dim=1, keepdim=True)  # (B, 1)
        bad_mask = (weight_sum.squeeze(1) == 0)

        joint_weights[bad_mask] = 1.0
        weight_sum = torch.where(weight_sum == 0, torch.tensor(float(N), device=dev), weight_sum)
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

        #then add whatever proposed step is necessary; I will need to come back to this to make it less hard-coded
        if "STEP" in param_names:
            if "SCATTER" in param_names:
                temp_index = -2
            else:
                temp_index = -1
            gamma = theta[:, temp_index].unsqueeze(1)       # (B,1)
            y = result[:, :, y_idx]                 # (B, n_target)
            step = torch.where(y < step_threshold, -gamma/2, gamma/2)  # (B, n_target)
            result[:, :, step_indices] += step.unsqueeze(-1)

        #Then if grey scatter is enabled, add it to this nonsense.            
        if "SCATTER" in param_names:
            temp_index = -1
            scatter = theta[:, temp_index].view(-1, 1, 1)
            scatter = torch.clamp(scatter, min=1e-6)
            noise = torch.randn_like(result[:, :, scatter_indices]) * scatter
            result[:, :, scatter_indices] += noise

        # --- Fill bad simulations with NaN ---
        result[bad_mask] = float('nan')

        if debug:
            dft = df.iloc[resampled_idx.squeeze(0)]
            if layout.counts["delta"] > 0:
                theta_delta = theta[:, layout.slices["delta"]]
                theta_delta = theta_delta.view(B, layout.counts["delta"], layout.n_params["delta"])
                theta_delta = theta_delta[:, 0, :]  # (B, 1)
                dft['SIM_beta'] = theta_delta.item()  # extract scalar
            return dft

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



#####################
### BEGIN SUPPORT FUNCTIONS
###########################

def preprocess_input_distribution(df, cols):
    return {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32)
        for col in cols
    }


def parameter_generation(list_of_parameter_names, dicts):
    
    empty_list = []
    function_dict, split_dict, priors_dict, corr_dict = dicts
    
    for name in list_of_parameter_names:
        if name in split_dict.keys():
            evol_type = (split_dict[name][0])
            if evol_type == 'Stepwise':
                if name == "STEP":  #hacky
                    pass
                else:
                    split = (split_dict[name][1])
                    #print(name,split)
                    empty_list.append(name+"_HIGH_"+split)
        empty_list.append(name)
    return empty_list


def validate_step(param_names, params_to_avoid):
    # Extract only the special params that are present
    present = [p for p in params_to_avoid if p in param_names]

    if not present:
        return  # nothing to check

    # Get their positions in param_names
    indices = [param_names.index(p) for p in present]

    # --- Check they are at the end ---
    expected_indices = list(range(len(param_names) - len(present), len(param_names)))
    if indices != expected_indices:
        raise ValueError(
            f"{present} must appear at the end of param_names in order {params_to_avoid}"
        )

    # --- Check order is correct ---
    if present != sorted(present, key=params_to_avoid.index):
        raise ValueError(
            f"{present} are not in the correct order {params_to_avoid}"
        )

def validate_order(param_names, dicts): 
    """
    Catchall error trapping function to be called during initialisation. 
    """
    function_dict, split_dict, priors_dict, corr_dict = dicts    

    params_to_avoid = ["STEP", "SCATTER"]

    validate_step(param_names, params_to_avoid)

    #Will need to add other functions in as they are implemented; make sure that Exponential is always last 
    order_priority = {
        DistGaussian: 2,
        DistTruncatedGaussian: 2,
        DistDelta:2,
        DistGaussian_EVOL:3,
        DistExponential: 4,
        DistExponential_EVOL: 5,
        DistDoubleGaussian: 6,
        DistLogistic: 7,
    }

    #Temporarily strip "step" from param names, since it's implemented differently.
    new_list = [x for x in param_names if x not in params_to_avoid] 
    priorities = [order_priority[function_dict[p]] for p in new_list]

    #Make sure that all entries are in the correct order
    if priorities != sorted(priorities):
        raise ValueError(f"Please ensure that the parameters are give in the following order: {order_priority.values}")

    #check to make sure that we don't have a split and Correlation enabled at the same time 
    conflicts = [k for k in split_dict if k in corr_dict and corr_dict[k] != 'None']
    for k in conflicts:
        if "EVOL" not in function_dict[k].__name__:
            raise ValueError(f"Conflict detected for keys: {conflicts}")
        else: 
            print(f"Found that there is a split and correlation entry for {conflicts}. Continuing.")

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

        if "Truncated" in name:
            name = name.replace("Truncated","")

        try:
            func_name = (function_dict[name].__name__)
            func_params = latex_dict[func_name] ; pname = latex_dict[name]
        except KeyError: #HackY!
            if name == "STEP":
                pname = "STEP"
                func_params = [r"$\gamma$"]
            elif name =="SCATTER":
                pname = "SCATTER"
                func_params = [r"$\sigma_{\rm int}$"]
            else:
                print(f"No idea what you passed me: {name}, {func_name}")
                
        
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


def preprocess_data(parameters_to_condition_on, dfdata, ):
    
    output_distribution = preprocess_input_distribution(dfdata, parameters_to_condition_on)

    x = torch.stack(
        [output_distribution[p] for p in parameters_to_condition_on],
        dim=-1
    )
    
    return x


def load_kestrel(filename):
    
    import ast
    import yaml
    
    FUNCTION_REGISTRY = {
        "DistGaussian": DistGaussian,
        "DistExponential": DistExponential,
        "DistLogistic": DistLogistic,
        "DistDoubleGaussian": DistDoubleGaussian,
        "DistGaussian_EVOL": DistGaussian_EVOL,
        "DistExponential_EVOL": DistExponential_EVOL,
        "DistTruncatedGaussian": DistTruncatedGaussian,
        "DistDelta": DistDelta,
    }


    with open(filename, 'r') as file:
        raw_yaml = yaml.safe_load(file)

    raw_yaml['param_names'] = raw_yaml['param_names'].split(" ")
        
    raw_yaml['Data_File'] = raw_yaml['Data_File'].split(" ")
    raw_yaml['Simbank_File'] = raw_yaml['Simbank_File'].split(" ")



    _priors = raw_yaml['Priors']
    for entry in _priors:
        for n in range(len(_priors[entry])):
            raw_yaml['Priors'][entry][n] = ast.literal_eval(raw_yaml['Priors'][entry][n])
            
    raw_yaml['Functions'] = {
        name: FUNCTION_REGISTRY[func_name]
        for name, func_name in raw_yaml["Functions"].items()
        }

    try:
        raw_yaml['Models_Comparison'] = raw_yaml['Models_Comparison'].split(" ")
    except KeyError:
        pass

    if 'Population_B' in raw_yaml:
        pop_b = raw_yaml['Population_B']
        for entry in pop_b['Priors']:
            for n in range(len(pop_b['Priors'][entry])):
                pop_b['Priors'][entry][n] = ast.literal_eval(pop_b['Priors'][entry][n])
        pop_b['mixing_prior'] = ast.literal_eval(pop_b['mixing_prior'])

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

    print("Only loading DES Data")
    dfdata = dfdata.loc[dfdata.IDSURVEY == 10]
    #dfdata.loc[dfdata.PROB_SNNV19 < -1., "PROB_SNNV19"] = 1
    dfdata = dfdata.loc[dfdata.PROB_SNNV19 >= 0.5]

    print("Ensuring only valid log masses.")
    dfdata = dfdata.loc[dfdata.HOST_LOGMASS > 0]
    df = df.loc[df.HOST_LOGMASS > 0 ]

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
    theta_tot = []
    mask_tot = []
    
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

                mask = ~torch.isnan(x_batch).any(dim=tuple(range(1, x_batch.ndim)))
                theta_tot.append(theta_batch)
                mask_tot.append(mask)

                # resize datasets
                theta_ds.resize(cursor + current_bs, axis=0)
                x_ds.resize(cursor + current_bs, axis=0)

                # write batch
                theta_ds[cursor:cursor + current_bs] = theta_np
                x_ds[cursor:cursor + current_bs] = x_np

                cursor += current_bs
                pbar.update(current_bs)
    
    theta_tot = torch.cat(theta_tot, dim=0)
    mask_tot = torch.cat(mask_tot, dim=0)

    theta_valid = theta_tot[mask_tot]

    p_vals = priors.sample((n_sim,))

    return theta_valid, p_vals


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

def add_beta_distance(df_tensor, SN_beta, SN_alpha=0.15):
    """
    df_tensor: dict of tensors, each (N,)
    SN_beta:   (B, 1) or (B,)
    SN_alpha:  scalar
    """

    x1 = df_tensor['SIM_x1']  # (N,)
    c  = df_tensor['SIM_c']   # (N,)
    mB = df_tensor['SIM_mB']  # (N,)

    # Ensure shapes are broadcastable
    if SN_beta.ndim == 1:
        SN_beta = SN_beta.unsqueeze(1)  # (B, 1)

    x1 = x1.unsqueeze(0)  # (1, N)
    c  = c.unsqueeze(0)
    mB = mB.unsqueeze(0)

    alpha = SN_alpha
    beta  = SN_beta
    M0    = 19.3643

    MU = alpha * x1 - beta * c + M0 + mB  # (B, N)

    return MU

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

def build_distribution_priors(param_names, dicts, device='cpu'):
    """Build the list of BoxUniform priors for distribution parameters only (no STEP/SCATTER)."""
    function_dict, split_dict, priors_dict, corr_dict = dicts
    list_o_priors = []

    params_to_avoid = ['EVOL', 'STEP', 'SCATTER']

    for name in param_names:
        if name in params_to_avoid:
            continue

        if "_HIGH_" in name:
            name = name.split("_HIGH_")[0]

        func_name = function_dict[name].__name__

        if "Gaussian" in func_name:
            mu0, sigma0 = priors_dict[name]
            mu_prior, sigma_prior = TwoDBoxPrior(mu0, sigma0, device=device)
            list_o_priors.extend([mu_prior, sigma_prior])
            if name in split_dict:
                evol_type = (split_dict[name][0])
                if evol_type == "Stepwise":
                    list_o_priors.extend([mu_prior, sigma_prior])
                elif evol_type == "Linear":
                    slope0 = priors_dict[name+"_EVOL"][0]
                    slope_prior = BoxUniform(
                        low= torch.tensor([slope0[0]], dtype=torch.float32, device=device),
                        high=torch.tensor([slope0[1]], dtype=torch.float32, device=device)
                            )
                    list_o_priors.extend([slope_prior])

        elif "DistDelta" in func_name:
            prior0 = priors_dict[name][0]

            delta_prior = BoxUniform(
                low= torch.tensor([prior0[0]], dtype=torch.float32, device=device),
                high=torch.tensor([prior0[1]], dtype=torch.float32, device=device)
                    )

            list_o_priors.append(delta_prior)

        elif "DistExponential" in func_name:
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
                    slope0 = priors_dict[name+"_EVOL"]
                    slope0 = BoxUniform(
                        low= torch.tensor([slope0[0]], dtype=torch.float32, device=device),
                        high=torch.tensor([slope0[1]], dtype=torch.float32, device=device)
                            )
                    list_o_priors.extend([slope_prior])


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
                    slope0 = priors_dict[name+"_EVOL"]
                    slope0 = BoxUniform(
                        low= torch.tensor([slope0[0]], dtype=torch.float32, device=device),
                        high=torch.tensor([slope0[1]], dtype=torch.float32, device=device)
                            )
                    list_o_priors.extend([slope_prior])

        if func_name == "DistLogistic":
            L0, k0, sigma0 = priors_dict[name]
            L_prior = BoxUniform(
                low= torch.tensor([L0[0]], dtype=torch.float32, device=device),
                high=torch.tensor([L0[1]], dtype=torch.float32, device=device)
                )
            k_prior = BoxUniform(
                low= torch.tensor([k0[0]], dtype=torch.float32, device=device),
                high=torch.tensor([k0[1]], dtype=torch.float32, device=device)
                )
            sigma_prior = BoxUniform(
                low= torch.tensor([sigma0[0]], dtype=torch.float32, device=device),
                high=torch.tensor([sigma0[1]], dtype=torch.float32, device=device)
                )
            list_o_priors.extend([L_prior, k_prior, sigma_prior])

    return list_o_priors


def build_special_priors(param_names, dicts, device='cpu'):
    """Build the list of priors for STEP and SCATTER (appended last in theta)."""
    _, _, priors_dict, _ = dicts
    list_o_priors = []

    if "STEP" in param_names:
        step0 = priors_dict["STEP"][0]
        step_prior = BoxUniform(
            low= torch.tensor([step0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([step0[1]], dtype=torch.float32, device=device)
            )
        list_o_priors.extend([step_prior])

    if "SCATTER" in param_names:
        scatter0 = priors_dict["SCATTER"][0]
        scatter_prior = BoxUniform(
            low= torch.tensor([scatter0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([scatter0[1]], dtype=torch.float32, device=device)
            )
        list_o_priors.extend([scatter_prior])

    return list_o_priors


def prior_generator(param_names, dicts, device='cpu'):
    all_priors = build_distribution_priors(param_names, dicts, device=device) + \
                 build_special_priors(param_names, dicts, device=device)
    print(f"Added {len(all_priors)} priors")
    return MultipleIndependent(all_priors, device=device)

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
