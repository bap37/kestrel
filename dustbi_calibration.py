import numpy as np
import torch
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import warnings
import sys

def preprocess_data(parameters_to_condition_on, dfdata, ):
    
    output_distribution = preprocess_input_distribution(dfdata, parameters_to_condition_on)

    x = torch.stack(
        [output_distribution[p] for p in parameters_to_condition_on],
        dim=-1
    )
    
    return x

def preprocess_input_distribution(df, cols):
    return {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32)
        for col in cols
    }

def add_distance(df_tensor):
    
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    
    beta = 3.1 ; alpha = 0.16 ; M0 = -19.3
    
    correction = alpha * x1_obs - beta * c_obs + M0 + mB_obs
        
    MURES = df_tensor['MU'] - correction
    
    return  MURES

# -----------------------------
# SNANA Calibration functions
# -----------------------------

def load_assignments(filename):

    GENPDF_LIST = []
    SIM_LIST = []
    
    with open(filename, 'r') as fh:
        for line in fh:
            line_list = line.split("Opened")
            #print(line_list)

            SIM = line_list[0]
            SIM = SIM.split("SNIaMODEL00-")[-1] ; SIM = SIM.split("_")[0]

            SIM = f"CALIB_DATA/output/PIP_BP-DUST-SAMPLES_DATADESSIM_IA-{SIM}/FITOPT000.FITRES.gz"
            SIM_LIST.append(SIM)
            
            GENPDF = line_list[1].split("/GENPDFS/")[1]
            GENPDF = GENPDF.split(".DAT")[0] ; 
            GENPDF = f"CALIB_GENPDF/{GENPDF}.DAT"
            GENPDF_LIST.append(GENPDF)

            
    
    return GENPDF_LIST, SIM_LIST



def load_truth(genpdf_file):
    with open(genpdf_file, 'r') as fh:
        for line in fh:
            if line.startswith("#INP:"):
                true_params = line.split(" ")[1:]
                true_params = [float(i) for i in true_params]
                true_params[7], true_params[8] = true_params[8], true_params[7]
                true_params += [0,0]


    return true_params

def load_just_data(datfilename, desired_evt, parameters_to_condition_on):

    from astropy.cosmology import Planck18
    import numpy as np
    import pandas as pd
   
    dfdata = pd.read_csv(datfilename, 
                             comment="#", sep=r'\s+')
    dfdata['MU'] = Planck18.distmod(dfdata.zHD.values).value

    dfdata = dfdata.loc[dfdata.HOST_LOGMASS > 0]

    dfdata = dfdata.sample(n=desired_evt)

    output_distribution = preprocess_input_distribution(
    dfdata, parameters_to_condition_on[:-1]+['x0', 'x0ERR', 'MU'])

    MURES_sims = add_distance(output_distribution)
    
    dfdata['MURES'] = MURES_sims

    return dfdata

def sample_worker(posterior, x, Nsamples, out_queue):
    """Worker for subprocess-based sampling."""
    try:
        posterior = posterior.set_default_x(x)
        samples = posterior.sample((Nsamples,))
        out_queue.put(samples)
    except Exception as e:
        out_queue.put(e)


def sample_with_timeout(posterior, x, Nsamples, timeout=60):
    """Sample posterior with timeout using subprocess."""
    out_queue = Queue()
    p = Process(target=sample_worker, args=(posterior, x, Nsamples, out_queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # timeout

    if not out_queue.empty():
        result = out_queue.get()
        if isinstance(result, Exception):
            return None
        return result
    return None

# -----------------------------
# SBC loop
# -----------------------------

def run_sbc(SIMS, GENPDFS, posterior, Nsamples=5000, timeout=60, parameters_to_condition_on=None):
    num_sims = len(SIMS)
    num_params = 13
    ranks = np.full((num_sims, num_params), np.nan)  # preallocate

    # Convert the 0% acceptance warning into an exception
    warnings.filterwarnings(
        "error",
        message=r"Only 0\.000% proposal samples are accepted.*"
    )

    for i in range(num_sims):
        simfile = SIMS[i]
        truths = load_truth(GENPDFS[i])

        dfdata = load_just_data(simfile, 2201, parameters_to_condition_on)
        x = preprocess_data(parameters_to_condition_on, dfdata)

        try:
            samples = sample_with_timeout(posterior, x, Nsamples, timeout)
            if samples is None:
                print(f"Iteration {i}: timeout or sampling failed → leaving NaNs")
                continue
        except Warning:
            print(f"Iteration {i}: 0% proposal acceptance → skipping")
            continue

        # Vectorized rank computation
        samples_np = samples.detach().cpu().numpy()
        truths_np = np.array(truths)
        n = min(samples_np.shape[1], len(truths_np))

        valid_mask = ~np.isnan(samples_np[:, :n])

        # convert to float so we can assign NaN
        rank_values = (samples_np[:, :n] < truths_np[:n]).sum(axis=0).astype(float)

        # handle all-NaN columns
        all_nan_cols = ~valid_mask.any(axis=0)
        rank_values[all_nan_cols] = np.nan

        ranks[i, :n] = rank_values

        print(f"Iteration {i}: done")

    return ranks


