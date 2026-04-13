# Kestrel

Simulation-Based Inference (SBI) framework for inferring Type Ia supernova dust population parameters from photometric survey data. Kestrel uses importance-resampling from a pre-existing simulation bank to train a Neural Posterior Estimator (NPE) on the population-level parameters governing the SN Ia colour, stretch, and dust distributions.

## Method overview

Rather than running a forward simulator from scratch, Kestrel draws parameter vectors θ from a prior, computes per-supernova importance weights against a pre-simulated bank (using analytic likelihood functions for each population parameter), and resamples a synthetic observed population. These (θ, x) pairs train a normalising flow (NSF) with an attention-pooled embedding network, yielding a posterior over population parameters conditioned on the full observed dataset.

Model comparison between two configurations is supported via Neural Ratio Estimation (NRE): a binary classifier is trained to distinguish simulations from each model, and its output logit is converted to an approximate log Bayes factor on the Jeffreys scale.

## Repository structure

```
config_files/          KESTREL YAML configuration files
data_files/            Observed data (FITRES format)
sim_bank/              Pre-simulated SN Ia bank (FITRES format)
posteriors/            Output: simulations (.h5) and trained posteriors (.pt)
notebooks/             Jupyter notebooks for exploration and diagnostics
scripts/
  run_sims.py          Simulate training data and/or train the network
  run_nre.py           NRE model comparison (Bayes factors)
  launcher.job         SLURM job script for NERSC GPU nodes
  slurm_logs/          SLURM stdout/stderr (gitignored)
src/kestrel/           Installable Python package
  Functions.py         Dist* likelihood functions and prior builders
  dustbi_simulator.py  Core simulator, data loading, YAML parsing
  dustbi_nn.py         PopulationEmbeddingFull attention network
  dustbi_plotting.py   Diagnostic and calibration plots
pyproject.toml         Package definition and dependencies
```

## Installation

```bash
pip install -e .
```

## Running the pipeline

All scripts are run from the repo root and take a `--CONFIG` argument pointing to a YAML file in `config_files/`.

### 1. Generate training simulations

```bash
python scripts/run_sims.py --CONFIG config_files/KESTREL.yml --SIMULATE
```

Importance-resamples the simulation bank for each prior draw and streams `(theta, x)` pairs to the HDF5 file specified by `simname` in the config. Also writes a surviving-priors diagnostic PDF.

### 2. Train the network

```bash
python scripts/run_sims.py --CONFIG config_files/KESTREL.yml --TRAIN
```

Reads the HDF5 file in chunks and trains an NSF density estimator with `PopulationEmbeddingFull`. Saves the posterior as a pickle (`.pt`) after each chunk and writes a loss curve PDF.

### 3. Model comparison (NRE)

```bash
python scripts/run_nre.py --CONFIG config_files/KESTREL.yml
```

Compares the nominal model against each config listed under `Models_Comparison` in the YAML. Trains a binary classifier and reports log₁₀(BF) on the Jeffreys scale.

### NERSC SLURM

Edit `scripts/launcher.job` for your account/allocation, then:

```bash
sbatch scripts/launcher.job
```

Logs are written to `scripts/slurm_logs/kestrel_<jobid>.log`.

## Configuration

Each YAML file in `config_files/` controls the full model specification:

| Key | Description |
|-----|-------------|
| `param_names` | Space-separated list of population parameters to infer |
| `parameters_to_condition_on` | Observed SN Ia features passed to the network |
| `Functions` | Maps each parameter to a `Dist*` likelihood function |
| `Splits` | Host-mass splits: `Stepwise` (above/below threshold) or `Linear` (slope) |
| `Correlations` | Host property used as the correlation variable for `_EVOL` distributions |
| `Priors` | `BoxUniform` bounds for each parameter (tuples parsed at runtime) |
| `sim_parameters` | `n_sim`, `n_batch`, output paths for simulations and posterior |
| `Models_Comparison` | (NRE only) Configs to compare against the nominal model |
| `Population_B` | (Optional) Second-population priors + `mixing_prior` for a two-component mixture |
| `Latex_Names` | LaTeX labels for diagnostic plots |

### Available distribution functions

| Function | Parameters | Notes |
|----------|-----------|-------|
| `DistGaussian` | μ, σ | Standard Gaussian likelihood |
| `DistGaussian_EVOL` | μ₀, σ, slope | μ evolves linearly with a host property |
| `DistExponential` | τ | Importance-weighted; proposal fixed at τ=0.5 |
| `DistDoubleGaussian` | μ₁, σ₁, μ₂, σ₂, a | Two-component Gaussian |
| `DistTruncatedGaussian` | μ, σ | Gaussian truncated at low=1.2 by default |
| `DistLogistic` | L, k, σ | Logistic mean as a function of host mass |

Special parameters `STEP` (host-mass step in MURES) and `SCATTER` (grey intrinsic scatter) must appear last in `param_names`, in that order.

### Parameter ordering constraint

Parameters must be ordered by distribution type: Gaussian/Delta → Gaussian_EVOL → Exponential → DoubleGaussian → Logistic. `validate_order` enforces this at runtime.

## Data format

Input files are gzip-compressed SNANA FITRES files. The simulation bank requires columns including `SIM_AV`, `SIM_RV`, `SIM_x1`, `SIM_c`, `SIM_mB`, `HOST_LOGMASS`, and `zHD`. The observed data additionally requires `IDSURVEY`, `PROB_SNNV19`, and SALT2 fit columns (`x0`, `x0ERR`, `mB`, `mBERR`, `x1`, `x1ERR`, `c`, `cERR`).

## Dependencies

Managed via `pyproject.toml`. Key packages: `torch>=2.5.1`, `sbi>=0.25.0`, `astropy>=7.2.0`, `numpy`, `scipy`, `matplotlib`, `pandas`, `h5py`, `tqdm`.
