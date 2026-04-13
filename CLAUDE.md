# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kestrel is a Simulation-Based Inference (SBI) framework for inferring Type Ia supernova dust population parameters from photometric survey data (e.g., DES 5-year). It uses importance-sampled simulations from a pre-existing simulation bank and trains a Neural Posterior Estimator (NPE/SNPE) via the `sbi` library with a custom attention-based population embedding network.

## Package layout

```
src/kestrel/          # installable Python package
  __init__.py
  Functions.py        # Dist* likelihood functions
  dustbi_simulator.py # simulator, priors, data loading
  dustbi_nn.py        # PopulationEmbeddingFull network
  dustbi_plotting.py  # loss / diagnostic plots
scripts/
  run_sims.py         # simulate + train entry point
  run_nre.py          # NRE model-comparison entry point
  launcher.job        # SLURM job script for NERSC
pyproject.toml        # pip-installable package definition
```

## Running the pipeline

Install the package first (editable install, once per environment):
```bash
conda activate SBI
pip install -e .
```

**Simulate training data:**
```bash
python scripts/run_sims.py --CONFIG KESTREL.yml --SIMULATE
```

**Train the network** (requires simulations to exist first):
```bash
python scripts/run_sims.py --CONFIG KESTREL.yml --TRAIN
```

**Both in sequence** (as in `scripts/launcher.job` for SLURM on NERSC):
```bash
python scripts/run_sims.py --CONFIG KESTREL.yml --SIMULATE
python scripts/run_sims.py --CONFIG KESTREL.yml --TRAIN
```

**Neural Ratio Estimation / model comparison** (Bayes factors between configs listed under `Models_Comparison` in the YAML):
```bash
python scripts/run_nre.py --CONFIG KESTREL.yml
```

Submit to NERSC GPU nodes via:
```bash
sbatch scripts/launcher.job
```

## Architecture

### Data flow
1. **Inputs**: Two FITRES files (gzipped) — a large simulation bank (`SIM_BANK_SB.FITRES.gz`) and real data (`DES5YR.FITRES.gz`)
2. **Simulate** (`--SIMULATE`): Importance-samples from the simulation bank using per-parameter likelihood functions and saves `(theta, x)` pairs to an HDF5 file (path set by `simname` in config)
3. **Train** (`--TRAIN`): Reads HDF5 in chunks, appends to `SNPE` inference object, trains an NSF density estimator with `PopulationEmbeddingFull`, saves posterior as `.pt` via pickle

### Key modules

| File | Role |
|------|------|
| `scripts/run_sims.py` | Main entry point: orchestrates simulate + train pipeline |
| `scripts/run_nre.py` | NRE model comparison: trains binary classifier to estimate log Bayes factors |
| `src/kestrel/dustbi_simulator.py` | Core simulator: `build_layout`, `make_batched_simulator`, prior/data loading helpers |
| `src/kestrel/Functions.py` | Torch-compatible likelihood functions (all prefixed `Dist`), prior construction, YAML loading (`load_kestrel`) |
| `src/kestrel/dustbi_nn.py` | `PopulationEmbeddingFull`: attention-pooling over the SN population for the SBI embedding net |
| `src/kestrel/dustbi_plotting.py` | Loss curves and surviving-prior plots |

### Configuration (YAML)

Config files (e.g., `KESTREL.yml`) control everything:

- `param_names`: Parameters to infer (e.g., `SIM_c`, `SIM_RV`, `SIM_EBV`, `SIM_beta`, `SIM_x1`, `STEP`, `SCATTER`)
- `parameters_to_condition_on`: Observed SN Ia features passed to the network
- `Functions`: Maps each param to a `Dist*` function in `Functions.py` (e.g., `DistGaussian`, `DistExponential`, `DistGaussian_EVOL`)
- `Splits`: Host-mass splits — `Stepwise` creates two independent values above/below a threshold; `Linear` adds a continuous trend
- `Correlations`: Which host property correlates with a parameter (used by `_EVOL` distributions)
- `Priors`: `BoxUniform` bounds for each parameter (tuples are parsed at runtime)
- `sim_parameters`: `n_sim`, `n_batch`, output file paths
- `Population_B` (optional): Second-population priors + `mixing_prior` for a two-component mixture model
- `Models_Comparison` (optional, NRE only): List of other YAML configs to compare against

### Distribution / prior system

- Distributions in `Functions.py` must be named `Dist*` and return importance weights (not log-likelihoods)
- `DistExponential` uses a hard-coded proposal of `tau_proposal=0.5` due to selection effects
- `_EVOL` variants add a linear trend with a host property (passed via `Correlations`)
- `STEP` and `SCATTER` are special parameters handled separately from the `Dist*` system

### Mixture mode

When `Population_B` is present in the config, `run_sims.py` builds two sets of priors (A and B) plus a scalar mixing fraction prior, concatenated via `MultipleIndependent`. The simulator then samples from a mixture of both populations.
