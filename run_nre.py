from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
import argparse
import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ModelComparisonNet(nn.Module):
    """Binary classifier for model comparison via the classification approach
    to Bayes factors (Cranmer et al. 2015).

    Embeds a population of SNe via attention pooling, then classifies
    which model generated the population.
    """
    def __init__(self, input_dim, embed_hidden=64, embed_output=32):
        super().__init__()
        self.embedding = PopulationEmbeddingFull(
            input_dim=input_dim, hidden_dim=embed_hidden, output_dim=embed_output
        )
        self.head = nn.Sequential(
            nn.Linear(embed_output, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.embedding(x)   # (batch, embed_output)
        return self.head(h)      # (batch, 1)  — raw logit


def standardise_features(x_train, x_obs, eps=1e-6):
    """Z-score x_train per feature; apply same transform to x_obs.

    Stats are computed across (batch, n_sne) for each of the n_feat columns
    using the training data only.
    """
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp(min=eps)
    return (x_train - mean) / std, (x_obs - mean) / std, mean, std


def train_classifier(net, x_train, y_train, x_val, y_val,
                     device="cpu", lr=1e-3, batch_size=64,
                     max_epochs=200, patience=15):
    """Train the binary classifier with early stopping on validation loss.

    Returns
    -------
    net : nn.Module
        Trained network loaded with the best-val-loss state.
    best_val_loss : float
    best_val_acc : float
        Val accuracy at the best-val-loss epoch (i.e. matches the returned net).
    train_curve : list[float]
    val_curve : list[float]
    """
    net = net.to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_state = None
    train_curve = []
    val_curve = []

    n_train = x_train.shape[0]

    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)

            logits = net(xb).squeeze(-1)
            loss = loss_fn(logits, yb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n_train

        net.eval()
        with torch.no_grad():
            val_logits_list = []
            for vs in range(0, x_val.shape[0], batch_size):
                vb = x_val[vs : vs + batch_size].to(device)
                val_logits_list.append(net(vb).squeeze(-1))
            val_logits = torch.cat(val_logits_list)
            y_val_dev = y_val.to(device)
            val_loss = loss_fn(val_logits, y_val_dev).item()
            val_acc = ((val_logits > 0).float() == y_val_dev).float().mean().item()

        train_curve.append(epoch_loss)
        val_curve.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 20 == 0 or epochs_without_improvement == 0:
            print(f"  Epoch {epoch+1:3d}  train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    net.load_state_dict(best_state)
    net.eval()
    return net, best_val_loss, best_val_acc, train_curve, val_curve


def fit_temperature(net, x_val, y_val, device, batch_size=64):
    """Post-hoc temperature scaling (Guo et al. 2017).

    Fit a single scalar T > 0 on the val set by minimising BCE on logits/T.
    Returns the fitted T.
    """
    net.eval()
    with torch.no_grad():
        logits_parts = []
        for i in range(0, x_val.shape[0], batch_size):
            logits_parts.append(net(x_val[i:i+batch_size].to(device)).squeeze(-1))
        logits = torch.cat(logits_parts)
    y = y_val.to(device)
    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / log_T.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def jeffreys_strength(abs_log10):
    if abs_log10 < 0.5:
        return "Not worth more than a bare mention"
    if abs_log10 < 1.0:
        return "Substantial"
    if abs_log10 < 1.5:
        return "Strong"
    if abs_log10 < 2.0:
        return "Very strong"
    return "Decisive"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CONFIG", help="Configuration yaml for NRE model comparison.", type=str)
    parser.add_argument("--SEED", help="Master random seed for reproducibility.", type=int, default=0)
    parser.add_argument("--N_ENSEMBLE", help="Number of classifier seeds to average over (>=2 enables error bars).",
                        type=int, default=1)
    parser.add_argument("--BIRD", help="Prints a nice bird :)", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    if args.BIRD:
        print("I'm very sorry but the kestrel hasn't taken flight yet!")
        quit()

    if not args.CONFIG:
        print("No configuration file provided via --CONFIG. Quitting.")
        quit()

    torch.manual_seed(args.SEED)
    np.random.seed(args.SEED)

    infos = load_kestrel(args.CONFIG)

    datfilename = infos['Data_File'][0]
    simfilename = infos['Simbank_File'][0]
    parameters_to_condition_on = infos['parameters_to_condition_on']
    ndim = len(parameters_to_condition_on)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, dfdata = load_data(simfilename, datfilename)

    num_simulations = infos['sim_parameters']['n_sim']

    output_distribution = preprocess_input_distribution(
        df, parameters_to_condition_on[:-1] + ['x0', 'x0ERR', 'MU'])
    df['MURES'] = add_distance(output_distribution)

    output_distribution = preprocess_input_distribution(
        dfdata, parameters_to_condition_on[:-1] + ['x0', 'x0ERR', 'MU'])
    dfdata['MURES'] = add_distance(output_distribution)

    x_obs = preprocess_data(parameters_to_condition_on, dfdata).unsqueeze(0)

    # --- Nominal model (model 1) ---
    dicts_1 = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations']]
    param_names_1 = infos['param_names']
    params_to_fit_1 = parameter_generation(param_names_1, dicts_1)
    layout_1 = build_layout(params_to_fit_1, dicts_1)

    mixture_1 = 'Population_B' in infos
    split_positions_1 = None
    if mixture_1:
        pop_b = infos['Population_B']
        shared_params_1 = [p for p in pop_b.get('shared_params', []) if p not in ('STEP', 'SCATTER')]
        split_names_1 = [n for n in param_names_1 if n not in shared_params_1 and n not in ('STEP', 'SCATTER')]
        dicts_1B = [infos['Functions'], infos['Splits'], pop_b['Priors'], infos['Correlations']]
        priors_A = build_distribution_priors(param_names_1, dicts_1)
        priors_B_split = build_distribution_priors(split_names_1, dicts_1B)
        mix = pop_b['mixing_prior']
        f_prior = BoxUniform(low=torch.tensor([mix[0]]), high=torch.tensor([mix[1]]))
        special = build_special_priors(param_names_1, dicts_1)
        priors_1 = MultipleIndependent(priors_A + priors_B_split + [f_prior] + special)
        split_positions_1 = compute_split_positions(layout_1, shared_params_1)
        assert len(split_positions_1) == len(priors_B_split)
    else:
        priors_1 = prior_generator(param_names_1, dicts_1)

    nominal_sim = make_batched_simulator(
        layout_1, df, param_names_1, parameters_to_condition_on,
        dicts_1, dfdata, sub_batch=500, device=device, mixture=mixture_1,
        split_positions=split_positions_1
    )

    print(f"Simulating {num_simulations} from nominal model ({args.CONFIG})...")
    theta_1 = priors_1.sample((num_simulations,)).to(device)
    x1 = nominal_sim(theta_1).cpu()

    mask1 = torch.isfinite(x1).all(dim=(1, 2))
    x1_clean = x1[mask1]
    print(f"  {args.CONFIG}: {x1_clean.shape[0]} valid / {x1.shape[0]} total")

    # --- Compare against each model ---
    for model_path in infos['Models_Comparison']:
        print(f"\n{'='*60}")
        print(f"Comparing {args.CONFIG} vs {model_path}")
        print(f"{'='*60}")

        comp_infos = load_kestrel(model_path)
        dicts_2 = [comp_infos['Functions'], comp_infos['Splits'],
                    comp_infos['Priors'], comp_infos['Correlations']]
        param_names_2 = comp_infos['param_names']
        params_to_fit_2 = parameter_generation(param_names_2, dicts_2)
        layout_2 = build_layout(params_to_fit_2, dicts_2)

        mixture_2 = 'Population_B' in comp_infos
        split_positions_2 = None
        if mixture_2:
            pop_b2 = comp_infos['Population_B']
            shared_params_2 = [p for p in pop_b2.get('shared_params', []) if p not in ('STEP', 'SCATTER')]
            split_names_2 = [n for n in param_names_2 if n not in shared_params_2 and n not in ('STEP', 'SCATTER')]
            dicts_2B = [comp_infos['Functions'], comp_infos['Splits'], pop_b2['Priors'], comp_infos['Correlations']]
            priors_2A = build_distribution_priors(param_names_2, dicts_2)
            priors_2B_split = build_distribution_priors(split_names_2, dicts_2B)
            mix2 = pop_b2['mixing_prior']
            f_prior_2 = BoxUniform(low=torch.tensor([mix2[0]]), high=torch.tensor([mix2[1]]))
            special_2 = build_special_priors(param_names_2, dicts_2)
            priors_2 = MultipleIndependent(priors_2A + priors_2B_split + [f_prior_2] + special_2)
            split_positions_2 = compute_split_positions(layout_2, shared_params_2)
            assert len(split_positions_2) == len(priors_2B_split)
        else:
            priors_2 = prior_generator(param_names_2, dicts_2)

        comp_sim = make_batched_simulator(
            layout_2, df, param_names_2, parameters_to_condition_on,
            dicts_2, dfdata, sub_batch=500, device=device, mixture=mixture_2,
            split_positions=split_positions_2
        )

        print(f"Simulating {num_simulations} from comparison model ({model_path})...")
        theta_2 = priors_2.sample((num_simulations,)).to(device)
        x2 = comp_sim(theta_2).cpu()

        mask2 = torch.isfinite(x2).all(dim=(1, 2))
        x2_clean = x2[mask2]
        print(f"  {model_path}: {x2_clean.shape[0]} valid / {x2.shape[0]} total")

        # Balance classes
        n_use = min(x1_clean.shape[0], x2_clean.shape[0])
        x_combined = torch.cat([x1_clean[:n_use], x2_clean[:n_use]], dim=0)
        y_combined = torch.cat([torch.zeros(n_use), torch.ones(n_use)])

        # Z-score features (computed on combined training data, applied to x_obs identically)
        x_combined, x_obs_norm, feat_mean, feat_std = standardise_features(x_combined, x_obs)
        x_obs_norm_dev = x_obs_norm.to(device)

        # Shuffle and split train/val
        perm = torch.randperm(2 * n_use)
        x_combined = x_combined[perm]
        y_combined = y_combined[perm]

        n_val = max(1, int(0.1 * 2 * n_use))
        n_train = 2 * n_use - n_val

        x_train, x_val = x_combined[:n_train], x_combined[n_train:]
        y_train, y_val = y_combined[:n_train], y_combined[n_train:]

        print(f"Training classifier: {n_train} train, {n_val} val samples (ensemble of {args.N_ENSEMBLE})")

        log10_bfs = []
        val_accs = []
        val_losses = []
        temperatures = []
        train_curves = []
        val_curves = []

        for k in range(args.N_ENSEMBLE):
            torch.manual_seed(args.SEED + k)
            np.random.seed(args.SEED + k)
            print(f"\n[Ensemble member {k+1}/{args.N_ENSEMBLE}, seed={args.SEED + k}]")
            net = ModelComparisonNet(input_dim=ndim)
            net, val_loss, val_acc, tc, vc = train_classifier(
                net, x_train, y_train, x_val, y_val, device=device
            )

            T = fit_temperature(net, x_val, y_val, device)
            print(f"  Calibration temperature T = {T:.3f}")

            net.eval()
            with torch.no_grad():
                logit = net(x_obs_norm_dev).squeeze() / T
            log10_bfs.append(-logit.item() / math.log(10))
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            temperatures.append(T)
            train_curves.append(tc)
            val_curves.append(vc)

        log10_bf_mean = float(np.mean(log10_bfs))
        favoured = args.CONFIG if log10_bf_mean > 0 else model_path

        print(f"\n{'-'*60}")
        if args.N_ENSEMBLE > 1:
            log10_bf_std = float(np.std(log10_bfs))
            print(f"log10(BF) {args.CONFIG} vs {model_path}: {log10_bf_mean:.4f} +/- {log10_bf_std:.4f}  (over {args.N_ENSEMBLE} seeds)")
        else:
            print(f"log10(BF) {args.CONFIG} vs {model_path}: {log10_bf_mean:.4f}")
        print(f"Mean Bayes Factor: {10**log10_bf_mean:.4f}")
        print(f"Mean val accuracy: {np.mean(val_accs):.3f}, mean T: {np.mean(temperatures):.3f}")
        print(f"  -> {jeffreys_strength(abs(log10_bf_mean))} evidence favouring {favoured}")

        # Diagnostic plot: per-ensemble train/val loss curves.
        # Each ensemble member gets a distinct colour; train solid, val dashed.
        try:
            os.makedirs("posteriors", exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            K = len(train_curves)
            cmap = plt.get_cmap('viridis')
            for k, (tc, vc) in enumerate(zip(train_curves, val_curves)):
                colour = cmap(k / max(K - 1, 1))
                ax.plot(tc, color=colour, alpha=0.8, lw=1, linestyle='-',
                        label=f'k={k} train' if K > 1 else 'train')
                ax.plot(vc, color=colour, alpha=0.8, lw=1, linestyle='--',
                        label=f'k={k} val' if K > 1 else 'val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('BCE loss')
            ax.set_yscale('log')
            ax.legend(loc='upper right', fontsize=8, ncol=max(1, K // 4))
            ax.set_title(f"{args.CONFIG} vs {model_path}")
            out = f"posteriors/nre_loss_{os.path.basename(args.CONFIG).replace('.yml','')}_vs_{os.path.basename(model_path).replace('.yml','')}.pdf"
            fig.savefig(out, bbox_inches='tight')
            plt.close(fig)
            print(f"  Loss curve plot saved to {out}")
        except Exception as e:
            print(f"  (loss plot skipped: {e})")
