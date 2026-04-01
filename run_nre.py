from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
import argparse
import torch
import torch.nn as nn


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


def train_classifier(net, x_train, y_train, x_val, y_val,
                     device="cpu", lr=1e-3, batch_size=64,
                     max_epochs=200, patience=15):
    """Train the binary classifier with early stopping on validation loss."""
    net = net.to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

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

        # Validation (batched to avoid OOM on large populations)
        net.eval()
        with torch.no_grad():
            val_logits_list = []
            for vs in range(0, x_val.shape[0], batch_size):
                vb = x_val[vs : vs + batch_size].to(device)
                val_logits_list.append(net(vb).squeeze(-1))
            val_logits = torch.cat(val_logits_list)
            val_loss = loss_fn(val_logits, y_val.to(device)).item()
            val_acc = ((val_logits > 0).float() == y_val.to(device)).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    return net, best_val_loss


def add_distance(df_tensor):
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    beta = 3.1 ; alpha = 0.16 ; M0 = -19.3
    correction = alpha * x1_obs - beta * c_obs + M0 + mB_obs
    MURES = df_tensor['MU'] - correction
    return MURES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CONFIG", help="Configuration yaml for NRE model comparison.", type=str)
    parser.add_argument("--BIRD", help="Prints a nice bird :)", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    if args.BIRD:
        print("I'm very sorry but the kestrel hasn't taken flight yet!")
        quit()

    if not args.CONFIG:
        print("No configuration file provided via --CONFIG. Quitting.")
        quit()

    infos = load_kestrel(args.CONFIG)

    datfilename = infos['Data_File'][0]
    simfilename = infos['Simbank_File'][0]
    parameters_to_condition_on = infos['parameters_to_condition_on']
    ndim = len(parameters_to_condition_on)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, dfdata = load_data(simfilename, datfilename)

    num_simulations = infos['sim_parameters']['n_sim']

    # Compute MURES for sim bank and data
    output_distribution = preprocess_input_distribution(
        df, parameters_to_condition_on[:-1] + ['x0', 'x0ERR', 'MU'])
    df['MURES'] = add_distance(output_distribution)

    output_distribution = preprocess_input_distribution(
        dfdata, parameters_to_condition_on[:-1] + ['x0', 'x0ERR', 'MU'])
    dfdata['MURES'] = add_distance(output_distribution)

    # Observed data — shape (n_sne, ndim) -> unsqueeze to (1, n_sne, ndim)
    x_obs = preprocess_data(parameters_to_condition_on, dfdata).unsqueeze(0)

    # --- Nominal model (model 1) ---
    dicts_1 = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations']]
    param_names_1 = infos['param_names']
    params_to_fit_1 = parameter_generation(param_names_1, dicts_1)
    priors_1 = prior_generator(param_names_1, dicts_1)
    layout_1 = build_layout(params_to_fit_1, dicts_1)

    nominal_sim = make_batched_simulator(
        layout_1, df, param_names_1, parameters_to_condition_on,
        dicts_1, dfdata, sub_batch=500, device='cpu'
    )

    print(f"Simulating {num_simulations} from nominal model ({args.CONFIG})...")
    theta_1 = priors_1.sample((num_simulations,))
    x1 = nominal_sim(theta_1)

    # NaN mask for model 1 (constant across comparisons)
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
        priors_2 = prior_generator(param_names_2, dicts_2)
        layout_2 = build_layout(params_to_fit_2, dicts_2)

        comp_sim = make_batched_simulator(
            layout_2, df, param_names_2, parameters_to_condition_on,
            dicts_2, dfdata, sub_batch=500, device='cpu'
        )

        print(f"Simulating {num_simulations} from comparison model ({model_path})...")
        theta_2 = priors_2.sample((num_simulations,))
        x2 = comp_sim(theta_2)

        mask2 = torch.isfinite(x2).all(dim=(1, 2))
        x2_clean = x2[mask2]
        print(f"  {model_path}: {x2_clean.shape[0]} valid / {x2.shape[0]} total")

        # Balance classes
        n_use = min(x1_clean.shape[0], x2_clean.shape[0])
        x_combined = torch.cat([x1_clean[:n_use], x2_clean[:n_use]], dim=0)
        y_combined = torch.cat([torch.zeros(n_use), torch.ones(n_use)])

        # Shuffle and split train/val
        perm = torch.randperm(2 * n_use)
        x_combined = x_combined[perm]
        y_combined = y_combined[perm]

        n_val = max(1, int(0.1 * 2 * n_use))
        n_train = 2 * n_use - n_val

        x_train, x_val = x_combined[:n_train], x_combined[n_train:]
        y_train, y_val = y_combined[:n_train], y_combined[n_train:]

        print(f"Training classifier: {n_train} train, {n_val} val samples")

        net = ModelComparisonNet(input_dim=ndim)
        net, best_val_loss = train_classifier(
            net, x_train, y_train, x_val, y_val, device=device
        )

        # Evaluate on observed data
        net.eval()
        with torch.no_grad():
            logit = net(x_obs.to(device)).squeeze()

        # BF_12 = p(x|M1) / p(x|M2) = exp(-logit)
        # (logit > 0 means classifier favours M2)
        log_bf = -logit.item()
        bf = torch.exp(torch.tensor(log_bf)).item()

        print(f"\nlog(BF) {args.CONFIG} vs {model_path}: {log_bf:.4f}")
        print(f"Bayes Factor: {bf:.4f}")
        if bf > 1:
            print(f"  -> Evidence favours {args.CONFIG}")
        else:
            print(f"  -> Evidence favours {model_path}")
