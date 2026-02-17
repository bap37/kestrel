import torch
import math

##############
# torch-compatible functions live in this code block; must start with "Dist"
##############

def DistGaussian(x, theta):
    """
    Gaussian likelihood for batched θ.

    x:     Tensor of shape (N,) or (batch_size, N)
    theta: Tensor of shape (batch_size, 2) -> [mu, sigma] for each θ

    Returns:
        Tensor of shape (batch_size, N) -> likelihood of each x for each θ
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)

    # Ensure theta has batch dimension
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # (1,2)
    batch_size = theta.shape[0]

    # Ensure x has shape (batch_size, N)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, N)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1)

    mu = theta[:, 0].unsqueeze(1)      # (batch_size, 1)
    sigma = theta[:, 1].unsqueeze(1)   # (batch_size, 1)

    return torch.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * math.sqrt(2.0 * math.pi))

def DistExponential(x, theta):
    """
    Exponential likelihood for batched θ.

    x:     Tensor of shape (N,) or (batch_size, N)
    theta: 

    Returns:
        Tensor of shape (batch_size, N) -> likelihood of each x for each θ
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)

    # Ensure theta has batch dimension
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # (1,2)
    batch_size = theta.shape[0]

    # Ensure x has shape (batch_size, N)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, N)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1)

    tau = theta[:, 0].unsqueeze(1)      # (batch_size, 1)\
    
    tau_proposal = x.mean(dim=1, keepdim=True)
    
    # Convert tau to lambda
    lambda_target = 1.0 / tau
    lambda_proposal = 1.0 / tau_proposal

    # Compute importance weights
    weights = (lambda_target / lambda_proposal) * torch.exp(-(lambda_target - lambda_proposal) * x)


    return weights



########################
# Functions below here have not been upgraded to use torch.

def DistLogNormal(input_distribution, *params):
    __name__ = "Lognormal"
    mu, std = params
    probs = np.exp(mu + std*input_distribution)
    
def DistDoubleGaussian(input_distribution, *params):
    __name__ = "DoubleGaussian"
    mu_1, sigma_1, mu_2, sigma_2, weight = params
    gauss_1 = np.exp(-0.5 * ((input_distribution - mu_1) / sigma_1) ** 2) / (sigma_1 * np.sqrt(2 * np.pi))
    gauss_2 = weight*np.exp(-0.5 * ((input_distribution - mu_2) / sigma_2) ** 2) / (sigma_2 * np.sqrt(2 * np.pi))    
    return gauss_1 + gauss_2 
