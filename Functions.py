import torch
import math

##############
# torch-compatible functions live in this code block; must start with "Dist"
##############

def DistGaussian(x, theta, correlation):
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


def DistGaussian_EVOL(x, theta, correlation):
    """
    Gaussian likelihood with μ evolving via m * correlation.

    x:           Tensor of shape (N,) or (batch_size, N)
    theta:       Tensor of shape (batch_size, 3) -> [mu0, sigma, m]
    correlation: Same shape as x

    Returns:
        Tensor of shape (batch_size, N)
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    correlation = torch.as_tensor(correlation, dtype=torch.float32)

    # Ensure theta has batch dimension
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # (1,3)
    batch_size = theta.shape[0]

    # Ensure x has shape (batch_size, N)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1)

    # Ensure correlation matches x
    if correlation.ndim == 1:
        correlation = correlation.unsqueeze(0)
    if correlation.shape[0] == 1 and batch_size > 1:
        correlation = correlation.expand(batch_size, -1)

    mu0 = theta[:, 0].unsqueeze(1)   # (batch_size, 1)
    sigma = theta[:, 1].unsqueeze(1)
    m = theta[:, 2].unsqueeze(1)

    # μ varies per data point
    mu = mu0 + m * correlation       # (batch_size, N)

    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))

def DistExponential(x, theta, correlation):
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
    
    #tau_proposal = x.mean(dim=1, keepdim=True)
    tau_proposal = 0.5 #Hard coded to 0.5 right now on account of selection effects.
    
    # Convert tau to lambda
    lambda_target = 1.0 / tau
    lambda_proposal = 1.0 / tau_proposal

    # Compute importance weights
    weights = (lambda_target / lambda_proposal) * torch.exp(-(lambda_target - lambda_proposal) * x)

    return weights

def DistExponential_EVOL(x, theta, correlation):

    return
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
    
    #tau_proposal = x.mean(dim=1, keepdim=True)
    tau_proposal = 0.5 #Hard coded to 0.5 right now on account of selection effects.
    
    # Convert tau to lambda
    lambda_target = 1.0 / tau
    lambda_proposal = 1.0 / tau_proposal

    # Compute importance weights
    weights = (lambda_target / lambda_proposal) * torch.exp(-(lambda_target - lambda_proposal) * x)

    return weights


def DistDoubleGaussian(x, theta, correlation):
    """
    Gaussian likelihood for batched θ.

    x:     Tensor of shape (N,) or (batch_size, N)
    theta: Tensor of shape (batch_size, 2) -> [mu1, sigma1, mu2, sigma2, a] for each θ

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

    mu1 = theta[:, 0].unsqueeze(1)      # (batch_size, 1)
    sigma1 = theta[:, 1].unsqueeze(1)   

    #weight a is only applied to one Gaussian to reduce dimensionality.
    mu2 = theta[:, 2].unsqueeze(1)      
    sigma2 = theta[:, 3].unsqueeze(1)  
    a = theta[:, 4].unsqueeze(1)     

    
    G1 = torch.exp(-0.5 * ((x - mu1)/sigma1)**2) / (sigma1 * math.sqrt(2.0 * math.pi))
    G2 = a*torch.exp(-0.5 * ((x - mu2)/sigma2)**2) / (sigma2 * math.sqrt(2.0 * math.pi))

    return G1+G2

def DistLogistic(x, theta, correlation):
    """
    Logistic-shaped likelihood for importance sampling.

    x:           (N,) or (B, N)
    theta:       (B, 3) -> [L, k, sigma]
    correlation: (N,) or (B, N)

    Returns:
        (B, N) likelihood weights
    """
    x = torch.as_tensor(x, dtype=torch.float32, device=theta.device)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    correlation = torch.as_tensor(correlation, dtype=torch.float32, device=theta.device)

    # --- Ensure batch dims ---
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    B = theta.shape[0]

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and B > 1:
        x = x.expand(B, -1)

    if correlation.ndim == 1:
        correlation = correlation.unsqueeze(0)
    if correlation.shape[0] == 1 and B > 1:
        correlation = correlation.expand(B, -1)

    # --- Parameters ---
    L     = theta[:, 0].unsqueeze(1)
    k     = theta[:, 1].unsqueeze(1)
    sigma = theta[:, 2].unsqueeze(1)

    # --- Logistic mean ---
    mean = L / (1 + torch.exp(k * correlation - 10)) + 2

    # --- Gaussian likelihood ---
    z = (x - mean) / sigma
    pdf = torch.exp(-0.5 * z**2) / (sigma * math.sqrt(2.0 * math.pi))

    return pdf

def DistGamma(x, theta, correlation):

    """
    Gamma likelihood for batched θ.

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
    gamma = theta[:,1].unsqueeze(1)
    
    # Convert tau to lambda
    lambda_target = 1.0 / tau

    # Compute importance weights
    weights = x**(gamma - 1) * torch.exp(-(lambda_target) * x)

    return weights


def DistTruncatedGaussian(x, theta, correlation=None, low=1.2, high=float("inf")):
    """
    Truncated Gaussian likelihood for batched θ.

    x:     Tensor (N,) or (batch_size, N)
    theta: Tensor (batch_size, 2) -> [mu, sigma]

    low, high: truncation bounds (can be scalars or tensors)

    Returns:
        Tensor (batch_size, N)
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)

    # Ensure theta has batch dimension
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    batch_size = theta.shape[0]

    # Ensure x has shape (batch_size, N)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1)

    mu = theta[:, 0].unsqueeze(1)      # (B,1)
    sigma = theta[:, 1].unsqueeze(1)   # (B,1)

    # --- Gaussian PDF ---
    pdf = torch.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * math.sqrt(2.0 * math.pi))

    # --- CDF using erf ---
    def normal_cdf(z):
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    a = (low - mu) / sigma
    b = (high - mu) / sigma

    Z = normal_cdf(b) - normal_cdf(a)  # normalization

    # Avoid division by zero
    Z = torch.clamp(Z, min=1e-12)

    # --- Apply truncation ---
    mask = (x >= low) & (x <= high)

    truncated_pdf = torch.zeros_like(pdf)
    truncated_pdf[mask] = pdf[mask] / Z.expand_as(pdf)[mask]

    return truncated_pdf