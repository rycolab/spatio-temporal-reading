import torch


def constant_mu(x, mu: torch.Tensor) -> torch.Tensor:

    # make a tensor  of size [B, L_max] with all ones
    mask = torch.ones(x.shape[0], 1, device=mu.device)
    # Baseline intensity function as a constant
    return mu * mask


def exponential_temporal_decay(
    x, alpha: torch.Tensor, beta: torch.Tensor, mask
) -> torch.Tensor:

    if alpha.dim():
        alpha = alpha.squeeze(-1)
    if beta.dim():
        beta = beta.squeeze(-1)

    temporal_kernel = alpha * torch.exp(-beta * x) * mask

    if torch.isnan(temporal_kernel).any():
        raise ValueError("NaNs detected in temporal kernel!")
    # Exponential temporal decay kernel
    return temporal_kernel


def gaussian_distribution_spatial_difference(x, mean, sigma, mask):
    return gaussian_distribution(x=x, mean=0, sigma=sigma, mask=mask)


def gaussian_distribution_markov(x, mean, sigma, mask):
    cumsum = mask.cumsum(dim=1)
    sum_per_row = mask.sum(dim=1, keepdim=True)
    mask = mask * (cumsum == sum_per_row)

    # Get the index of the last 1 in each batch element
    return gaussian_distribution_spatial_difference(x, mean, sigma, mask=mask)


def gaussian_distribution(x, mean, sigma, mask):

    # mean should be [B,L_max,2]
    mean = mean * mask.unsqueeze(-1)

    # Gaussian spatial decay kernel
    d = x.shape[-1]  # Number of dimensions
    if d != 2:
        raise ValueError("Only 2D spatial decay is supported")

    kernel = -torch.sum((x - mean) ** 2, axis=-1) * mask

    if sigma.dim():
        kernel = kernel.unsqueeze(-1)
    spatial_kernel = (
        (1 / (2 * torch.pi * (sigma))) * torch.exp(kernel / (2 * (sigma))) * mask
    )

    if torch.isnan(spatial_kernel).any():
        raise ValueError("NaNs detected in spatial kernel!")
    return spatial_kernel
