import pdb
import torch
import torch.nn as nn


def logpdf_lognormal(
    t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log-pdf of Log-Normal(t; mu, sigma).
    PDF: f(t) = (1 / (t * sigma * sqrt(2*pi))) * exp( - (log(t) - mu)^2 / (2*sigma^2) )
    log f(t) = -log(t) - log(sigma) - 0.5*log(2*pi) - ((log(t)-mu)^2) / (2*sigma^2)

    Parameters:
        t (torch.Tensor): Durations, with t > 0.
        mu (torch.Tensor): Mean of the logarithm.
        sigma (torch.Tensor): Standard deviation of the logarithm.
        eps (float): A small number to ensure numerical stability.

    Returns:
        torch.Tensor: The log probability density for each duration.
    """

    # Clamp t and sigma to avoid numerical issues
    t_clamped = torch.clamp(t, min=eps)
    sigma_clamped = torch.clamp(sigma, min=eps)

    log_t = torch.log(t_clamped)

    return (
        -torch.log(t_clamped)
        - torch.log(sigma_clamped)
        - 0.5 * 1.84
        - ((log_t - mu) ** 2) / (2 * sigma_clamped**2)
    )


def logpdf_exponential(
    t: torch.Tensor, rate: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log-pdf of Exponential(t; lambda = rate).
    PDF: f(t) = lambda * exp(-lambda * t),  t >= 0
    log f(t) = log(lambda) - lambda * t
    """
    # Ensure positivity
    rate_clamped = torch.clamp(rate, min=eps)
    t_clamped = torch.clamp(t, min=eps)
    return torch.log(rate_clamped) - rate_clamped * t_clamped


def logpdf_rayleigh(
    t: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log-pdf of Rayleigh(t; sigma).
    PDF: f(t) = (t / sigma^2) * exp(-t^2 / (2*sigma^2)),  t >= 0
    log f(t) = log(t) - 2*log(sigma) - (t^2) / (2*sigma^2)
    """
    sigma_clamped = torch.clamp(sigma, min=eps)
    t_clamped = torch.clamp(t, min=eps)
    return (
        torch.log(t_clamped)
        - 2 * torch.log(sigma_clamped)
        - (t_clamped**2) / (2 * sigma_clamped**2)
    )


def logpdf_normal(
    t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log-pdf of Normal(t; mu, sigma).
    PDF: f(t) = (1 / sqrt(2 * pi * sigma^2)) * exp(- (t - mu)^2 / (2 * sigma^2))
    log f(t) = -0.5 * log(2 * pi * sigma^2) - (t - mu)^2 / (2 * sigma^2)
    """
    sigma_clamped = torch.clamp(sigma, min=eps)
    return -0.5 * torch.log(2 * torch.pi * sigma_clamped**2) - (
        (t - mu) ** 2 / (2 * sigma_clamped**2)
    )


class DurationNLL(nn.Module):
    """
    A simple module to compute the negative log-likelihood for a batch of
    durations under either an exponential a Rayleigh distribution or a lognormal distribution.

    distribution in {'exponential', 'rayleigh', 'lognormal'}.
    """

    def __init__(self, distribution: str = "exponential", reduce: str = "mean"):
        """
        Args:
            distribution: 'exponential' or 'rayleigh'
            reduce: 'sum' or 'mean'
        """
        super().__init__()
        self.distribution = distribution.lower()
        assert self.distribution in [
            "exponential",
            "rayleigh",
            "lognormal",
            "normal",
        ], "distribution must be one of ['exponential', 'rayleigh', 'normal']"
        self.reduce = reduce

    def forward(
        self, durations: torch.Tensor, params: torch.Tensor, reduce=None
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood for a batch of durations.

        Parameters:
            durations: Tensor shape [B], each entry is T_i >= 0.
            params: Tensor shape [B] or shape [] (scalar) containing either:
                    - rate (lambda) for 'exponential'
                    - sigma for 'rayleigh'
        Returns:
            A scalar representing the NLL (summed or averaged over batch).
        """
        if reduce:
            self.reduce = reduce
        if self.distribution == "exponential":
            # params = rate = lambda
            ll = logpdf_exponential(durations, params)
        elif self.distribution == "rayleigh":
            # params = sigma
            ll = logpdf_rayleigh(durations, params)
        elif self.distribution == "lognormal":
            # params = (mu, sigma)
            mu, sigma = params
            ll = logpdf_lognormal(t=durations, mu=mu, sigma=sigma)
        elif self.distribution == "normal":
            # params = (mu, sigma)
            mu, sigma = params
            ll = logpdf_normal(t=durations, mu=mu, sigma=sigma)

        # Negative log-likelihood
        nll = -ll
        if self.reduce == "sum":
            return nll.sum()
        elif self.reduce == "mean":
            # Default to mean
            return nll.mean()
        elif self.reduce == "none":
            return nll
        else:
            raise ValueError(f"Unknown reduce option: {self.reduce}")
