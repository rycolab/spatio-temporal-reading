from calendar import c
from typing import Callable, Optional
import torch
import torch.nn as nn

from model.saccades.hp_integral import any_event_log_prob, any_event_log_prob_poisson
from model.saccades.hp_kernels import (
    constant_mu,
    exponential_temporal_decay,
    gaussian_distribution,
    gaussian_distribution_markov,
    gaussian_distribution_spatial_difference,
)

from .hp_intensity import Lambda


class SaccadesNLL(nn.Module):
    def __init__(
        self,
        baseline_fn: Callable[..., torch.Tensor],
        temporal_kernel_fn: Callable[..., torch.Tensor],
        division_factor_space: float,
        spatial_kernel_fn: Optional[Callable[..., torch.Tensor]] = None,
        any_event_log_fn: Optional[Callable[..., torch.Tensor]] = None,
        spatial_distance_kernel: bool = True,
    ):
        """
        Log-likelihood computation for a self-exciting point process (Hawkes) in a
        *batched* manner.

        Parameters:
        - baseline_fn: Function for baseline intensity.
        - temporal_kernel_fn: Function for temporal kernel.
        - spatial_kernel_fn: Function for spatial kernel. Defaults to None.
        - marker_fn: Function for marker interactions. Defaults to None.
        - marker_density_fn: Density function for marker probabilities. Defaults to None.
        - any_event_log_fn: Function to compute the log-probability of any event. Defaults to None.
        - spatial_distance_kernel: Whether to use distances in the spatial kernel.
        """
        super(SaccadesNLL, self).__init__()
        self.baseline_fn = baseline_fn
        self.temporal_kernel_fn = temporal_kernel_fn
        self.spatial_kernel_fn = spatial_kernel_fn
        self.any_event_log_fn = any_event_log_fn
        self.division_factor_space = division_factor_space

        # The 'Lambda' class should similarly be adapted to handle batched inputs.
        self.lambda_fn = Lambda(
            baseline_fn=baseline_fn,
            temporal_kernel_fn=temporal_kernel_fn,
            spatial_kernel_fn=spatial_kernel_fn,
            spatial_distance_kernel=spatial_distance_kernel,
        )

    def forward(
        self,
        current: torch.Tensor,  # Shape: [B, 3] or [B, 4] if markers
        history: torch.Tensor,  # Shape: [B, N_max, 3] or [B, N_max, 4]
        baseline_kwargs: dict = {},
        temporal_kwargs: dict = {},
        spatial_kwargs: dict = {},
        compute_probability: bool = False,
        radius=None,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the log-likelihood for a batch of processes in a vectorized manner.

        Parameters:
        - current: shape (B, 3) or (B, 4). Each row is the "new" event for a process.
                   Example (when no marker):
                     current[b] = [t_b, x_b, y_b]
                   Example (when markers):
                     current[b] = [t_b, x_b, y_b, marker_b]

        - history: shape (B, N_max, 3) or (B, N_max, 4). The padded history of each process.
                   Example (when no marker):
                     history[b, i] = [t_i, x_i, y_i]
                   Example (when markers):
                     history[b, i] = [t_i, x_i, y_i, marker_i]
                   where i goes up to N_max, with zero-padding for shorter sequences.

        Returns:
        - A scalar tensor: the *negative* log-likelihood summed over the batch.
        """

        # Small constant to avoid log(0).
        epsilon = 1e-4

        # -----------------------------------------------------
        # 1) Separate (time, location) -- and optionally marker
        # -----------------------------------------------------
        # current: shape [B, 3 or 4]
        # history: shape [B, N_max, 3 or 4]
        times_current = current[:, 0].unsqueeze(-1)  # shape [B,1]
        # For location, we assume columns 1,2 are x,y. If there's a marker, it's column 3.
        locations_current = current[:, 1:].unsqueeze(1)  # shape [B, 1, 2]

        # -------------------------------------------------------------------
        # 2) Compute intensity λ for the "new" event in each process (vectorized)
        # -------------------------------------------------------------------
        # Adapt self.lambda_fn so it can handle batched input.
        # E.g. it might accept:
        #   times_current, locations_current,
        #   times_history, locations_history, ...
        # and return a tensor of shape [B].
        intensity = self.lambda_fn(
            times_current,
            locations_current,
            history,
            baseline_kwargs=baseline_kwargs,
            temporal_kwargs=temporal_kwargs,
            spatial_kwargs=spatial_kwargs,
        )  # shape [B]
        # ---------------------------------------------------
        # log(λ) for each process
        log_likelihood = torch.log(intensity + epsilon)  # shape [B]
        # ---------------------------------------------------
        # 4) Subtract any "normalization" or "compensation" terms
        # ---------------------------------------------------
        # For Hawkes processes, you often have a term that corresponds to
        # ∫ λ(t) dt or some integral that ensures the process is normalized.
        # We'll assume any_event_log_fn is vectorized, returning shape [B].
        # normalization_factor = None
        if not self.any_event_log_fn:
            raise ValueError("any_event_log_fn must be provided for Hawkes processes.")
            # We'll pass `current` and `history` so that the function can integrate or
            # do what it needs in a batched manner.
        normalization_factor = self.any_event_log_fn(
            current=current,
            history=history,
            division_factor_space=self.division_factor_space,
            **baseline_kwargs,
            **temporal_kwargs,
            **spatial_kwargs,
        )  # shape [B]

        log_likelihood = -log_likelihood + normalization_factor

        # -----------------
        # 5) Final scalar
        # -----------------
        # Usually, we sum across the batch to get a scalar
        # (or you might want an average, depending on your training objective).
        # Then return the *negative* log-likelihood, typical for PyTorch training.

        if compute_probability:
            probability = self.lambda_fn.probability_over_ball_ratio(
                times_current,
                locations_current,
                history,
                baseline_kwargs=baseline_kwargs,
                temporal_kwargs=temporal_kwargs,
                spatial_kwargs=spatial_kwargs,
                ratio=radius,
                division_factor_space=self.division_factor_space,
            )

        else:
            probability = None
        if reduce == "mean":
            log_likelihood_scalar = torch.sum(log_likelihood) / current.shape[0]
            return log_likelihood_scalar, probability
        elif reduce == "none":
            return log_likelihood, probability


def set_saccadesNLL(cfg):

    if cfg.saccade_likelihood == "HomogenousPoisson":
        return SaccadesNLL(
            baseline_fn=constant_mu,
            temporal_kernel_fn=None,
            spatial_kernel_fn=None,
            any_event_log_fn=any_event_log_prob_poisson,
            division_factor_space=cfg.division_factor_space,
        )
    elif cfg.saccade_likelihood == "LastFixationModel":
        return SaccadesNLL(
            baseline_fn=constant_mu,
            temporal_kernel_fn=None,
            spatial_kernel_fn=gaussian_distribution_markov,
            any_event_log_fn=any_event_log_prob_poisson,
            spatial_distance_kernel=True,
            division_factor_space=cfg.division_factor_space,
        )

    elif cfg.saccade_likelihood == "StandardHawkesProcess":
        return SaccadesNLL(
            baseline_fn=constant_mu,
            temporal_kernel_fn=exponential_temporal_decay,
            spatial_kernel_fn=gaussian_distribution_spatial_difference,
            any_event_log_fn=any_event_log_prob,
            spatial_distance_kernel=True,
            division_factor_space=cfg.division_factor_space,
        )

    elif cfg.saccade_likelihood == "ExtendedHawkesProcess":
        return SaccadesNLL(
            baseline_fn=constant_mu,
            temporal_kernel_fn=exponential_temporal_decay,
            spatial_kernel_fn=gaussian_distribution,
            any_event_log_fn=any_event_log_prob,
            spatial_distance_kernel=False,
            division_factor_space=cfg.division_factor_space,
        )

    else:
        raise NotImplementedError
