from typing import Callable, Optional
from torch import nn
import torch
import numpy as np
from scipy.stats import ncx2


class Lambda(nn.Module):
    def __init__(
        self,
        baseline_fn: Callable[..., torch.Tensor],
        temporal_kernel_fn: Callable[..., torch.Tensor] = None,
        spatial_kernel_fn: Optional[Callable[..., torch.Tensor]] = None,
        marker_fn: Optional[Callable[..., torch.Tensor]] = None,
        spatial_distance_kernel: bool = True,
    ):
        """
        PyTorch-style self-exciting point process intensity function.

        Parameters:
        - baseline_fn: Baseline intensity function.
        - temporal_kernel_fn: Temporal interaction kernel.
        - spatial_kernel_fn: Spatial interaction kernel. Defaults to None.
        - marker_fn: Marker interaction function. Defaults to None.
        """
        super(Lambda, self).__init__()
        self.baseline_fn = baseline_fn
        self.temporal_kernel_fn = temporal_kernel_fn
        self.spatial_kernel_fn = spatial_kernel_fn
        self.marker_fn = marker_fn
        self.spatial_distance_kernel = spatial_distance_kernel

    def forward(
        self,
        current_time: torch.Tensor,
        current_location: torch.Tensor,
        event_history: torch.Tensor,
        baseline_kwargs: dict = {},
        temporal_kwargs: dict = {},
        spatial_kwargs: dict = {},
    ) -> torch.Tensor:
        """
        Vectorized forward pass to compute the intensity.

        Parameters:
        - current_time: Scalar tensor representing the current time.
        - current_location: Tensor representing the current spatial location/feature.
        - event_history: Tensor of shape [N, 3 or 4].
            If shape is [N,3], columns are (time, loc_x, loc_y)
            If shape is [N,4], columns are (time, loc_x, loc_y, mark)

        Returns:
        - Intensity value as a tensor (scalar).
        """

        # Compute baseline intensity
        if len(current_location.shape) != 3:
            raise ValueError(
                f"Wrong shape for current location, {current_location.shape}"
            )
        if current_location.shape[-1] != 2:
            raise ValueError(
                f"Wrong shape for current location, {current_location.shape}"
            )

        baseline_intensity = self.baseline_fn(
            current_location.squeeze(1), **baseline_kwargs
        )

        if len(event_history.shape) != 3:
            raise ValueError(f"Wrong shape for event history, {event_history.shape}")
        if event_history.shape[2] != 3:
            raise ValueError(f"Wrong shape for event history, {event_history.shape}")

        # Extract time and location from event_history
        past_time = event_history[:, :, 0]
        past_location = event_history[:, :, 1:]

        if not is_sorted_ignoring_zeros(past_time):
            raise ValueError(f"Past Time is not sorted in ascending order! {past_time}")

        # past time should have shape [batch_size, L_max]
        if len(past_time.shape) != 2:
            raise ValueError(f"Wrong shape for past time, {past_time.shape}")

        mask = (past_time != 0) * (past_time < current_time)
        # Create a mask for events that occurred before current_time and that had non-zero time (i.e. are not part of padding)
        if len(mask.shape) != 2:
            raise ValueError(f"Wrong shape for mask, {mask.shape}")
        if mask.shape[1] != past_time.shape[1]:
            raise ValueError(f"Wrong shape for mask, {mask.shape}")
        if torch.any(input=mask) and (
            bool(self.temporal_kernel_fn or self.spatial_kernel_fn)
        ):
            # Filter only those events
            # filtered_past_time = past_time[mask]
            temporal_contribution, spatial_contribution = (
                self.compute_kernels_contributions(
                    current_time,
                    past_time,
                    current_location,
                    past_location,
                    mask,
                    temporal_kwargs,
                    spatial_kwargs,
                )
            )

            # Combine all contributions
            total_contribution = temporal_contribution * spatial_contribution
            check_for_nans(total_contribution, "total_contribution")
            # Sum over all contributing events and add to intensity
            intensity = torch.sum(total_contribution, axis=1).unsqueeze(-1)
            if intensity.shape != baseline_intensity.shape:
                raise ValueError(f"Wrong shape for intensity, {intensity.shape}")

            intensity = intensity + baseline_intensity
        else:
            return baseline_intensity

        if (intensity < 0).any():
            raise ValueError(f"Negative intensity found: {intensity}")

        return intensity

    def compute_kernels_contributions(
        self,
        current_time,
        past_time,
        current_location,
        past_location,
        mask,
        temporal_kwargs,
        spatial_kwargs,
    ):
        delta_t = (current_time - past_time) * mask
        # Compute deltas
        if self.spatial_distance_kernel:
            # [B, 1, 3] - [B, L_max, 3] = [B, L_max, 3]
            delta_x = current_location - past_location
            delta_x = delta_x * mask.unsqueeze(-1)
            check_for_nans(delta_x, "delta_x")

        check_for_nans(delta_t, "delta_t")

        # Compute spatial contributions for all filtered events (if available)

        if self.spatial_distance_kernel:
            spatial_contribution = self.spatial_kernel_fn(
                delta_x, **spatial_kwargs, mask=mask
            )
        else:

            spatial_contribution = self.spatial_kernel_fn(
                current_location, **spatial_kwargs, mask=mask
            )
        if self.temporal_kernel_fn:
            # Compute temporal contributions for all filtered events
            temporal_contribution = self.temporal_kernel_fn(
                delta_t, **temporal_kwargs, mask=mask
            )
        else:
            temporal_contribution = torch.full_like(spatial_contribution, fill_value=1)
            temporal_contribution[spatial_contribution <= 0] = 0

        check_for_nans(temporal_contribution, "temporal_contribution")
        check_for_nans(spatial_contribution, "spatial_contribution")

        return temporal_contribution, spatial_contribution

    def probability_over_ball_ratio(
        self,
        current_time,
        current_location,
        event_history,
        baseline_kwargs,
        temporal_kwargs,
        spatial_kwargs,
        ratio,
        division_factor_space,
    ):
        with torch.no_grad():
            baseline_intensity = self.baseline_fn(
                current_location.squeeze(1), **baseline_kwargs
            )
            S_area = (2000 / division_factor_space) * (700 / division_factor_space)

            integral_circle_baseline_intensity = (
                baseline_intensity * torch.pi * ratio**2
            )

            integral_domain_baseline_intensity = S_area * baseline_intensity

            # Extract time and location from event_history
            past_time = event_history[:, :, 0]
            past_location = event_history[:, :, 1:]
            mask = (past_time != 0) * (past_time < current_time)
            # Create a mask for events that occurred before current_time and that had non-zero time (i.e. are not part of padding)

            baseline_intensity = self.baseline_fn(
                current_location.squeeze(1), **baseline_kwargs
            )
            temporal_contribution, _ = self.compute_kernels_contributions(
                current_time,
                past_time,
                current_location,
                past_location,
                mask,
                temporal_kwargs,
                spatial_kwargs,
            )

            spatial_integration = bivariate_normal_ball_prob(
                current_location.squeeze(1),
                spatial_kwargs["mean"],
                r=ratio,
                sigma=spatial_kwargs["sigma"],
            )

            element_wise_integral = spatial_integration * temporal_contribution * mask

            integral_circle_excitation_part = torch.sum(
                element_wise_integral, axis=1
            ).unsqueeze(-1)

            integral_domain_excitation_part = torch.sum(
                temporal_contribution, axis=1
            ).unsqueeze(-1)

            nominator = (
                integral_circle_baseline_intensity + integral_circle_excitation_part
            )
            denominator = (
                integral_domain_baseline_intensity + integral_domain_excitation_part
            )

            probability = (nominator) / (denominator)

            probability = probability[probability.isnan() == False]

            return probability


def is_sorted_ignoring_zeros(tensor):
    for row in tensor:
        # Filter out zeros
        non_zero_elements = row[row > 0]
        # Check if non-zero elements are sorted
        if not torch.all(non_zero_elements == torch.sort(non_zero_elements)[0]):
            return False
    return True


def check_for_nans(tensor, tensor_name):
    """Check if a tensor contains NaNs and raise an error if so."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaNs detected in {tensor_name}!")


def bivariate_normal_ball_prob(
    centers: torch.Tensor, means: torch.Tensor, r: float, sigma: float
) -> torch.Tensor:
    """
    Computes P(||X - renter|| <= r) for X ~ N(mean, sigma^2 I_2)
    using the noncentral chi-square CDF.

    Args:
        centers : [B, 2] tensor (each row is a point (x1, x2))
        means   : [B, M, 2] tensor, where means[b, m, :] is the center for the b-th row, m-th mean
        r       : scalar radius
        sigma   : scalar standard deviation

    Returns:
        probs : [B, M] tensor where
                probs[b, m] = P(||X_{b,m} - centers[b]|| <= r).
    """
    # Move data to NumPy (no grad needed)
    centers_np = centers.detach().cpu().numpy()  # shape [B, 2]
    means_np = means.detach().cpu().numpy()  # shape [B, M, 2]

    # Compute squared Euclidean distances between each renter and each mean
    # Resulting shape: [B, M]
    d2 = np.sum((means_np - centers_np[:, None, :]) ** 2, axis=-1)

    # Scale distances and compute the noncentrality parameter: delta = d^2/sigma^2.
    nc = d2 / (sigma)
    # The threshold value for the chi-square variable is r^2/sigma^2.
    z_val = (r**2) / (sigma)

    # Evaluate the CDF in a vectorized way. ncx2.cdf broadcasts over nc.
    cdf_array = ncx2.cdf(z_val, df=2, nc=nc)

    # For numerical safety, clip the result to [0, 1]
    np.clip(cdf_array, 0.0, 1.0, out=cdf_array)

    # Convert back to a PyTorch tensor (preserving the device and dtype)
    return torch.from_numpy(cdf_array).to(centers.device, centers.dtype)
