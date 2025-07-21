import torch
import pdb


def any_event_log_prob_poisson(
    current,
    history,
    alpha,
    beta,
    sigma,
    mean,
    mu,
    division_factor_space,
    ball_prob_comp=False,
):
    if ball_prob_comp:
        raise NotImplementedError

    return any_event_log_prob(
        current,
        history,
        alpha,
        beta,
        sigma,
        mean,
        mu=mu,
        division_factor_space=division_factor_space,
        ball_prob_comp=ball_prob_comp,
        poisson=True,
    )


def any_event_log_prob(
    current,
    history,
    alpha,
    beta,
    sigma,
    mean,
    mu,
    division_factor_space,
    ball_prob_comp=False,
    poisson=False,
):
    epsilon = 1e-9

    # Create the mask
    mask = history[:, :, 0] != 0

    # boxes [B, N_max, 5]

    current = current.unsqueeze(1)

    if (history == current).prod(axis=2).sum(axis=1).any() != 1:
        pdb.set_trace()

    curr_idx_hist = (
        ((history == current).prod(axis=2) == 1)
        .nonzero(as_tuple=False)[:, 1]
        .unsqueeze(-1)
    )
    if curr_idx_hist.shape[0] != current.shape[0]:
        pdb.set_trace()

    times_history = history[:, :, 0]
    zeros_column = torch.zeros(times_history.shape[0], 1, device=times_history.device)

    # Concatenate along the second dimension (dim=1)
    times_history_shifted = torch.cat((zeros_column, times_history), dim=1)

    upper_t = times_history.gather(1, curr_idx_hist)
    lower_t = times_history_shifted.gather(1, curr_idx_hist)
    if ball_prob_comp:
        upper_t = torch.full_like(upper_t, 1000, device=upper_t.device)

    mask = current[:, :, 0] > times_history

    first_term = lower_t - times_history * mask
    second_term = upper_t - times_history * mask

    if beta.dim():
        beta = beta.squeeze(-1)
    if alpha.dim():
        alpha = alpha.squeeze(-1)

    integral_elementwise = (
        (torch.exp(-beta * first_term) - torch.exp(-beta * second_term))
        * alpha
        / (beta + epsilon)
    ) * mask
    integral_second_part = torch.sum(integral_elementwise, axis=1).unsqueeze(-1)

    S_area = (2000 / division_factor_space) * (700 / division_factor_space)
    baseline_intensity_contribution = S_area * (upper_t - lower_t) * mu

    if torch.isnan(integral_second_part + baseline_intensity_contribution).any():
        pdb.set_trace()
    if poisson == True:
        return baseline_intensity_contribution
    else:
        # Log-probability of any event occurring
        return baseline_intensity_contribution + integral_second_part
