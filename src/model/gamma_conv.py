import torch


def shifted_gamma(tdiff, alpha, beta, delta):
    """
    tdiff: B x M
    alpha, beta, delta: scalar tensors

    returns a B x M tensor
    """

    if delta < 0:
        raise ValueError("delta must be non-negative")
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1")

    # beta^alpha / Gamma(alpha)
    # gamma = torch.lgamma(alpha).exp()

    gamma = torch.lgamma(alpha).exp()

    gamma_factor = beta**alpha / gamma

    # (t_)^(alpha-1) * exp(-beta * t_)
    val = (
        gamma_factor
        * ((tdiff + delta) ** (alpha - 1.0))
        * torch.exp(-beta * (tdiff + delta))
    )
    return val


def gamma_convolution(
    input_feature,
    input_feature_times,
    current_time,
    alpha,
    beta,
    delta,
    tracker,
    logger,
    mark: str = None,
):
    """
    input_features: last dimension is time

    """

    #################################################################################
    # WE MASK ALL THE PAST FIXATIONS THAT OCCUR AFTER THE CURRENT RESPONSE
    #################################################################################
    current_times = current_time.unsqueeze(-1)

    mask_check = input_feature_times == current_times
    if mask_check.any(dim=1).sum() != current_times.shape[0]:
        raise ValueError("Not all current times match the input feature times. ")
    time_mask = (
        (input_feature_times < current_times)
        * (input_feature_times != 0.0)
        * (input_feature.isnan() == False)
    )

    input_feature = input_feature.masked_fill(time_mask == False, 0.0)
    input_feature = input_feature * time_mask

    # QUICK ASSESSMENT THAT THE INPUT FEATURES ARE SORTED IN ASCENDING ORDER
    tensor = input_feature_times.clone().detach()
    tensor[tensor == 0.0] = float("inf")
    assert torch.all(tensor[:, :-1] <= tensor[:, 1:])

    B = current_times.shape[0]
    T = input_feature_times.shape[1]
    # p = input_features.shape[2]
    # assert p == 1

    matches = torch.nonzero(input_feature_times == current_times, as_tuple=False)

    """
    values_minus_1 = input_feature_times[matches[:, 0], matches[:, 1]] - (
        input_feature_times[matches[:, 0], matches[:, 1] - 1]
    )
    values_minus_2 = input_feature_times[matches[:, 0], matches[:, 1]] - (
        input_feature_times[matches[:, 0], matches[:, 1] - 2]
    )
    values_minus_3 = input_feature_times[matches[:, 0], matches[:, 1]] - (
        input_feature_times[matches[:, 0], matches[:, 1] - 3]
    )
    values_minus_4 = input_feature_times[matches[:, 0], matches[:, 1]] - (
        input_feature_times[matches[:, 0], matches[:, 1] - 4]
    )
    values_minus_5 = input_feature_times[matches[:, 0], matches[:, 1]] - (
        input_feature_times[matches[:, 0], matches[:, 1] - 5]
    )
    values_minus_6 = input_feature_times[matches[:, 0], matches[:, 1]] - (
        input_feature_times[matches[:, 0], matches[:, 1] - 6]
    )
    """

    #####################################################
    # WE CENTER THE TIME-AXIS AROUND THE CURRENT RESPONSE
    #####################################################

    # tdfiff[n, x] = current_times[n]- input_feature_times[x]
    tdiff = (current_times - input_feature_times) * time_mask
    # Evaluate IRF for all (n, x)
    G_k = shifted_gamma(tdiff, alpha, beta=beta, delta=delta)  # shape (N, X)
    """ 
    We comment out the following lines because it makes the logger output too verbose.
    # We can uncomment them if we want to see the values of G_k_minus_1, G_k_minus_2, etc.
    # and the time differences values_minus_1, values_minus_2, etc.
    # This is useful for debugging or understanding the model's behavior
    
    
    # t = G_k * time_mask
    G_k_minus_1 = G_k[matches[:, 0], matches[:, 1] - 1]
    G_k_minus_2 = G_k[matches[:, 0], matches[:, 1] - 2]
    G_k_minus_3 = G_k[matches[:, 0], matches[:, 1] - 3]
    G_k_minus_4 = G_k[matches[:, 0], matches[:, 1] - 4]
    G_k_minus_5 = G_k[matches[:, 0], matches[:, 1] - 5]
    G_k_minus_6 = G_k[matches[:, 0], matches[:, 1] - 6]

    logger.info(
        "Current Gamma parameters: alpha={}, beta={}, delta={}, mark = {}".format(
            alpha.item(),
            beta.item(),
            delta.item(),
            mark if mark is not None else "main",
        )
    )
    """
    if tracker is not None:

        """
        logger.info(
            "Current Gamma values: G_k-1={}, G_k-2={}, G_k-3={}, G_k-4={}, G_k-5={}, G_k-6={}".format(
                G_k_minus_1.mean().item(),
                G_k_minus_2.mean().item(),
                G_k_minus_3.mean().item(),
                G_k_minus_4.mean().item(),
                G_k_minus_5.mean().item(),
                G_k_minus_6.mean().item(),
            )
        )
        logger.info(
            "Time differences:  tdiff-1={}, tdiff-2={}, tdiff-3={}, tdiff-4={}, tdiff-5={}, tdiff-6={}".format(
                values_minus_1.mean().item(),
                values_minus_2.mean().item(),
                values_minus_3.mean().item(),
                values_minus_4.mean().item(),
                values_minus_5.mean().item(),
                values_minus_6.mean().item(),
            )
        )
        """
        tracker["gamma_alpha"].append(alpha.item())
        tracker["gamma_beta"].append(beta.item())
        tracker["gamma_delta"].append(delta.item())
    # Multiply by the predictor values for column k:
    # but Xpred[:, k] is shape (X,)
    # We'll broadcast (N, X) * (X,) by aligning on X dim:
    # => need to transpose or unsqueeze.
    # The simplest is to unsqueeze: shape (1, X).
    G_k_weighted = G_k * input_feature  # shape (N, X)
    # logger.info("Output convolution: {}".format(G_k_weighted.mean().item()))

    # Sum across x to get final convolved predictor for dimension k
    result = G_k_weighted.sum(dim=1).unsqueeze(-1)

    return result
