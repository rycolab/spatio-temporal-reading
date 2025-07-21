from copy import deepcopy
import pdb
import matplotlib
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
from consts import NUMBER_OF_FRAMES_ANIMATION
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import numpy as np
import torch
import math
from src.paths import DATA_DIR

matplotlib.use("Agg")


def animating_predicted_reading_session(
    inference_model, likelihood, dataset, output_dir, division_factor_space
):

    # *******************
    # * Retrieve held out session
    # *******************

    with torch.no_grad():
        # -1 is used to get the information for the entire held out sequence
        (
            input_features_stpp,
            input_features_dur,
            history_points,
            duration_points,
            boxes,
            duration_onset_real_times,
        ) = dataset[-1]
        sample_idx = dataset.held_out_reader
        text_id, reader_id, _ = sample_idx

        input_features_stpp = input_features_stpp.unsqueeze(
            0
        )  # [1, Sequence_Length , n_predictors]
        history_points = history_points.unsqueeze(0)  # [1, Sequence_Length , 3]
        time_points = history_points[:, :, 0].detach().numpy()
        locations = history_points[:, :, 1:3].squeeze(0).detach().numpy()

        boxes = boxes.unsqueeze(0)

        # ********************* #
        # * Animation Setup   * #
        # ********************* #

        x_min, x_max = 0, 2000
        y_min, y_max = 0, 700
        num_x, num_y = 200, 200
        x = np.linspace(x_min, x_max, num_x)
        y = np.linspace(y_min, y_max, num_y)
        X, Y = np.meshgrid(x, y)

        inference_model.eval()
        inference_model.to(device="cpu")
        t_start, t_end = 0.00, float(time_points.max())
        t_values = np.linspace(t_start, t_end, NUMBER_OF_FRAMES_ANIMATION)

        # ****************** #
        # * Model Output   * #
        # ****************** #

        neural_pars = inference_model(
            input_features_stpp,
            input_features_dur,
            None,
        )
        mu, alpha, beta, sigma, means, _ = neural_pars

        kwargs_outputs = {
            "baseline_kwargs": {"mu": mu},
            "temporal_kwargs": {"alpha": alpha, "beta": beta},
            "spatial_kwargs": {"mean": means, "sigma": sigma},
        }

        # ******************* #
        # * Animation Loop  * #
        # ******************* #

        anim = compute_animation_loop(
            X=X,
            Y=Y,
            t_values=t_values,
            likelihood_func=likelihood,
            history_points=history_points,
            kwargs_outputs=kwargs_outputs,
            ordered_time=time_points.squeeze(0),
            ordered_locations=locations,
            text_id=text_id,
            division_factor_space=division_factor_space,
        )

        mp4_writer = FFMpegWriter(fps=10, metadata={"title": "Animation"})
        anim.save(
            str(
                output_dir / f"animation_text_{sample_idx[0]}_reader{sample_idx[1]}.mp4"
            ),
            writer=mp4_writer,
        )


def compute_animation_loop(
    X,
    Y,
    t_values,
    likelihood_func,
    history_points,
    kwargs_outputs,
    ordered_time,
    ordered_locations,
    text_id,
    division_factor_space,
):
    """
    Create an animated contour plot of lambda_inference over time and space,
    overlaying scatter points that become active based on their time_points and
    plotting a circle of radius RATIO around the next unobserved point.

    In addition, contour lines are drawn and labeled with their numeric values.
    The filled contours and colorbar use a fixed normalization so that the color
    gradient remains consistent over the entire animation.

    Parameters:
    - X, Y (np.ndarray): Meshgrid arrays for spatial coordinates.
    - t_values (np.ndarray): Array of time points.
    - likelihood_func (object): Object with a method 'lambda_fn' that returns the lambda values.
    - history_points (torch.Tensor): Tensor of historical data points.
    - kwargs_outputs (dict): Dictionary of keyword arguments for likelihood_func.
    - ordered_time (np.ndarray): Array of time points corresponding to positions.
    - ordered_locations (np.ndarray): Array of positions [x, y].
    - text_id (int): Identifier used to select the background image.

    Returns:
    - ani (FuncAnimation): Matplotlib animation object.
    """

    # Load the background image.
    if text_id > 9:
        image_path = DATA_DIR / "MECO" / "texts_en_images" / f"Item_{text_id}.png"
    else:
        image_path = DATA_DIR / "MECO" / "texts_en_images" / f"Item_0{text_id}.png"
    image = mpimg.imread(image_path)

    # Create figure and axes.
    fig, ax = plt.subplots(figsize=(10, 10), dpi=400)
    ax.imshow(image)

    # Define spatial grid boundaries.
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    ax.set_ylim(bottom=750, top=0)

    side_len = X.shape[0]

    # Build the spatial domain.
    X_vect = torch.tensor(X.reshape(-1, 1))
    Y_vect = torch.tensor(Y.reshape(-1, 1))
    spatial_domain = torch.cat((X_vect, Y_vect), axis=1).unsqueeze(1)
    times_initial = torch.tensor(t_values[0]).expand(X_vect.shape[0], 1)

    # Expand tensors in history_points and in kwargs_outputs.
    history_points = history_points.expand(X_vect.shape[0], -1, -1)
    if "boxes" in kwargs_outputs["baseline_kwargs"]:
        boxes = kwargs_outputs["baseline_kwargs"]["boxes"]
        kwargs_outputs["baseline_kwargs"]["boxes"] = boxes.expand(
            X_vect.shape[0], -1, -1
        )
    mean = kwargs_outputs["spatial_kwargs"]["mean"]
    kwargs_outputs["spatial_kwargs"]["mean"] = mean.expand(X_vect.shape[0], -1, -1)

    # Compute lambda values for the initial frame.
    Z_initial = (
        likelihood_func.lambda_fn(
            times_initial,
            spatial_domain / division_factor_space,
            history_points,
            **kwargs_outputs,
        )
        .detach()
        .numpy()
        .reshape(side_len, side_len)
    )

    # Create a fixed normalization (here from 0 to 1.10)
    norm_fixed = plt.Normalize(vmin=0, vmax=10)

    # Create a ScalarMappable using the fixed colormap and normalization.
    # This will guarantee that the colorbar always shows the full gradient.
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm_fixed)
    sm.set_array([])

    # Draw the initial filled contour using the fixed normalization.
    contour = ax.contourf(
        X,
        Y,
        Z_initial,
        levels=150,
        cmap="viridis",
        norm=norm_fixed,
        alpha=0.3,
        linewidth=0,
    )
    # Create the colorbar from the ScalarMappable.
    cbar = fig.colorbar(mappable=sm, ax=ax, label="Lambda Inference Intensity")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    # ax.set_title(f"Lambda Inference at t = {float(times_initial[0]):.2f}")

    # Initialize scatter plots.
    scatter = ax.scatter(
        [],
        [],
        c="red",
        edgecolors="none",
        s=30,
        label="Past Observed Points",
        zorder=10,
        alpha=0.4,
    )
    scatter_g = ax.scatter(
        [],
        [],
        c="green",
        edgecolors="none",
        s=30,
        label="Next Unobserved Point",
        zorder=10,
        alpha=1,
    )
    legend = ax.legend(loc="best")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1)
    # Initialize the circle patch and storage lists.
    circle_patch = None
    contours = [contour]  # List for filled contour sets.
    line_contours = []  # List for contour lines and labels.

    global cumulative_likelihood
    cumulative_likelihood = []
    global cumulative_probability
    cumulative_probability = []

    def update(frame):
        nonlocal circle_patch

        # Remove previous filled contours.
        for c in contours:
            for coll in c.collections:
                coll.remove()
        contours.clear()

        # Remove previous contour lines and labels.
        for artist in line_contours:
            try:
                artist.remove()
            except Exception:
                pass
        line_contours.clear()

        # Compute lambda values for the current frame.
        times = torch.tensor(t_values[frame]).expand(X_vect.shape[0], 1)
        Z = (
            likelihood_func.lambda_fn(
                times,
                spatial_domain / division_factor_space,
                history_points,
                **kwargs_outputs,
            )
            .detach()
            .numpy()
            .reshape(side_len, side_len)
        )

        current_max = Z.max()
        # if current_max > norm_fixed.vmax
        if current_max > 0:
            norm_fixed.vmax = current_max
            sm.set_norm(norm_fixed)
            cbar.mappable.set_clim(0, current_max)
            cbar.update_normal(cbar.mappable)

        if np.isclose(Z.min(), Z.max()):
            Z[0][0] = Z[0][0] + 0.1
        # Draw the updated filled contour using the fixed normalization.
        contour = ax.contourf(
            X, Y, Z, levels=200, cmap="viridis", norm=norm_fixed, alpha=0.4, linewidth=0
        )
        contours.append(contour)
        # (No need to update the colorbar since it's driven by the ScalarMappable 'sm')

        # Draw contour lines.
        # Here, we use fixed contour levels that span the full range.
        line_levels = np.linspace(0, current_max, 10)
        line_levels = np.unique(np.sort(line_levels))

        if np.isclose(Z.min(), Z.max()):
            # Define a small delta to create an artificial range for contours
            delta = 0.1  # Adjust this value as needed
            line_levels = np.linspace(Z.min() - delta, Z.max() + delta, 10)
        else:
            line_levels = np.linspace(Z.min(), Z.max(), 10)

        # Update active scatter points.
        active_indices = np.where(ordered_time <= float(times[0]))[0]
        active_points = ordered_locations[active_indices]
        in_bounds = (
            (active_points[:, 0] >= x_min)
            & (active_points[:, 0] <= x_max)
            & (active_points[:, 1] >= y_min)
            & (active_points[:, 1] <= y_max)
        )
        active_points = active_points[in_bounds]
        scatter.set_offsets(active_points * division_factor_space)
        print(
            f"Frame {frame}: t={float(times[0]):.2f}, Active Points={len(active_points)}"
        )
        title = f"Predicting the Next Fixation Landing Spot; reader:70, text:3\n"
        # title += f"Lambda Inference at t = {float(times[0]):.2f} \n"
        # Update the next unobserved point if available.
        if len(active_indices):
            if active_indices[-1] + 1 < len(ordered_locations):
                next_point_coord = (
                    ordered_locations[active_indices[-1] + 1] * division_factor_space
                )
                radius = 100
                probability_est = probability_in_circle(
                    X, Y, Z, x1=next_point_coord[0], x2=next_point_coord[1], r=radius
                )
                scatter_g.set_offsets(next_point_coord)
                # if circle_patch is not None:
                # circle_patch.remove()
                circle_patch = plt.Circle(
                    next_point_coord, radius=radius, color="blue", fill=False, lw=2
                )
                # ax.add_patch(circle_patch)

                current_point = history_points[0, active_indices[-1] + 1, :].unsqueeze(
                    0
                )
                # current_point[:, 0] = times[0].item()
                likelihood, probability = likelihood_func(
                    current_point,
                    history_points[0, :, :].unsqueeze(0),
                    **adjust_kwargs_for_likelihood(1, kwargs_outputs),
                    compute_probability=True,
                    radius=radius,
                )

                cumulative_likelihood.append(math.exp(-likelihood.item()))
                cumulative_probability.append(probability)
                title = title + (
                    f"Likelihood of next saccade (density): {math.exp(-likelihood.item()):.2f}\n"
                    f"Average density: {sum(cumulative_likelihood)/len(cumulative_likelihood):.2f}\n"
                )

        # title += (
        # f"Alpha: {kwargs_outputs['temporal_kwargs']['alpha'].mean().item()}\n"
        # f"Beta: {kwargs_outputs['temporal_kwargs']['beta'].mean().item()}\n"
        # f"Sigma: {kwargs_outputs['spatial_kwargs']['sigma'].item()}\n"
        # )
        ## if "mu" in kwargs_outputs["baseline_kwargs"]:
        #    title += f"Mu: {kwargs_outputs['baseline_kwargs']['mu'].item()}\n"
        # if "gamma" in kwargs_outputs["baseline_kwargs"]:
        #    title += f"Gamma: {kwargs_outputs['baseline_kwargs']['gamma'].item()}\n"
        ax.set_title(title)

        # Return all updated artists.
        artists = contour.collections + [scatter, scatter_g]
        # for artist in line_contours:
        #   artists.append(artist)
        if circle_patch is not None:
            artists.append(circle_patch)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_values),
        blit=False,  # Blitting is off because we dynamically add/remove artists.
        repeat=False,
    )

    plt.close(
        fig
    )  # Prevents the static figure from displaying alongside the animation.
    return ani


def probability_in_circle(X, Y, Z, x1, x2, r):
    """
    Given:
      - X, Y: 2D coordinate arrays of shape (200, 200)
      - Z: 2D array of function (density) values, shape (200, 200)
      - (x1, x2): the center point
      - r: radius of the circle

    Returns:
      The ratio (sum of Z in the circle) / (sum of all Z).
      If Z is a probability density (up to a constant cell area), this ratio
      approximates the probability mass contained within the circle.
    """
    # Create boolean mask for points inside the circle
    mask = ((X - x1) ** 2 + (Y - x2) ** 2) <= r**2

    # Sum the values of Z in the circle and overall
    mass_inside = Z[mask].sum()
    total_mass = Z.sum()

    return mass_inside / total_mass


def to_numpy(arr):
    """Convert a torch tensor to a numpy array, or pass through if already a numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.array(arr)


def probability_in_circle(X, Y, Z, x1, x2, r):
    """
    Computes P(||(x,y) - (x1, x2)|| <= r) given a grid of coordinates (X, Y)
    and corresponding function/density values Z.

    If X, Y, or Z are PyTorch tensors, they are converted to NumPy arrays.

    Args:
        X       : 2D array of x-coordinates (shape: [200,200])
        Y       : 2D array of y-coordinates (shape: [200,200])
        Z       : 2D array of function values (shape: [200,200])
        x1, x2  : Coordinates of the center point
        r       : Radius of the circle

    Returns:
        The ratio (sum of Z inside the circle) / (sum of all Z), which approximates
        the probability mass inside the circle.
    """
    # Convert inputs to NumPy arrays if they aren't already
    X_np = to_numpy(X)
    Y_np = to_numpy(Y)
    Z_np = to_numpy(Z)

    # Create a boolean mask for points inside the circle
    mask = ((X_np - x1) ** 2 + (Y_np - x2) ** 2) <= r**2

    # Sum Z over the masked region and over the entire grid
    mass_inside = Z_np[mask].sum()
    total_mass = Z_np.sum()

    return mass_inside / total_mass


def adjust_kwargs_for_likelihood(sequence_length, kwargs_outputs):
    # deep copy a dictionary:
    kwargs_outputs = deepcopy(kwargs_outputs)
    if "boxes" in kwargs_outputs["baseline_kwargs"]:
        boxes = kwargs_outputs["baseline_kwargs"]["boxes"]
        kwargs_outputs["baseline_kwargs"]["boxes"] = boxes[:sequence_length, :, :]

    mean = kwargs_outputs["spatial_kwargs"]["mean"]
    kwargs_outputs["spatial_kwargs"]["mean"] = mean[:sequence_length, :, :]
    return kwargs_outputs
