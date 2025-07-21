import numpy as np
import matplotlib.pyplot as plt


def create_box_plot_comparison(
    loss_vector, subsets_matrix, subsets_names, filename="box_plot_comparison.png"
):
    B, K = subsets_matrix.shape

    # Collect loss values and count observations for each subset
    box_data = []
    counts = []
    for i in range(K):
        subset_losses = loss_vector[subsets_matrix[:, i] == 1]
        box_data.append(subset_losses)
        counts.append(np.sum(subsets_matrix[:, i] == 1))

    # Add the overall loss vector as the last entry
    box_data.append(loss_vector)
    counts.append(len(loss_vector))
    subsets_names.append("Overall")

    # Create new labels that include the observation counts
    labels_with_counts = [
        f"{name}\n(n = {cnt})" for name, cnt in zip(subsets_names, counts)
    ]

    # Set a style that is often acceptable in academic publications
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Customize box plot properties for a clean appearance
    bp = plt.boxplot(
        box_data,
        labels=labels_with_counts,
        patch_artist=True,
        medianprops=dict(color="firebrick", linewidth=2),
        boxprops=dict(facecolor="lightblue", color="blue", linewidth=2),
        whiskerprops=dict(color="blue", linewidth=2),
        capprops=dict(color="blue", linewidth=2),
        showfliers=False,  # Do not display outliers
    )

    # Format axes and labels
    plt.xticks(rotation=45, fontsize=12)
    plt.xlabel("Subsets", fontsize=14)
    plt.ylabel("Loss Values", fontsize=14)
    plt.title("Box Plot Comparison of Loss Across Subsets", fontsize=16)
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # Annotate each box with the mean (μ) and standard deviation (σ)
    ax = plt.gca()
    for i, data in enumerate(box_data):
        # Skip if no data is present
        if len(data) == 0:
            continue
        mean_val = np.mean(data)
        std_val = np.std(data)

        # Compute the third quartile to help position the annotation
        q3 = np.percentile(data, 75)
        q1 = np.percentile(data, 25)
        # Define an offset. If the data have no spread (q3 == q1), use a small constant offset.
        offset = (q3 - q1) * 0.1 if q3 != q1 else 0.05 * q3
        # Place the text just above the upper quartile.
        ax.text(
            i + 1,
            q3 + offset,
            f"mean = {round(float(mean_val), 2)}\n std = {round(float(std_val),2)} \n {round(float(np.percentile(data, 50)),2)}",
            horizontalalignment="center",
            fontsize=10,
            color="black",
        )

    # Save the figure with high resolution suitable for publication
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Tuple


def plot_llr_violins(
    data: Sequence[np.ndarray],
    *,
    labels: Sequence[str] | None = None,
    title: str | None = ("Bootstrapped Log-Likelihood Ratio\n" "from Poisson Baseline"),
    y_label: str = "Log Likelihood Ratio (higher is better)",
    save_path: str | Path | None = None,
    step=0.25,
    start=0.0,
    dpi=300,
    fig_size=(6, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw violin plots styled like the reference image.

    Parameters
    ----------
    data
        A sequence of 1-D NumPy arrays – one per model/condition.
    labels
        Tick labels; must match `len(data)`.  If *None*, defaults to
        ["LF\nBaseline", "SH\nBaseline", "CSS", "CSS + RME"] when there are
        four datasets, otherwise to simple integers.
    title
        Multi-line figure title (omit to suppress).
    y_label
        Y-axis label text.
    save_path
        If provided, write the figure to this file (format from suffix).

    Returns
    -------
    (fig, ax)
        Matplotlib handles for further tweaking.
    """
    n = len(data)
    if n == 0:
        raise ValueError("`data` must contain at least one array.")

    # sensible label defaults
    if labels is None:
        if n == 4:
            labels = ["LF\nBaseline", "SH\nBaseline", "CSS", "CSS + RME"]
        else:
            labels = [str(i + 1) for i in range(n)]
    if len(labels) != n:
        raise ValueError("`labels` length must match `data` length.")

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # ---------------- violins ---------------- #
    positions = np.arange(1, n + 1)
    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("lightgray")
        body.set_edgecolor("black")
        body.set_linewidth(1.5)

    # ---------------- mean bars & 95 % CI ---------------- #
    cap_halfwidth = 0.12  # width of little end-caps
    bar_halfwidth = 0.18  # width of red mean bar

    for pos, sample in zip(positions, data):
        mean = np.mean(sample)
        ci_low, ci_high = np.percentile(sample, [2.5, 97.5])

        # 95 % vertical whisker + caps
        ax.vlines(pos, ci_low, ci_high, color="black", linewidth=2)
        ax.hlines(
            [ci_low, ci_high],
            pos - cap_halfwidth,
            pos + cap_halfwidth,
            color="black",
            linewidth=2,
        )

        # red mean bar
        ax.hlines(
            mean, pos - bar_halfwidth, pos + bar_halfwidth, color="red", linewidth=3
        )

    # ---------------- cosmetics ---------------- #
    ax.set_xticks(positions)

    # --- NEW: angled tick labels so long names don’t overlap
    rotation = 30  # ≤ 30° usually keeps things readable
    ax.set_xticklabels(
        labels,
        fontsize=12,
        ha="right",
        rotation=rotation,
        rotation_mode="anchor",  # anchor at the text’s right edge
    )

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylabel(y_label, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, weight="bold", pad=12)

    # y-limits and dashed grid — choose nice round numbers
    y_min, y_max = min(map(np.min, data)), max(map(np.max, data))
    span = y_max - y_min
    padding = 0.05 * span
    ax.set_ylim(y_min - padding, y_max + padding)

    # dashed grid every 0.25 (adjust if that’s too dense or sparse)
    ax.set_yticks(np.arange(start, ax.get_ylim()[1] + step, step))
    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.5)
    ax.set_axisbelow(True)

    # faint vertical guidelines behind each violin
    for x in positions:
        ax.axvline(x, color="gray", alpha=0.2)

    fig.tight_layout()
    if save_path is not None:

        fig.savefig(save_path)
        print(f"Figure saved to {Path(save_path).resolve()}")

    return fig, ax
