import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import matplotlib.axes as mpl_axes
    import matplotlib.pyplot as plt
    import numpy as np
    from experiments.sweeps import relu_sweep_models
    from figures.fonts.config import FIGURE_FONT_CONFIGS
    from figures.fonts.config import register_bundled_fonts
    from matplotlib.transforms import ScaledTranslation

    from symmetries import clustering
    from symmetries import colors

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    register_bundled_fonts()

    plt.style.use("./style.mplstyle")

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STEM = "xor-sweep-geometry"

    # The values the solution table predicts; the panels tick there.
    PREDICTED_DISTANCES = (-np.sqrt(2.0), 0.0)
    DISTANCE_TICK_LABELS = (r"$-\sqrt{2}$", r"$0$")
    PREDICTED_ANGLES = tuple(k * np.pi / 4 for k in (1, 3, 5, 7))
    ANGLE_TICK_LABELS = (
        r"$\pi/4$",
        r"$3\pi/4$",
        r"$5\pi/4$",
        r"$7\pi/4$",
    )

    PANEL_TITLES = {
        "distance": r"Signed distance $d_{j}$",
        "angle": r"Boundary angle $\phi_{j}$",
    }
    COUNT_LABEL = "Neurons"

    BAR_COLOR = colors.get_color("grey", "L3", "medium")

    TITLE_PAD = 12.0  # points between a panel and its title

    # Nudge of the radial tick labels, in points, off the rings they
    # label and away from the petals.
    R_LABEL_OFFSET = (-4.0, 3.0)


@app.function
def sweep_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return the angle and signed distance of every converged neuron.

    Returns:
        The boundary angles and the signed distances, one entry each
        per hidden neuron.
    """
    models, converged = relu_sweep_models()
    kept = [m for m, ok in zip(models, converged, strict=True) if ok]
    angles, distances, *_ = clustering.standardize_mlp_solutions(kept)
    return np.asarray(angles).ravel(), np.asarray(distances).ravel()


@app.function
def plot_distances(
    ax: mpl_axes.Axes,
    distances: np.ndarray,
    *,
    bin_width: float,
    x_range: tuple[float, float],
) -> None:
    """Draw the distribution of signed distances.

    Args:
        ax: Axes to draw on.
        distances: Signed distance of every neuron.
        bin_width: Approximate histogram bin width. It is snapped so
          that both predicted distances land on bin centers.
        x_range: Axis limits.
    """
    span = PREDICTED_DISTANCES[1] - PREDICTED_DISTANCES[0]
    width = span / round(span / bin_width)
    low = np.floor(x_range[0] / width + 0.5) * width - width / 2
    edges = np.arange(low, x_range[1] + width, width)
    ax.hist(distances, bins=edges, color=BAR_COLOR, linewidth=0)

    ax.set_xlim(*x_range)
    ax.set_xticks(PREDICTED_DISTANCES, labels=DISTANCE_TICK_LABELS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(PANEL_TITLES["distance"], pad=TITLE_PAD)


@app.function
def plot_angles(
    ax: mpl_axes.Axes,
    angles: np.ndarray,
    *,
    n_bins: int,
    r_max: float,
    r_ticks: tuple[int, ...],
) -> None:
    """Draw the distribution of boundary angles around the turn.

    Polar: the angles are directions in the input space, and the four
    the sweep finds form two diameters, one per solution family.

    Args:
        ax: Polar axes to draw on.
        angles: Boundary angle of every neuron.
        n_bins: Number of bins over the turn. The grid is aligned so
          that zero is a bin center.
        r_max: Outer limit of the count axis.
        r_ticks: Ticks of the count axis.
    """
    half = np.pi / n_bins
    turn = 2 * np.pi
    wrapped = (angles + half) % turn - half
    ax.hist(
        wrapped,
        bins=np.linspace(-half, turn - half, n_bins + 1),
        color=BAR_COLOR,
        linewidth=0,
    )

    ax.set_xticks(PREDICTED_ANGLES, labels=ANGLE_TICK_LABELS)
    ax.set_ylim(0.0, r_max)
    ax.set_yticks(r_ticks)
    ax.set_rlabel_position(270.0)
    nudge = ScaledTranslation(
        R_LABEL_OFFSET[0] / 72,
        R_LABEL_OFFSET[1] / 72,
        ax.get_figure().dpi_scale_trans,
    )
    for label in ax.get_yticklabels():
        label.set_transform(label.get_transform() + nudge)
    ax.grid(visible=True, linewidth=0.4, alpha=0.5)
    ax.set_title(PANEL_TITLES["angle"], pad=TITLE_PAD)


@app.function
def plot_sweep_geometry(
    *,
    fig_size: tuple[float, float],
    distance_bin_width: float,
    distance_range: tuple[float, float],
    n_angle_bins: int,
    count_limits: tuple[float, float],
    count_ticks: tuple[int, ...],
    angle_count_max: float,
    angle_count_ticks: tuple[int, ...],
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the geometry of every neuron the sweep converged on.

    Each converged network contributes its two hidden neurons. The
    distance panel counts on a logarithmic axis, which keeps the few
    neurons away from the predicted values visible.

    Args:
        fig_size: Figure size in inches.
        distance_bin_width: Bin width of the distance panel.
        distance_range: Axis limits of the distance panel.
        n_angle_bins: Number of bins over one turn of the angle panel.
        count_limits: Limits of the distance panel's count axis.
        count_ticks: Ticks of the distance panel's count axis.
        angle_count_max: Outer limit of the angle panel's count axis.
        angle_count_ticks: Ticks of the angle panel's count axis.
        output_path: If given, the path to save the figure to.
        show: Whether to display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.
    """
    angles, distances = sweep_geometry()

    fig = plt.figure(figsize=fig_size)
    ax_distance = fig.add_subplot(1, 2, 1)
    ax_angle = fig.add_subplot(1, 2, 2, projection="polar")

    plot_distances(
        ax_distance,
        distances,
        bin_width=distance_bin_width,
        x_range=distance_range,
    )
    plot_angles(
        ax_angle,
        angles,
        n_bins=n_angle_bins,
        r_max=angle_count_max,
        r_ticks=angle_count_ticks,
    )

    ax_distance.set_yscale("log")
    ax_distance.set_ylim(*count_limits)
    ax_distance.set_yticks(count_ticks, labels=[f"{t:d}" for t in count_ticks])
    ax_distance.set_ylabel(COUNT_LABEL)

    # A polar axes is square; squaring this one keeps both titles at
    # the same height.
    ax_distance.set_box_aspect(1.0)

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.function
def export_sweep_geometry_variants(
    *,
    output_dir: Path,
    output_stem: str,
    fig_size: tuple[float, float],
    distance_bin_width: float,
    distance_range: tuple[float, float],
    n_angle_bins: int,
    count_limits: tuple[float, float],
    count_ticks: tuple[int, ...],
    angle_count_max: float,
    angle_count_ticks: tuple[int, ...],
) -> None:
    """Export submission and preprint variants with matched fonts."""
    for variant, font_config in FIGURE_FONT_CONFIGS.items():
        variant_output_dir = output_dir / variant
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = variant_output_dir / f"{output_stem}.pdf"
        with plt.rc_context(font_config):
            fig = plot_sweep_geometry(
                fig_size=fig_size,
                distance_bin_width=distance_bin_width,
                distance_range=distance_range,
                n_angle_bins=n_angle_bins,
                count_limits=count_limits,
                count_ticks=count_ticks,
                angle_count_max=angle_count_max,
                angle_count_ticks=angle_count_ticks,
                output_path=output_path,
            )
        plt.close(fig)


@app.cell
def _():
    export_sweep_geometry_variants(
        output_dir=OUTPUT_DIR,
        output_stem=OUTPUT_STEM,
        fig_size=(5.8, 3.1),
        distance_bin_width=0.02,
        distance_range=(-1.55, 0.55),
        n_angle_bins=72,
        count_limits=(0.7, 1000.0),
        count_ticks=(1, 10, 100, 1000),
        angle_count_max=180.0,
        angle_count_ticks=(50, 100, 150),
    )
    return


if __name__ == "__main__":
    app.run()
