import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import jax.numpy as jnp
    import matplotlib.axes as mpl_axes
    import matplotlib.pyplot as plt
    from figures.appendix.xor_solutions import align_colorbar
    from figures.appendix.xor_solutions import seal_colorbar
    from figures.fonts.config import FIGURE_FONT_CONFIG
    from figures.fonts.config import register_bundled_fonts
    from figures.layout import scale_pdf_to_text_width
    from jaxtyping import Array
    from matplotlib import colors as mpl_colors

    from figures.main.dissociation.components import RSM_CMAP
    from symmetries import activations
    from symmetries import distances
    from symmetries import plotting
    from symmetries import rsa

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    register_bundled_fonts()

    # Importing the Figure 1 components changes a few rcParams for their
    # SVG exports, so start from the Matplotlib defaults.
    plt.style.use(["default", "./style.mplstyle"])

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STEM = "nonunique-mwnp-geometry"

    RELU = activations.get_activation("relu")

    # The input set. Its values label both axes of every RSM.
    INPUTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
    INPUT_LABELS = tuple(rf"${x:.0f}$" for x in INPUTS)

    # The tent-function parameterization theta^+ as (w, b, a) triples,
    # as in ``opposite_orientations.py``. Each neuron is the only member
    # of its essential class, so the class's unit-norm representative is
    # the direction of (w, b), and its orbit invariant beta_q is the
    # readout a scaled by the norm of (w, b).
    NEURONS = (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, -2.0),
        (1.0, -1.0, 1.0),
    )
    _params = jnp.array(NEURONS)
    _norms = jnp.linalg.norm(_params[:, :2], axis=1)
    ESSENTIAL_DIRECTIONS = _params[:, :2] / _norms[:, None]
    ORBIT_INVARIANTS = _params[:, 2] * _norms

    # Orientation splits t of the panels. The endpoints are the two RSMs
    # available for 3 <= N_h < 6; the interior values need all six
    # neurons.
    SPLITS = (0.0, 0.25, 0.5, 0.75, 1.0)

    TITLES = (
        r"$t = 0$",
        r"$t = \frac{1}{4}$",
        r"$t = \frac{1}{2}$",
        r"$t = \frac{3}{4}$",
        r"$t = 1$",
    )

    COLORBAR_LABEL = r"$[\mathbf{M}(t)]_{\mu\nu}$"


@app.function
def oriented_features() -> tuple[Array, Array]:
    """Evaluate both orientations of the essential ReLU features.

    Returns:
        A ``(positive, negative)`` pair of ``(P, 3)`` arrays holding the
        features f_q^+ = psi(w_q^T x) and f_q^- = psi(-w_q^T x) column
        by column, for the unit-norm extended incoming parameters
        w_q = (w, b) and the extended inputs x = (x, 1).
    """
    xs = jnp.array(INPUTS)
    extended = jnp.stack([xs, jnp.ones_like(xs)], axis=1)
    preactivations = extended @ ESSENTIAL_DIRECTIONS.T
    return RELU(preactivations), RELU(-preactivations)


@app.function
def mwnp_rsm(t: float) -> Array:
    """Return the RSM M(t) of the minimum-weight-norm parameterizations.

    M(t) = t M_+ + (1 - t) M_- is the dot-product RSM of the hidden
    activations of the width-6 parameterization that gives each
    essential class a neuron of squared gain t |beta_q| on its positive
    orientation and one of squared gain (1 - t) |beta_q| on its negative
    orientation.

    Args:
        t: Orientation split, shared by the three essential classes.

    Returns:
        The ``(P, P)`` matrix M(t).
    """
    positive, negative = oriented_features()
    weights = jnp.abs(ORBIT_INVARIANTS)
    hidden = jnp.concatenate(
        [
            jnp.sqrt(t * weights) * positive,
            jnp.sqrt((1.0 - t) * weights) * negative,
        ],
        axis=1,
    )
    return rsa.calculate_rdm(hidden, distances.dot_product, return_matrix=True)


@app.function
def plot_rsm_panel(
    ax: mpl_axes.Axes,
    rsm: Array,
    *,
    title: str,
    vmax: float,
    edge_lw: float,
    label_pad: float,
) -> None:
    """Draw one RSM as a lower triangle, with the inputs as tick labels.

    Args:
        ax: Axes to draw the panel on.
        rsm: The ``(P, P)`` matrix to draw.
        title: Panel title, naming the split.
        vmax: Upper end of the color range shared by all panels.
        edge_lw: Line width of the cell edges.
        label_pad: Gap between the cells and the tick labels, in points.
    """
    plotting.plot_rdm(
        rsm,
        tri="lower",
        imshow_kwargs={"cmap": RSM_CMAP, "vmin": 0.0, "vmax": vmax},
        edge_kwargs={"linewidth": edge_lw},
        ax=ax,
    )

    # The RSMs of Figures 1 and 2 mark their axes with the data markers.
    # Here the inputs are scalars, so their values stand in, without
    # tick marks. Without marks the labels sit at the tick pad alone,
    # so it is widened to keep them off the cell edges.
    positions = range(len(INPUTS))
    ax.set_xticks(positions, labels=INPUT_LABELS)
    ax.set_yticks(positions, labels=INPUT_LABELS)
    ax.tick_params(length=0, pad=label_pad)
    ax.set_title(title)


@app.function
def plot_nonunique_mwnp_geometry(
    *,
    fig_size: tuple[float, float],
    panel_spacing: float,
    edge_lw: float,
    label_pad: float,
    titles: tuple[str, ...],
    colorbar_label: str,
    colorbar_kwargs: dict,
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the one-parameter family of minimum-weight-norm RSMs.

    One panel per orientation split t, all on one color scale. The
    endpoints M_- (t = 0) and M_+ (t = 1) are mirror images of each
    other, since reversing the inputs swaps the two orientations of
    every essential class. The interior panels interpolate linearly
    between them, so no two panels agree.

    Args:
        fig_size: Figure size in inches.
        panel_spacing: Gap between panels, as a fraction of their width.
        edge_lw: Line width of the cell edges.
        label_pad: Gap between the cells and the tick labels, in points.
        titles: Panel titles, one per split.
        colorbar_label: Label of the shared colorbar.
        colorbar_kwargs: Arguments passed to ``fig.colorbar``.
        output_path: If given, the path to save the figure to.
        show: Whether to display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.
    """
    rsms = [mwnp_rsm(t) for t in SPLITS]

    # M(t) is linear in t, so the range is set by the endpoints.
    vmax = max(float(rsm.max()) for rsm in rsms)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(SPLITS),
        sharex=True,
        sharey=True,
        figsize=fig_size,
        squeeze=False,
    )

    # Only the first panel carries row labels, so the others would sit
    # at the style's default gap, too close to read as separate RSMs.
    fig.get_layout_engine().set(wspace=panel_spacing)

    for ax, rsm, title in zip(axs[0], rsms, titles, strict=True):
        plot_rsm_panel(
            ax,
            rsm,
            title=title,
            vmax=vmax,
            edge_lw=edge_lw,
            label_pad=label_pad,
        )

    # One colorbar for all five panels, which share the color range.
    mappable = plt.cm.ScalarMappable(
        norm=mpl_colors.Normalize(vmin=0.0, vmax=vmax),
        cmap=RSM_CMAP,
    )
    cbar = fig.colorbar(mappable, ax=axs, **colorbar_kwargs)
    cbar.set_label(colorbar_label)
    seal_colorbar(cbar)

    align_colorbar(fig, cbar, axs)

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.function
def export_nonunique_mwnp_geometry(
    *,
    output_dir: Path,
    output_stem: str,
    fig_size: tuple[float, float],
    panel_spacing: float,
    edge_lw: float,
    label_pad: float,
    colorbar_kwargs: dict,
    width_fraction: float,
) -> None:
    """Export the figure with the paper's fonts, at its width in the paper.

    Args:
        output_dir: Directory to write the PDF to.
        output_stem: File name of the PDF, without extension.
        fig_size: Figure size in inches, at which the figure is designed.
        panel_spacing: Gap between panels, as a fraction of their width.
        edge_lw: Line width of the cell edges.
        label_pad: Gap between the cells and the tick labels, in points.
        colorbar_kwargs: Arguments passed to ``fig.colorbar``.
        width_fraction: Width of the figure in the paper, as a fraction
          of the text width.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.pdf"
    with plt.rc_context(FIGURE_FONT_CONFIG):
        fig = plot_nonunique_mwnp_geometry(
            fig_size=fig_size,
            panel_spacing=panel_spacing,
            edge_lw=edge_lw,
            label_pad=label_pad,
            titles=TITLES,
            colorbar_label=COLORBAR_LABEL,
            colorbar_kwargs=colorbar_kwargs,
            output_path=output_path,
        )
    plt.close(fig)
    scale_pdf_to_text_width(output_path, width_fraction)


@app.cell
def _():
    export_nonunique_mwnp_geometry(
        output_dir=OUTPUT_DIR,
        output_stem=OUTPUT_STEM,
        fig_size=(8.0, 2.0),
        panel_spacing=0.2,
        edge_lw=0.6,
        label_pad=6.0,  # default: 3.5
        # ``align_colorbar`` releases the bar's box aspect, so its width
        # is the fraction of the panels' width reserved for it.
        colorbar_kwargs={
            "shrink": 0.9,
            "fraction": 0.018,
            "pad": 0.02,
        },
        width_fraction=0.95,  # \includegraphics[width=0.95\textwidth]
    )
    return


if __name__ == "__main__":
    app.run()
