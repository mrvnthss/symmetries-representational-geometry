import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import jax.numpy as jnp
    import matplotlib.axes as mpl_axes
    import matplotlib.pyplot as plt
    from figures.appendix.common import CURVE_COLOR
    from figures.appendix.common import component_color
    from figures.fonts.config import FIGURE_FONT_CONFIGS
    from figures.fonts.config import register_bundled_fonts
    from jaxtyping import Array

    from symmetries import activations

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    register_bundled_fonts()

    # Plotting defaults
    plt.style.use(["./style.mplstyle", "./frameless.mplstyle"])

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STEM = "opposite-orientations"

    RELU = activations.get_activation("relu")

    # The width-3 parameterization theta^+ as (w, b, a) triples. Its
    # neurons have kinks at -1, 0, and 1, and together realize the tent
    # function psi(1 - |x|).
    NEURONS = (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, -2.0),
        (1.0, -1.0, 1.0),
    )

    # The three panels. "pos" and "neg" decompose the oppositely oriented
    # parameterizations theta^+ and theta^-; "lin" holds the net
    # contributions of the linear-neuron groups connecting the two.
    PANELS = ("pos", "neg", "lin")

    TITLES = {
        "pos": r"$\theta^{+}$",
        "neg": r"$\theta^{-}$",
        "lin": r"$\theta^{+} \to \theta^{-}$",
    }

    # Hue marks orientation, following the convention of Figure 1: orange
    # for +(w, b), teal for -(w, b), grey for the mixed-sign pairs making
    # up a linear-neuron group.
    HUES = {"pos": "orange", "neg": "teal", "lin": "grey"}

    # Per-neuron terms, in the order of NEURONS, using the z_j shorthand
    # of the example (z_1 = x+1, z_2 = x, z_3 = x-1). MathText does not
    # apply italic correction between custom italic and roman fonts. Add
    # the corrections needed by the Libertinus glyphs.
    TERM_LABELS = {
        "submission": {
            "pos": (r"$\psi(z_{1})$", r"$-2\,\psi(z_{2})$", r"$\psi(z_{3})$"),
            "neg": (
                r"$\psi(-z_{1})$",
                r"$-2\,\psi(-z_{2})$",
                r"$\psi(-z_{3})$",
            ),
            "lin": (r"$-z_{1}$", r"$2z_{2}$", r"$-z_{3}$"),
        },
        "preprint": {
            "pos": (
                r"$\psi\hspace{0.25}(z_{1}\hspace{0.12})$",
                r"$-2\,\psi\hspace{0.25}(z_{2}\hspace{0.12})$",
                r"$\psi\hspace{0.25}(z_{3}\hspace{0.12})$",
            ),
            "neg": (
                r"$\psi\hspace{0.25}(-z_{1}\hspace{0.12})$",
                r"$-2\,\psi\hspace{0.25}(-z_{2}\hspace{0.12})$",
                r"$\psi\hspace{0.25}(-z_{3}\hspace{0.12})$",
            ),
            "lin": (r"$-z_{1}$", r"$2z_{2}$", r"$-z_{3}$"),
        },
    }

    # Labeled in place, so the legend stays a single row of three terms.
    ANNOTATIONS = {
        "submission": {
            "pos": r"$\psi(1-|x|)$",
            "neg": r"$\psi(1-|x|)$",
            "lin": r"$\Sigma \equiv 0$",
        },
        "preprint": {
            "pos": r"$\psi\hspace{0.25}(1-|x\hspace{0.12}|)$",
            "neg": r"$\psi\hspace{0.25}(1-|x\hspace{0.12}|)$",
            "lin": r"$\Sigma \equiv 0$",
        },
    }

    # (x, y, horizontal alignment, vertical alignment) per panel, in the
    # space each one leaves clear beside its black curve.
    ANNOTATION_POS = {
        "pos": (-1.1, 0.6, "right", "center"),
        "neg": (1.1, 0.6, "left", "center"),
        "lin": (2.1, 0.18, "right", "bottom"),
    }

    # One color per panel; line style separates the three terms, as it
    # separates f / e / o in the activation figures.
    TERM_LINESTYLES = ("-", "--", ":")


@app.function
def compute_terms(panel: str, xs: Array) -> tuple[Array, Array]:
    """Evaluate the per-neuron terms of a panel and their sum.

    Args:
        panel: One of ``"pos"`` (the neurons of theta^+), ``"neg"`` (the
          neurons of theta^-), or ``"lin"`` (the net contributions of the
          three linear-neuron groups).
        xs: Input values at which to evaluate the terms.

    Returns:
        A ``(terms, total)`` pair, where ``terms`` stacks the three
        per-neuron terms row-wise and ``total`` is their sum.
    """
    rows = []
    for w, b, a in NEURONS:
        z = w * xs + b
        if panel == "pos":
            rows.append(a * RELU(z))
        elif panel == "neg":
            rows.append(a * RELU(-z))
        else:
            rows.append(-a * z)
    terms = jnp.stack(rows)
    return terms, terms.sum(axis=0)


@app.function
def plot_panel(
    ax: mpl_axes.Axes,
    panel: str,
    xs: Array,
    z_orders: tuple[int, int],
    term_labels: tuple[str, str, str],
    annotation: str,
    annotation_pos: tuple[float, float, str, str],
) -> None:
    """Plot the per-neuron terms of one panel together with their sum."""
    z_terms, z_total = z_orders
    terms, total = compute_terms(panel, xs)

    term_color = component_color(HUES[panel])

    styled = zip(terms, term_labels, TERM_LINESTYLES, strict=True)
    for i, (ys, label, linestyle) in enumerate(styled):
        ax.plot(
            xs,
            ys,
            label=label,
            color=term_color,
            ls=linestyle,
            zorder=z_terms + i,
        )
    ax.plot(
        xs,
        total,
        color=CURVE_COLOR,
        lw=2,  # default: 1.5
        zorder=z_total,
    )

    ann_x, ann_y, ann_ha, ann_va = annotation_pos
    ax.text(ann_x, ann_y, annotation, ha=ann_ha, va=ann_va, zorder=z_total)

    ax.set_title(TITLES[panel])


@app.function
def plot_opposite_orientations(
    *,
    fig_size: tuple[float, float],
    x_range: float,
    y_range: float,
    n_points: int,
    z_orders: tuple[int, int],
    term_labels: dict[str, tuple[str, str, str]],
    annotations: dict[str, str],
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the two oppositely oriented ReLU parameterizations.

    The first two panels decompose theta^+ and theta^- into their neuron
    terms, both summing to the same tent function. The third panel holds
    the net contributions of the three linear-neuron groups whose joint
    addition connects the two parameterizations; these cancel
    collectively without cancelling individually.

    The panels are not equal-aspect: the terms reach a slope of 2, which
    would leave them twice as tall as they are wide.

    Args:
        fig_size: Figure size in inches.
        x_range: Half-width of the shared x-axis limits.
        y_range: Half-height of the shared y-axis limits.
        n_points: Number of points at which to evaluate the terms.
        z_orders: Base z-order of the per-neuron terms and z-order of
          their sum.
        term_labels: Legend labels for the per-neuron terms, by panel.
        annotations: In-plot label for the sum, by panel.
        output_path: If given, the path to save the figure to.
        show: Whether to display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.
    """
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(PANELS),
        sharex=True,
        sharey=True,
        figsize=fig_size,
    )

    # Set axes limits once for all subplots
    axs[0].set_xlim(-x_range, x_range)
    axs[0].set_ylim(-y_range, y_range)

    # Precompute xs to use for plotting (+1 to cover the entire plot area)
    xs = jnp.linspace(-(x_range + 1.0), x_range + 1.0, num=n_points)

    for ax, panel in zip(axs, PANELS, strict=True):
        plot_panel(
            ax,
            panel,
            xs,
            z_orders,
            term_labels[panel],
            annotations[panel],
            ANNOTATION_POS[panel],
        )

        # Terms differ per panel, so each carries its own legend. No
        # corner is free in all three, so they sit below the axes.
        ax.legend(
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
            bbox_transform=ax.transAxes,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.2,
        )

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.function
def export_opposite_orientations_variants(
    *,
    output_dir: Path,
    output_stem: str,
    fig_size: tuple[float, float],
    x_range: float,
    y_range: float,
    n_points: int,
    z_orders: tuple[int, int],
) -> None:
    """Export submission and preprint variants with matched fonts."""
    for variant, font_config in FIGURE_FONT_CONFIGS.items():
        variant_output_dir = output_dir / variant
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = variant_output_dir / f"{output_stem}.pdf"
        with plt.rc_context(font_config):
            fig = plot_opposite_orientations(
                fig_size=fig_size,
                x_range=x_range,
                y_range=y_range,
                n_points=n_points,
                z_orders=z_orders,
                term_labels=TERM_LABELS[variant],
                annotations=ANNOTATIONS[variant],
                output_path=output_path,
            )
        plt.close(fig)


@app.cell
def _():
    export_opposite_orientations_variants(
        output_dir=OUTPUT_DIR,
        output_stem=OUTPUT_STEM,
        fig_size=(8.0, 3.1),
        x_range=2.2,
        y_range=2.4,
        n_points=1001,
        z_orders=(1, 4),
    )
    return


if __name__ == "__main__":
    app.run()
