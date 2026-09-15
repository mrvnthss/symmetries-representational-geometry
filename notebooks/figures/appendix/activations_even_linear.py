import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from collections.abc import Sequence
    from pathlib import Path

    import jax.numpy as jnp
    import matplotlib.axes as mpl_axes
    import matplotlib.pyplot as plt
    from figures.appendix.common import CURVE_COLOR
    from figures.appendix.common import DISPLAY_NAMES
    from figures.appendix.common import component_color
    from figures.appendix.common import draw_origin
    from figures.fonts.config import FIGURE_FONT_CONFIGS
    from figures.fonts.config import register_bundled_fonts
    from jaxtyping import Array

    from symmetries import activations

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    register_bundled_fonts()

    # Plotting defaults
    plt.style.use(["./style.mplstyle", "./frameless.mplstyle"])

    # Both components of a panel share one color beside its black curve
    COMPONENT_COLOR = component_color("orange")

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    FIGURE_LEGEND_LABELS = {
        "submission": (r"$f(x)$", r"$e(x)$", r"$o(x)$"),
        # MathText does not apply italic correction between custom italic and
        # roman fonts. Add the corrections needed by the Libertinus glyphs.
        "preprint": (
            r"$f\hspace{0.25}(x\hspace{0.12})$",
            r"$e\hspace{0.10}(x\hspace{0.12})$",
            r"$o\hspace{0.10}(x\hspace{0.12})$",
        ),
    }

    # Even-linear activation functions
    ACTIVATIONS = [
        "gelu",
        "hard_silu",
        ("leaky_relu", {"negative_slope": 0.05}),  # default: 0.01
        "log_sigmoid",
        "relu",
        "silu",
        "softplus",
        "sparse_plus",
        ("squareplus", {"b": 4}),  # default: 4
    ]


@app.function
def compute_components(
    spec: activations.ActivationSpec,
    xs: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate an activation function and its even and odd components."""
    fn = activations.get_activation(spec)
    ys = fn(xs)
    ys_neg = fn(-xs)
    ys_even = 0.5 * (ys + ys_neg)
    ys_odd = 0.5 * (ys - ys_neg)
    return ys, ys_even, ys_odd


@app.function
def plot_activation_components(
    ax: mpl_axes.Axes,
    spec: activations.ActivationSpec,
    xs: Array,
    z_orders: tuple[int, int, int],
    legend_labels: tuple[str, str, str],
) -> None:
    """Plot an activation function and its even and odd components."""
    z_fn, z_even, z_odd = z_orders
    ys, ys_even, ys_odd = compute_components(spec, xs)

    ax.plot(
        xs,
        ys,
        label=legend_labels[0],
        color=CURVE_COLOR,
        lw=2,  # default: 1.5
        zorder=z_fn,
    )
    ax.plot(
        xs,
        ys_even,
        label=legend_labels[1],
        color=COMPONENT_COLOR,
        ls="--",
        zorder=z_even,
    )
    ax.plot(
        xs,
        ys_odd,
        label=legend_labels[2],
        color=COMPONENT_COLOR,
        ls=":",
        zorder=z_odd,
    )
    name = spec if isinstance(spec, str) else spec[0]
    ax.set_title(DISPLAY_NAMES[name])


@app.function
def plot_activation_components_grid(
    activation_specs: Sequence[activations.ActivationSpec],
    *,
    fig_size: tuple[float, float],
    n_rows: int,
    n_cols: int,
    y_range: float,
    x_to_y_ratio: float,
    n_points: int,
    z_orders: tuple[int, int, int],
    legend_labels: tuple[str, str, str],
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a grid of activations with their even/odd components."""
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=True,
        figsize=fig_size,
    )

    # Set axes limits once for all subplots
    x_range = x_to_y_ratio * y_range
    axs.flat[0].set_xlim(-x_range, x_range)
    axs.flat[0].set_ylim(-y_range, y_range)

    # Precompute xs to use for plotting (+1 to cover the entire plot area)
    xs = jnp.linspace(-(x_range + 1.0), x_range + 1.0, num=n_points)

    # Iterate over activation functions
    for ax, spec in zip(axs.flat, activation_specs, strict=True):
        plot_activation_components(
            ax=ax,
            spec=spec,
            xs=xs,
            z_orders=z_orders,
            legend_labels=legend_labels,
        )
        ax.set_aspect("equal")
        draw_origin(ax)

    # Add global legend (use the first axis as the source of handles/labels)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.4),
        bbox_transform=axs[n_rows - 1, n_cols // 2].transAxes,
    )

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.function
def export_activation_components_grid_variants(
    activation_specs: Sequence[activations.ActivationSpec],
    *,
    output_dir: Path,
    output_stem: str,
    fig_size: tuple[float, float],
    n_rows: int,
    n_cols: int,
    y_range: float,
    x_to_y_ratio: float,
    n_points: int,
    z_orders: tuple[int, int, int],
) -> None:
    """Export submission and preprint variants with matched fonts."""
    for variant, font_config in FIGURE_FONT_CONFIGS.items():
        variant_output_dir = output_dir / variant
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = variant_output_dir / f"{output_stem}.pdf"
        with plt.rc_context(font_config):
            fig = plot_activation_components_grid(
                activation_specs,
                fig_size=fig_size,
                n_rows=n_rows,
                n_cols=n_cols,
                y_range=y_range,
                x_to_y_ratio=x_to_y_ratio,
                n_points=n_points,
                z_orders=z_orders,
                legend_labels=FIGURE_LEGEND_LABELS[variant],
                output_path=output_path,
            )
        plt.close(fig)


@app.cell
def _():
    export_activation_components_grid_variants(
        ACTIVATIONS,
        output_dir=OUTPUT_DIR,
        output_stem="activations-even-linear",
        fig_size=(8.0, 6.15),
        n_rows=3,
        n_cols=3,
        y_range=5.0,
        x_to_y_ratio=1.4,
        n_points=1001,
        z_orders=(3, 1, 2),
    )
    return


if __name__ == "__main__":
    app.run()
