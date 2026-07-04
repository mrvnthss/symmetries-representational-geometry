import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from collections.abc import Sequence
    from pathlib import Path

    import cycler
    import jax.numpy as jnp
    import matplotlib.axes as mpl_axes
    import matplotlib.lines as mpl_lines
    import matplotlib.pyplot as plt
    from jaxtyping import Array

    from symmetries import activations
    from symmetries import colors

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    # Plotting defaults
    plt.style.use("./style.mplstyle")

    # Sample orange sequential colormap at discrete values
    cmap = colors.get_sequential_cmap("orange")
    orange_colors = [cmap(t) for t in [0.9, 0.6, 0.4]]
    plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=orange_colors)

    OUTPUT_PATH = Path("../figures/appendix/fig_a1.pdf")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
def setup_grid(ax: mpl_axes.Axes) -> None:
    """Apply custom grid settings to an axis."""
    ax.grid(
        visible=True,
        which="major",
        linewidth=0.60,  # matches tick width
        alpha=0.6,
    )
    ax.grid(
        visible=True,
        which="minor",
        linewidth=0.45,  # matches tick width
        alpha=0.3,
    )
    ax.minorticks_on()
    ax.set_axisbelow(True)


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
) -> tuple[mpl_lines.Line2D, mpl_lines.Line2D, mpl_lines.Line2D]:
    """Plot an activation function and its even and odd components."""
    z_fn, z_even, z_odd = z_orders
    ys, ys_even, ys_odd = compute_components(spec, xs)

    (line_y,) = ax.plot(
        xs,
        ys,
        label=r"$f(x)$",
        lw=2,  # default: 1.5
        zorder=z_fn,
    )
    (line_y_even,) = ax.plot(
        xs,
        ys_even,
        label=r"$e(x)$",
        ls="--",
        zorder=z_even,
    )
    (line_y_odd,) = ax.plot(
        xs,
        ys_odd,
        label=r"$o(x)$",
        ls=":",
        zorder=z_odd,
    )
    name = spec if isinstance(spec, str) else spec[0]
    ax.set_title(name)

    return line_y, line_y_even, line_y_odd


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
    for i, (ax, spec) in enumerate(
        zip(axs.flat, activation_specs, strict=True)
    ):
        plot_activation_components(
            ax=ax,
            spec=spec,
            xs=xs,
            z_orders=z_orders,
        )
        ax.set_aspect("equal")

        # Declutter to keep only outer tick labels
        row, col = divmod(i, n_cols)
        if row < (n_rows - 1):
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if col > 0:
            ax.tick_params(axis="y", which="both", labelleft=False)

        setup_grid(ax)

    # Add global legend (use the first axis as the source of handles/labels)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.4),
        bbox_transform=axs[n_rows - 1, n_cols // 2].transAxes,
    )

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.cell
def _():
    plot_activation_components_grid(
        ACTIVATIONS,
        fig_size=(8.0, 6.4),
        n_rows=3,
        n_cols=3,
        y_range=5.0,
        x_to_y_ratio=1.4,
        n_points=1001,
        z_orders=(3, 1, 2),
        output_path=OUTPUT_PATH,
    )
    return


if __name__ == "__main__":
    app.run()
