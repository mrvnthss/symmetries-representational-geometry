import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path
    from typing import Any

    import matplotlib.axes as mpl_axes
    import matplotlib.pyplot as plt
    import numpy as np
    from figures.fonts.config import FIGURE_FONT_CONFIGS
    from figures.fonts.config import register_bundled_fonts
    from jaxtyping import Array
    from matplotlib import colorbar as mpl_colorbar
    from matplotlib import colors as mpl_colors
    from matplotlib import legend as mpl_legend
    from matplotlib import transforms as mpl_transforms
    from matplotlib.font_manager import FontProperties
    from matplotlib.legend_handler import HandlerLine2D
    from matplotlib.lines import Line2D
    from matplotlib.textpath import TextPath

    from figures.main.dissociation.components import DATA_COLOR_0
    from figures.main.dissociation.components import DATA_COLOR_1
    from figures.main.dissociation.components import DATA_EDGECOLOR
    from figures.main.dissociation.components import DATA_KWARGS
    from figures.main.dissociation.components import DATA_LINEWIDTH
    from figures.main.dissociation.components import DATA_MARKERS
    from figures.main.dissociation.components import HEATMAP_KWARGS
    from figures.main.dissociation.components import HYPERPLANE_KWARGS
    from symmetries import colors
    from symmetries import mlp
    from symmetries import plotting
    from symmetries import xor

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    register_bundled_fonts()

    # Importing the Figure 1 components changes a few rcParams for their
    # SVG exports, so start from the Matplotlib defaults.
    plt.style.use(["default", "./style.mplstyle"])

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STEM = "xor-solutions"

    # One family per row, in the order of ``xor.relu_solution_specs``:
    # diagonal solutions on top, anti-diagonal ones below.
    N_ROWS = 2
    N_COLS = len(xor.relu_solutions) // N_ROWS

    # Panels are numbered as the rows of the solution table.
    SOLUTION_TITLES = tuple(
        f"Solution {idx + 1}" for idx in range(len(xor.relu_solutions))
    )

    # Boundary color by the sign of the readout weight (positive, zero,
    # negative), as in ``plotting.plot_mlp_hyperplanes``.
    NEURON_COLORS = ("1.0", "0.5", "0.0")

    # Legend entries in drawing order: the two boundary types, then the
    # two classes of the XOR data.
    FIGURE_LEGEND_LABELS = (
        r"$a_{j} > 0$",
        r"$a_{j} < 0$",
        r"$y^{\mu} = 0$",
        r"$y^{\mu} = 1$",
    )

    # Colorbar ticks, at multiples of the logit every solution gives
    # every data point (see ``data_logit_magnitude``). Twice that logit
    # is the distance the data project onto.
    COLORBAR_TICK_LABELS = (
        r"$-\sqrt{2}$",
        r"$-\frac{1}{\sqrt{2}}$",
        r"$0$",
        r"$\frac{1}{\sqrt{2}}$",
        r"$\sqrt{2}$",
    )

    # MathText does not apply italic correction between custom italic and
    # roman fonts. Add the corrections needed by the Libertinus glyphs.
    COLORBAR_LABELS = {
        "submission": r"$f_{\boldsymbol{\theta}}(\mathbf{x})$",
        "preprint": (
            r"$f_{\boldsymbol{\theta}}\hspace{0.15}"
            r"(\mathbf{x}\hspace{0.12})$"
        ),
    }

    # Height of a legend handle box, in multiples of the legend font
    # size. It sets how large the handles may be drawn (see
    # ``legend_marker_size``); ``MathAxisHandlerLine2D`` places them.
    # Above about 1.05 the box grows taller than the labels and the
    # frame's border padding goes out of balance.
    LEGEND_HANDLE_HEIGHT = 1.0

    # The rounded grey box the composed figures put their panel groups
    # in. Outline and corner radius come from
    # ``main/dissociation/dissociation.svg``. The fill is a darker grey
    # than that one, so the white boundary swatch stays visible.
    LEGEND_BOX_FACECOLOR = colors.get_color("grey", "L5", "muted")
    LEGEND_BOX_EDGECOLOR = "#666666"
    LEGEND_BOX_LINEWIDTH = 0.567  # points
    LEGEND_BOX_ROUNDING = 2.834  # corner radius, in points

    class MathAxisHandlerLine2D(HandlerLine2D):
        """Draw legend handles on the math axis of the labels.

        Matplotlib places a handle at ``(height - ydescent) / 2`` above
        the baseline, which rises with the handle height. Since the
        handle height also sets how large a marker fits, a larger marker
        would sit above its label. Reading the height off the font
        instead keeps the two independent.

        The handle's y coordinate is in points above the baseline, the
        origin of the area a handle is drawn into.
        """

        def __init__(self, y_center: float, **kwargs: Any) -> None:
            """Store the handle height, in points above the baseline."""
            super().__init__(**kwargs)
            self._y_center = y_center

        def create_artists(
            self,
            legend: mpl_legend.Legend,
            orig_handle: Line2D,
            xdescent: float,
            ydescent: float,
            width: float,
            height: float,
            fontsize: float,
            trans: mpl_transforms.Transform,
        ) -> list[Line2D]:
            """Create the handle artists, on the math axis."""
            artists = super().create_artists(
                legend,
                orig_handle,
                xdescent,
                ydescent,
                width,
                height,
                fontsize,
                trans,
            )
            for artist in artists:
                artist.set_ydata(
                    np.full_like(artist.get_xdata(), self._y_center)
                )
            return artists

    # Stroke width that closes the colorbar's seams (see
    # ``seal_colorbar``), under half the outline width so it stays
    # hidden beneath the colorbar's spine.
    COLORBAR_SEAM_LW = 0.4


@app.function
def plot_xor_solution(
    ax: mpl_axes.Axes,
    model: mlp.MLP,
    data: tuple[Array, Array],
    *,
    title: str,
    data_size: float,
    normal_lw: float,
) -> None:
    """Draw one solution: logit landscape, boundaries, and XOR data.

    Args:
        ax: Axes to draw the panel on.
        model: The solution network to visualize.
        data: The ``(inputs, labels)`` pair of the XOR task.
        title: Panel title, naming the solution.
        data_size: Marker area of the data points, in points squared.
        normal_lw: Line width of the boundary normal arrows.
    """
    plotting.plot_mlp(
        model=model,
        data=data,
        heatmap_kwargs=HEATMAP_KWARGS,
        hyperplane_kwargs={
            **HYPERPLANE_KWARGS,
            "neuron_colors": NEURON_COLORS,
            "normal_lw": normal_lw,
        },
        data_kwargs={**DATA_KWARGS, "s": data_size},
        ax=ax,
    )

    # The brackets already mark the four data points.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


@app.function
def data_logit_magnitude() -> float:
    """Return the logit magnitude every solution gives every data point.

    All six map all four inputs to a logit of the same magnitude.

    Returns:
        The shared magnitude of the logits.
    """
    inputs, _ = xor.xor_dataset()
    return max(
        abs(float(model(point)[0]))
        for model in xor.relu_solutions
        for point in inputs
    )


@app.function
def math_axis_height(fontsize: float) -> float:
    """Return the height of the math axis above the baseline, in points.

    The math axis is the line relation and operator signs are centered
    on. Every label here is a relation, so it is the height a swatch
    beside one should sit at. A label's ink box is not: a descender
    lowers its center, a superscript raises it.

    Measured from the glyph outline of an equals sign, which the axis
    runs through by construction.

    Args:
        fontsize: Legend font size, in points.

    Returns:
        The axis height, in points above the baseline.
    """
    ink = TextPath((0.0, 0.0), r"$=$", size=fontsize).get_extents()
    return (ink.y0 + ink.y1) / 2


@app.function
def legend_marker_size(handle_height: float, fontsize: float) -> float:
    """Return the marker size a legend handle box has room for.

    Matplotlib draws a legend handle into a box of ``fontsize *
    handle_height``, less a descent of ``0.35 * fontsize *
    (handle_height - 0.7)``.

    Args:
        handle_height: Handle box height, in multiples of the font size.
        fontsize: Legend font size, in points.

    Returns:
        The marker size, in points.
    """
    descent = 0.35 * fontsize * (handle_height - 0.7)
    return fontsize * handle_height - descent


@app.function
def legend_handles(marker_size: float) -> list[Line2D]:
    """Build proxy artists for the boundary types and data classes.

    Args:
        marker_size: Side length of the data markers, in points.

    Returns:
        The four handles, boundaries first, in label order.
    """
    boundaries = [
        Line2D([], [], color=color, ls="--")
        for color in (NEURON_COLORS[0], NEURON_COLORS[2])
    ]
    points = [
        Line2D(
            [],
            [],
            ls="none",
            marker=DATA_MARKERS[0],
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor=DATA_EDGECOLOR,
            markeredgewidth=DATA_LINEWIDTH,
        )
        for color in (DATA_COLOR_0, DATA_COLOR_1)
    ]
    return boundaries + points


@app.function
def style_legend(legend: mpl_legend.Legend, fontsize: float) -> None:
    """Give the legend the rounded grey frame of the composed figures.

    Args:
        legend: The legend to restyle, created with ``fancybox=True``
          so that its frame has roundable corners.
        fontsize: Legend font size, in points.
    """
    frame = legend.get_frame()

    # Matplotlib draws legend frames at 80% opacity, which would tint
    # the colors below toward the background.
    frame.set_alpha(None)
    frame.set_facecolor(LEGEND_BOX_FACECOLOR)
    frame.set_edgecolor(LEGEND_BOX_EDGECOLOR)
    frame.set_linewidth(LEGEND_BOX_LINEWIDTH)

    # The frame's mutation scale is reset to the font size on every
    # draw, so the corner radius is given in those units.
    frame.set_boxstyle(
        "round", pad=0.0, rounding_size=LEGEND_BOX_ROUNDING / fontsize
    )


@app.function
def seal_colorbar(cbar: mpl_colorbar.Colorbar) -> None:
    """Close the hairline gaps the colorbar renders with.

    Matplotlib rasterizes a color mesh of this many quads, and in the
    PDF the resulting image does not meet the vector patch that draws
    the lower extension, leaving a white line across the bar. Drawing
    the mesh as vectors removes that boundary but leaves a hairline
    between neighboring quads, which stroking each one in its own fill
    color closes.
    """
    cbar.solids.set_rasterized(False)
    cbar.solids.set(edgecolor="face", linewidth=COLORBAR_SEAM_LW)


@app.function
def align_colorbar(
    fig: plt.Figure,
    cbar: mpl_colorbar.Colorbar,
    axs: np.ndarray,
) -> None:
    """Stretch the colorbar to the vertical extent of the panel grid.

    The panels have an equal aspect ratio, so they are shorter than the
    cells the layout engine gives them, while the colorbar fills its own
    cell. Letting the layout settle first makes the panels' extent
    readable off the figure.

    ``fig.colorbar`` ties the bar's length to its width through a box
    aspect ratio, which would override the position set here, so that
    constraint is released. The ``aspect`` passed to ``colorbar`` then
    only sets the bar's width. The position spans the whole bar,
    extension triangles included.

    Args:
        fig: The figure the colorbar belongs to.
        cbar: The colorbar to stretch.
        axs: The grid of panels to align it with.
    """
    fig.draw_without_rendering()  # let constrained layout settle
    fig.set_layout_engine("none")  # freeze it, so the move survives

    top = axs[0, -1].get_position().y1
    bottom = axs[-1, -1].get_position().y0
    pos = cbar.ax.get_position(original=True)
    cbar.ax.set_box_aspect(None)
    cbar.ax.set_position((pos.x0, bottom, pos.width, top - bottom))


@app.function
def plot_xor_solutions(
    *,
    fig_size: tuple[float, float],
    data_size: float,
    normal_lw: float,
    legend_labels: tuple[str, str, str, str],
    colorbar_label: str,
    colorbar_kwargs: dict,
    legend_y: float,
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the six analytical ReLU solutions of the XOR task.

    Each panel shows the logit landscape of one solution over the input
    square, the activation boundary of each hidden unit together with
    its normal, and the four data points. Rows hold the two families:
    within a row the solutions share the orientation of their
    boundaries, and differ only in where those boundaries sit and in the
    signs of the readout weights. All six realize the same function on
    the data, mapping every point to a logit of magnitude 1/sqrt(2).

    Args:
        fig_size: Figure size in inches.
        data_size: Marker area of the data points, in points squared.
        normal_lw: Line width of the boundary normal arrows.
        legend_labels: Legend labels for the two boundary types and the
          two classes of the XOR data.
        colorbar_label: Label of the shared colorbar.
        colorbar_kwargs: Arguments passed to ``fig.colorbar``.
        legend_y: Vertical anchor of the legend, in axes coordinates of
          the bottom center panel.
        output_path: If given, the path to save the figure to.
        show: Whether to display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.
    """
    data = xor.xor_dataset()

    fig, axs = plt.subplots(
        nrows=N_ROWS,
        ncols=N_COLS,
        sharex=True,
        sharey=True,
        figsize=fig_size,
    )

    panels = zip(axs.flat, xor.relu_solutions, SOLUTION_TITLES, strict=True)
    for ax, model, title in panels:
        plot_xor_solution(
            ax,
            model,
            data,
            title=title,
            data_size=data_size,
            normal_lw=normal_lw,
        )

    # One colorbar for all six panels: they share the color range,
    # which the landscapes exceed only in the far corners.
    mappable = plt.cm.ScalarMappable(
        norm=mpl_colors.Normalize(
            vmin=HEATMAP_KWARGS["vmin"], vmax=HEATMAP_KWARGS["vmax"]
        ),
        cmap=HEATMAP_KWARGS["cmap"],
    )
    cbar = fig.colorbar(mappable, ax=axs, extend="both", **colorbar_kwargs)
    cbar.set_label(colorbar_label)
    logit = data_logit_magnitude()
    cbar.set_ticks(
        (-2 * logit, -logit, 0.0, logit, 2 * logit),
        labels=COLORBAR_TICK_LABELS,
    )
    seal_colorbar(cbar)

    # Anchored below the panel grid, not the figure, which the colorbar
    # pushes off center.
    legend_fontsize = FontProperties(
        size=plt.rcParams["legend.fontsize"]
    ).get_size_in_points()
    legend = fig.legend(
        legend_handles(
            legend_marker_size(LEGEND_HANDLE_HEIGHT, legend_fontsize)
        ),
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=axs[N_ROWS - 1, N_COLS // 2].transAxes,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.6,
        fancybox=True,
        handler_map={
            Line2D: MathAxisHandlerLine2D(math_axis_height(legend_fontsize))
        },
    )
    style_legend(legend, legend_fontsize)

    align_colorbar(fig, cbar, axs)

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.function
def export_xor_solutions_variants(
    *,
    output_dir: Path,
    output_stem: str,
    fig_size: tuple[float, float],
    data_size: float,
    normal_lw: float,
    colorbar_kwargs: dict,
    legend_y: float,
) -> None:
    """Export submission and preprint variants with matched fonts."""
    for variant, font_config in FIGURE_FONT_CONFIGS.items():
        variant_output_dir = output_dir / variant
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = variant_output_dir / f"{output_stem}.pdf"
        with plt.rc_context(font_config):
            fig = plot_xor_solutions(
                fig_size=fig_size,
                data_size=data_size,
                normal_lw=normal_lw,
                legend_labels=FIGURE_LEGEND_LABELS,
                colorbar_label=COLORBAR_LABELS[variant],
                colorbar_kwargs=colorbar_kwargs,
                legend_y=legend_y,
                output_path=output_path,
            )
        plt.close(fig)


@app.cell
def _():
    export_xor_solutions_variants(
        output_dir=OUTPUT_DIR,
        output_stem=OUTPUT_STEM,
        fig_size=(6.8, 4.29),
        data_size=160.0,
        normal_lw=1.5,
        colorbar_kwargs={
            "shrink": 0.9,
            "aspect": 30,
            "pad": 0.02,
        },
        legend_y=-0.144,
    )
    return


if __name__ == "__main__":
    app.run()
