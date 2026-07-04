"""2D visualization utilities for neural networks.

This module provides Matplotlib-based plotting functions for visualizing
two-dimensional neural networks and their decision boundaries. It is
designed for networks with 2D inputs, allowing inspection of learned
representations in the input space.

The module includes:

* hyperplane and normal vector plotting,
* output landscape visualization as heatmaps, and
* overlay of decision boundaries with data points.

Functions:
---------

* :func:`plot_data`: Plot data points with label-based coloring.
* :func:`plot_normal`: Draw the normal vector of a hyperplane.
* :func:`plot_hyperplane`: Plot a linear decision boundary and its
  normal.
* :func:`plot_mlp_hyperplanes`: Plot decision boundaries of all hidden
  neurons.
* :func:`plot_mlp_heatmap`: Plot the output landscape of a 2D MLP.
* :func:`plot_mlp`: Convenience wrapper combining heatmap, hyperplanes,
  and data.
* :func:`plot_mlps`: Plot multiple MLPs in a grid layout.
* :func:`plot_rdm`: Plot a representational (dis)similarity matrix.
* :func:`plot_matrix`: Plot a general 2D matrix as a heatmap.

Notes:
-----
All plotting functions accept an optional ``ax`` parameter to draw on
existing axes, enabling composition into multi-panel figures. Colors
for hidden neuron boundaries are determined by the sign of their
outgoing weights.
"""

__all__ = [
    "plot_data",
    "plot_hyperplane",
    "plot_matrix",
    "plot_mlp",
    "plot_mlp_heatmap",
    "plot_mlp_hyperplanes",
    "plot_mlps",
    "plot_normal",
    "plot_rdm",
]

import warnings
from collections.abc import Sequence
from typing import Any
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from symmetries import mlp

EPS = 1e-8

TriMode = Literal["full", "upper", "lower"]


def plot_data(
    data: tuple[Float[Array, "n 2"], Float[Array, " n"]],
    *,
    ax: plt.Axes,
    markers: Sequence[str] | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot data points on axes.

    Args:
        data: Tuple of (inputs, labels) to scatter plot.
        ax: Matplotlib axes to plot on.
        markers: Optional sequence of marker styles, one per data point.
          If provided, each point is plotted individually with its
          marker.
        **kwargs: Arguments passed to ``scatter``. Defaults to coloring
          points by label if ``c`` is not specified.

    Returns:
        The axes with the data points added.
    """
    kwargs.setdefault("zorder", 10)  # ensure data points are on top

    if markers is None:
        kwargs.setdefault("c", data[1])
        ax.scatter(data[0][:, 0], data[0][:, 1], **kwargs)
    else:
        # Plot each point individually with its marker
        cmap = kwargs.pop("cmap", None)
        c = kwargs.pop("c", data[1])
        if cmap is not None:
            cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            colors = [cmap_obj(int(label)) for label in c]
        else:
            colors = [None] * len(c)
        for i, (x, y) in enumerate(data[0]):
            ax.scatter(x, y, marker=markers[i], c=[colors[i]], **kwargs)
    return ax


def plot_normal(
    w: Float[Array, " 2"],
    b: float,
    *,
    ax: plt.Axes,
    color: str | None = None,
    length: float = 1.0,
    lw: float = 1.5,
) -> plt.Axes:
    """Plot normal vector of hyperplane wx + b = 0.

    Draws an arrow from the point on the hyperplane closest to the
    origin, pointing in the direction of the normal vector.

    Args:
        w: Normal vector of the hyperplane.
        b: Bias term of the hyperplane.
        ax: Matplotlib axes to plot on.
        color: Color for the arrow.
        length: Arrow length (fixed, independent of ``||w||``).
        lw: Line width for the arrow.

    Returns:
        The axes with the normal vector added.
    """
    w_norm = jnp.linalg.norm(w)
    w_unit = w / w_norm  # unit normal vector
    (x1, x2) = -(b / jnp.dot(w, w)) * w  # pt closest to origin
    (dx1, dx2) = length * w_unit  # fixed length, independent of ||w||

    ax.annotate(
        "",
        xytext=(x1, x2),
        xy=(x1 + dx1, x2 + dx2),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": lw,
        },
    )

    return ax


def plot_hyperplane(
    w: Float[Array, " 2"],
    b: float,
    *,
    ax: plt.Axes,
    color: str | None = None,
    show_normal: bool = True,
    normal_length: float = 1.0,
    normal_lw: float = 1.5,
) -> plt.Axes:
    """Plot the decision boundary wx + b = 0.

    Args:
        w: Normal vector of the hyperplane.
        b: Bias term of the hyperplane.
        ax: Matplotlib axes to plot on.
        color: Color for the hyperplane and normal vector.
        show_normal: If True, also plot the normal vector.
        normal_length: Length of the normal vector arrow. Ignored when
          ``show_normal`` is False.
        normal_lw: Line width for the normal vector arrow. Ignored when
          ``show_normal`` is False.

    Returns:
        The axes with the hyperplane added.
    """
    if jnp.linalg.norm(w) < EPS:
        msg = (
            "Hyperplane not numerically well-defined for near-zero "
            "normal vector"
        )
        warnings.warn(msg, stacklevel=2)
        return ax

    w1, w2 = w
    x1_min, x1_max = ax.get_xlim()
    x2_min, x2_max = ax.get_ylim()

    corner_vals = [
        w1 * x1 + w2 * x2 + b
        for x1 in (x1_min, x1_max)
        for x2 in (x2_min, x2_max)
    ]

    hyperplane_visible = not (
        all(v > 0 for v in corner_vals) or all(v < 0 for v in corner_vals)
    )

    if not hyperplane_visible:
        warnings.warn(
            "Hyperplane is not visible within the current axes limits",
            stacklevel=2,
        )
        return ax

    # NOTE: w1 x1 + w2 x2 + b = 0 gives x2 = (-w1 / w2) x1 - b / w2
    if jnp.abs(w2) > EPS:
        point = (0, -b / w2)
        slope = -w1 / w2
        ax.axline(xy1=point, slope=slope, linestyle="--", color=color)
    elif jnp.abs(w1) > EPS:
        ax.axvline(-b / w1, linestyle="--", color=color)

    if show_normal:
        ax = plot_normal(
            w,
            b,
            ax=ax,
            color=color,
            length=normal_length,
            lw=normal_lw,
        )

    return ax


def plot_mlp_hyperplanes(
    model: mlp.MLP,
    *,
    ax: plt.Axes,
    neuron_colors: tuple[str, str, str] = ("1.0", "0.5", "0.0"),
    show_normals: bool = True,
    normal_length: float = 1.0,
    normal_lw: float = 1.5,
    sign_threshold: float = 1e-8,
) -> plt.Axes:
    """Plot decision boundaries of all hidden neurons in a 2D MLP.

    Args:
        model: MLP model to visualize (must have 2D input).
        ax: Matplotlib axes to plot on.
        neuron_colors: Tuple of (positive, zero, negative) colors for
          neurons based on outgoing weight sign.
        show_normals: If True, plot normal vectors of hyperplanes.
        normal_length: Length of normal vector arrows. Ignored when
          ``show_normals`` is False.
        normal_lw: Line width for the normal vector arrows, applied
          uniformly to every neuron (independent of the readout weight
          magnitude). Ignored when ``show_normals`` is False.
        sign_threshold: Threshold for classifying outgoing weights as
          positive, zero, or negative.

    Returns:
        The axes with the hyperplanes added.
    """
    n_hidden = model.layers[0].out_features
    for neuron_idx in range(n_hidden):
        w = model.layers[0].weight[neuron_idx, :]
        b = model.layers[0].bias[neuron_idx]
        a = model.layers[2].weight[0, neuron_idx]

        if a > sign_threshold:
            color = neuron_colors[0]
        elif a < -sign_threshold:
            color = neuron_colors[2]
        else:
            color = neuron_colors[1]

        ax = plot_hyperplane(
            w=w,
            b=b,
            ax=ax,
            color=color,
            show_normal=show_normals,
            normal_length=normal_length,
            normal_lw=normal_lw,
        )

    return ax


def plot_mlp_heatmap(
    model: mlp.MLP,
    *,
    logits: bool = True,
    lim: float = 2.0,
    resolution: int = 200,
    colorbar_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the output landscape of a 2D MLP as a heatmap.

    Args:
        model: MLP model to visualize (must have 2D input).
        logits: If True, plot raw logits; if False, plot probabilities.
        lim: Axis limits (-lim, lim) for both x and y.
        resolution: Number of grid points along each axis.
        colorbar_kwargs: Arguments for colorbar. If None, no colorbar is
          drawn. Pass ``{}`` for defaults or a dict with options:
          ``pad`` (gap as fraction of axes width, default 0.08),
          ``width`` (colorbar width as fraction, default 0.05),
          ``label`` (colorbar label), and any other ``fig.colorbar()``
          kwargs.
        ax: Matplotlib axes to plot on. If None, creates a new figure.
        **kwargs: Additional arguments passed to ``pcolormesh`` and
          ``contour`` (e.g., ``cmap``, ``vmin``, ``vmax``).

    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    xx, yy = jnp.meshgrid(
        jnp.linspace(-lim, lim, resolution + 1),
        jnp.linspace(-lim, lim, resolution + 1),
    )
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    values = jax.vmap(model)(grid).squeeze(-1).reshape(xx.shape)
    if not logits:
        values = jax.nn.sigmoid(values)

    # Separate kwargs: rasterized only applies to pcolormesh, not contour
    contour_kwargs = {k: v for k, v in kwargs.items() if k != "rasterized"}
    mesh = ax.pcolormesh(xx, yy, values, shading="auto", **kwargs)
    ax.contour(xx, yy, values, **contour_kwargs)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # Draw colorbar if requested
    if colorbar_kwargs is not None:
        cbar_opts = colorbar_kwargs.copy()
        cbar_pad = cbar_opts.pop("pad", 0.08)
        cbar_width = cbar_opts.pop("width", 0.05)
        _draw_inset_colorbar(ax, mesh, cbar_pad, cbar_width, **cbar_opts)

    return fig, ax


def plot_mlp(
    model: mlp.MLP,
    data: tuple[Float[Array, "n 2"], Float[Array, " n"]] | None = None,
    *,
    heatmap_kwargs: dict | None = None,
    hyperplane_kwargs: dict | Literal[False] | None = None,
    data_kwargs: dict | None = None,
    colorbar_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize the output landscape of a 2D MLP.

    Convenience function that plots the network output as a heatmap with
    contour lines, optionally overlaying decision boundaries of hidden
    neurons and data points.

    Args:
        model: MLP model to visualize (must have 2D input).
        data: Optional tuple of (inputs, labels) to scatter plot.
        heatmap_kwargs: Arguments passed to :func:`plot_mlp_heatmap`.
          Supports ``logits``, ``lim``, ``resolution``, and any
          ``pcolormesh``/``contour`` kwargs (e.g., ``cmap``, ``vmin``,
          ``vmax``).
        hyperplane_kwargs: Arguments passed to
          :func:`plot_mlp_hyperplanes`. Supports ``neuron_colors``,
          ``sign_threshold``, ``show_normals``, ``normal_length``, and
          ``normal_lw``. Pass ``False`` to hide hyperplanes entirely.
        data_kwargs: Arguments passed to :func:`plot_data` (i.e.,
          ``scatter`` kwargs like ``c``, ``s``, ``edgecolors``, etc.).
        colorbar_kwargs: Arguments for colorbar. If None, no colorbar
          is drawn. Pass ``{}`` for defaults or a dict with options:
          ``pad`` (gap as fraction of axes width, default 0.08),
          ``width`` (colorbar width as fraction, default 0.05),
          ``label`` (colorbar label), and any other ``fig.colorbar()``
          kwargs.
        ax: Matplotlib axes to plot on. If None, creates a new figure.

    Returns:
        Tuple of (figure, axes).
    """
    heatmap_kwargs = heatmap_kwargs or {}
    data_kwargs = data_kwargs or {}

    fig, ax = plot_mlp_heatmap(
        model, ax=ax, colorbar_kwargs=colorbar_kwargs, **heatmap_kwargs
    )

    if hyperplane_kwargs is not False:
        plot_mlp_hyperplanes(model, ax=ax, **(hyperplane_kwargs or {}))

    if data:
        plot_data(data, ax=ax, **data_kwargs)

    return fig, ax


def plot_mlps(
    models: Sequence[mlp.MLP],
    ncols: int = 4,
    **kwargs: Any,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot multiple MLPs in a grid layout.

    Args:
        models: Sequence of MLP models to visualize.
        ncols: Number of columns in the grid.
        **kwargs: Additional arguments passed to :func:`plot_mlp`.

    Returns:
        Tuple of (figure, list of axes).
    """
    n = len(models)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    for model, ax in zip(models, axes[:n], strict=True):
        plot_mlp(model, ax=ax, **kwargs)

    for ax in axes[n:]:
        ax.axis("off")

    return fig, list(axes[:n])


def plot_rdm(  # noqa: C901
    rdm: Float[Array, "n n"],
    *,
    tri: TriMode = "full",
    show_diagonal: bool = True,
    imshow_kwargs: dict | None = None,
    edge_kwargs: dict | Literal[False] | None = None,
    tick_kwargs: dict | None = None,
    colorbar_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a representational (dis)similarity matrix as a heatmap.

    Uses ``imshow`` for the fill and ``LineCollection`` for grid lines
    that only appear around visible cells.

    Args:
        rdm: Square RDM/RSM matrix of shape (n, n).
        tri: Which part to display. One of "full", "upper", or "lower".
        show_diagonal: Whether to show the diagonal when ``tri`` is
          "upper" or "lower". Ignored when ``tri="full"``.
        imshow_kwargs: Arguments for color mapping. Supports ``cmap``,
          ``vmin``, ``vmax``, and any other ``imshow`` kwargs.
        edge_kwargs: Arguments for cell edge styling. Supports ``color``
          (default "0.5") and ``linewidth`` (default 0.8). Pass
          ``False`` to hide edges entirely.
        tick_kwargs: Arguments for marker ticks on axes. Supports
          ``markers`` (sequence of marker styles), ``colors`` (sequence
          of colors, default black), ``size`` (default 20), ``pad``
          (default 0.08), ``edgecolors``, and ``linewidths``.
        colorbar_kwargs: Arguments for colorbar. If None, no colorbar
          is drawn. Pass ``{}`` for defaults or a dict with options:
          ``pad`` (gap as fraction of axes width, default 0.08),
          ``width`` (colorbar width as fraction, default 0.05),
          ``label`` (colorbar label), and any other ``fig.colorbar()``
          kwargs.
        ax: Matplotlib axes to plot on. If None, creates a new figure.

    Returns:
        Tuple of (figure, axes).
    """
    data = np.asarray(rdm)

    if data.ndim != 2 or data.shape[0] != data.shape[1]:  # noqa: PLR2004
        msg = f"rdm must be square, got shape {data.shape}"
        raise ValueError(msg)

    n = data.shape[0]
    mask = _tri_mask(n, tri, show_diagonal=show_diagonal)
    masked = np.ma.array(data, mask=~mask)

    # Process imshow kwargs (copy to avoid mutating caller's dict)
    imshow_kwargs = dict(imshow_kwargs) if imshow_kwargs else {}
    cmap = imshow_kwargs.pop("cmap", None)
    vmin = imshow_kwargs.pop("vmin", None)
    vmax = imshow_kwargs.pop("vmax", None)

    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # NOTE: We're using pcolormesh instead of imshow for vector-based rendering
    #       to avoid interpolation when exporting to PDF.
    x_edges = np.arange(-0.5, n + 0.5, 1)
    y_edges = np.arange(-0.5, n + 0.5, 1)

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap=cmap_obj,
        norm=norm,
        shading="flat",
        **imshow_kwargs,
    )

    # Process edge kwargs
    if edge_kwargs is not False:
        edge_opts = edge_kwargs or {}
        edge_color = edge_opts.get("color", "0.5")
        edge_linewidth = edge_opts.get("linewidth", 0.8)

        if edge_color is not None and edge_linewidth > 0:
            segments = _visible_cell_edges(mask)
            lc = LineCollection(
                segments,
                colors=edge_color,
                linewidths=edge_linewidth,
                clip_on=False,
            )
            ax.add_collection(lc)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Process tick kwargs and draw marker symbols
    if tick_kwargs is not None:
        markers = tick_kwargs.get("markers")
        if markers is not None:
            _draw_rdm_tick_markers(
                ax,
                n,
                markers,
                tick_kwargs.get("colors"),
                tick_kwargs.get("size", 20),
                tick_kwargs.get("pad", 0.08),
                tick_kwargs.get("edgecolors"),
                tick_kwargs.get("linewidths"),
            )

    # Process colorbar kwargs and draw colorbar
    if colorbar_kwargs is not None:
        cbar_opts = colorbar_kwargs.copy()
        cbar_pad = cbar_opts.pop("pad", 0.08)
        cbar_width = cbar_opts.pop("width", 0.05)
        _draw_inset_colorbar(ax, im, cbar_pad, cbar_width, **cbar_opts)

    return fig, ax


def plot_matrix(  # noqa: C901
    matrix: Float[Array, "n_rows n_cols"],
    *,
    imshow_kwargs: dict | None = None,
    edge_kwargs: dict | Literal[False] | None = None,
    tick_kwargs: dict | None = None,
    colorbar_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a matrix as a heatmap with grid lines.

    Uses ``pcolormesh`` for the fill and ``LineCollection`` for grid
    lines, matching the visual style of :func:`plot_rdm`.

    Args:
        matrix: 2D array of shape (n_rows, n_cols).
        imshow_kwargs: Arguments for color mapping.  Supports ``cmap``,
          ``vmin``, ``vmax``, and any other ``pcolormesh`` kwargs.
        edge_kwargs: Arguments for cell edge styling.  Supports ``color``
          (default "0.5") and ``linewidth`` (default 0.8).  Pass
          ``False`` to hide edges entirely.
        tick_kwargs: Arguments for column marker ticks (bottom axis).
          Supports ``markers`` (sequence of marker styles), ``colors``
          (sequence of colors, default black), ``size`` (default 20),
          ``pad`` (default 0.08), ``edgecolors``, and ``linewidths``.
        colorbar_kwargs: Arguments for colorbar.  If None, no colorbar
          is drawn.  Pass ``{}`` for defaults or a dict with options:
          ``pad`` (gap as fraction of axes width, default 0.08),
          ``width`` (colorbar width as fraction, default 0.05),
          ``label`` (colorbar label), and any other ``fig.colorbar()``
          kwargs.
        ax: Matplotlib axes to plot on.  If None, creates a new figure.

    Returns:
        Tuple of (figure, axes).
    """
    data = np.asarray(matrix)

    if data.ndim != 2:  # noqa: PLR2004
        msg = f"matrix must be 2D, got shape {data.shape}"
        raise ValueError(msg)

    n_rows, n_cols = data.shape
    mask = np.ones((n_rows, n_cols), dtype=bool)

    # Process imshow kwargs (copy to avoid mutating caller's dict)
    imshow_kwargs = dict(imshow_kwargs) if imshow_kwargs else {}
    cmap = imshow_kwargs.pop("cmap", None)
    vmin = imshow_kwargs.pop("vmin", None)
    vmax = imshow_kwargs.pop("vmax", None)

    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap).copy()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # NOTE: We're using pcolormesh instead of imshow for vector-based rendering
    #       to avoid interpolation when exporting to PDF.
    x_edges = np.arange(-0.5, n_cols + 0.5, 1)
    y_edges = np.arange(-0.5, n_rows + 0.5, 1)

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        data,
        cmap=cmap_obj,
        norm=norm,
        shading="flat",
        **imshow_kwargs,
    )

    # Process edge kwargs
    if edge_kwargs is not False:
        edge_opts = edge_kwargs or {}
        edge_color = edge_opts.get("color", "0.5")
        edge_linewidth = edge_opts.get("linewidth", 0.8)

        if edge_color is not None and edge_linewidth > 0:
            segments = _visible_cell_edges(mask)
            lc = LineCollection(
                segments,
                colors=edge_color,
                linewidths=edge_linewidth,
                clip_on=False,
            )
            ax.add_collection(lc)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Process tick kwargs and draw column marker symbols (bottom axis)
    if tick_kwargs is not None:
        markers = tick_kwargs.get("markers")
        if markers is not None:
            _draw_col_tick_markers(
                ax,
                n_cols,
                markers,
                tick_kwargs.get("colors"),
                tick_kwargs.get("size", 20),
                tick_kwargs.get("pad", 0.08),
                tick_kwargs.get("edgecolors"),
                tick_kwargs.get("linewidths"),
            )

    # Process colorbar kwargs and draw colorbar
    if colorbar_kwargs is not None:
        cbar_opts = colorbar_kwargs.copy()
        cbar_pad = cbar_opts.pop("pad", 0.08)
        cbar_width = cbar_opts.pop("width", 0.05)
        _draw_inset_colorbar(ax, im, cbar_pad, cbar_width, **cbar_opts)

    return fig, ax


def _draw_col_tick_markers(
    ax: plt.Axes,
    n_cols: int,
    markers: Sequence[str],
    colors: Sequence[str] | None,
    size: float,
    pad: float,
    edgecolors: str | None,
    linewidths: float | None,
) -> None:
    """Draw marker symbols as column ticks (bottom axis) for matrix plots."""
    if colors is None:
        colors = ["black"] * n_cols

    # The pad is specified as a fraction of axes height. For non-square
    # matrices with equal aspect, scale it so the physical gap matches
    # what a square (n_cols x n_cols) matrix would produce.
    ylim = ax.get_ylim()
    n_rows_data = abs(ylim[0] - ylim[1])
    xlim = ax.get_xlim()
    n_cols_data = abs(xlim[1] - xlim[0])
    scale = n_cols_data / n_rows_data if n_rows_data > 0 else 1.0

    adj_pad = pad * scale
    tick_width = adj_pad

    # Build scatter kwargs for edge styling
    scatter_kwargs: dict = {"clip_on": False}
    if edgecolors is not None:
        scatter_kwargs["edgecolors"] = edgecolors
    if linewidths is not None:
        scatter_kwargs["linewidths"] = linewidths

    # Bottom edge (x-axis ticks) - inset below main axes
    ax_bottom = ax.inset_axes(
        (0, -tick_width - adj_pad, 1, tick_width),
        transform=ax.transAxes,
    )
    ax_bottom.set_xlim(-0.5, n_cols - 0.5)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_axis_off()
    for i in range(n_cols):
        ax_bottom.scatter(
            i,
            0.5,
            marker=markers[i],
            s=size,
            c=[colors[i]],
            **scatter_kwargs,
        )


def _draw_rdm_tick_markers(
    ax: plt.Axes,
    n: int,
    markers: Sequence[str],
    colors: Sequence[str] | None,
    size: float,
    pad: float,
    edgecolors: str | None,
    linewidths: float | None,
) -> None:
    """Draw marker symbols as axis ticks for RDM plots."""
    if colors is None:
        colors = ["black"] * n

    tick_width = pad  # width of tick axes as fraction of main axes

    # Build scatter kwargs for edge styling
    scatter_kwargs: dict = {"clip_on": False}
    if edgecolors is not None:
        scatter_kwargs["edgecolors"] = edgecolors
    if linewidths is not None:
        scatter_kwargs["linewidths"] = linewidths

    # Left edge (y-axis ticks) - inset to the left of main axes
    ax_left = ax.inset_axes(
        (-tick_width - pad, 0, tick_width, 1),
        transform=ax.transAxes,
    )
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(n - 0.5, -0.5)  # match main axes y-limits
    ax_left.set_axis_off()
    for i in range(n):
        ax_left.scatter(
            0.5,
            i,
            marker=markers[i],
            s=size,
            c=[colors[i]],
            **scatter_kwargs,
        )

    # Bottom edge (x-axis ticks) - inset below main axes
    ax_bottom = ax.inset_axes(
        (0, -tick_width - pad, 1, tick_width),
        transform=ax.transAxes,
    )
    ax_bottom.set_xlim(-0.5, n - 0.5)  # match main axes x-limits
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_axis_off()
    for i in range(n):
        ax_bottom.scatter(
            i,
            0.5,
            marker=markers[i],
            s=size,
            c=[colors[i]],
            **scatter_kwargs,
        )


def _draw_inset_colorbar(
    ax: plt.Axes,
    mappable: plt.cm.ScalarMappable,
    pad: float,
    width: float,
    **kwargs: Any,
) -> None:
    """Draw colorbar as inset_axes to the right of the plot."""
    fig = ax.get_figure()
    cax = ax.inset_axes(
        (1 + pad, 0, width, 1),
        transform=ax.transAxes,
    )
    fig.colorbar(mappable, cax=cax, **kwargs)


def _tri_mask(
    n: int,
    tri: TriMode,
    *,
    show_diagonal: bool,
) -> np.ndarray:
    """Create a boolean mask for displaying triangular matrices."""
    if tri == "full":
        return np.ones((n, n), dtype=bool)
    if tri == "upper":
        k = 0 if show_diagonal else 1
        return np.triu(np.ones((n, n), dtype=bool), k=k)
    if tri == "lower":
        k = 0 if show_diagonal else -1
        return np.tril(np.ones((n, n), dtype=bool), k=k)
    msg = f"tri must be 'full', 'upper', or 'lower', got {tri!r}"
    raise ValueError(msg)


def _visible_cell_edges(
    mask: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return all edge segments for visible cells."""
    nrows, ncols = mask.shape
    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    def normalize_edge(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Order edge endpoints consistently for deduplication."""
        return (p1, p2) if p1 <= p2 else (p2, p1)

    for row in range(nrows):
        for col in range(ncols):
            if not mask[row, col]:
                continue

            x0, x1 = col - 0.5, col + 0.5
            y0, y1 = row - 0.5, row + 0.5

            edges.add(normalize_edge((x0, y0), (x1, y0)))  # top
            edges.add(normalize_edge((x1, y0), (x1, y1)))  # right
            edges.add(normalize_edge((x1, y1), (x0, y1)))  # bottom
            edges.add(normalize_edge((x0, y1), (x0, y0)))  # left

    return list(edges)
