"""Shared style for the appendix figures.

``appendix.mplstyle`` drops the frame, grid and ticks. Each panel then
draws its subject in black and that subject's components in one color.
"""

__all__ = [
    "COMPONENT_CMAP_SAMPLE",
    "CURVE_COLOR",
    "DISPLAY_NAMES",
    "ORIGIN_COLOR",
    "ORIGIN_LINEWIDTH",
    "ORIGIN_ZORDER",
    "component_color",
    "draw_origin",
]

import matplotlib.axes as mpl_axes

from symmetries import colors

# The subject of a panel: its activation, or the realized function.
CURVE_COLOR = "k"

# Medium of the three samples the sequential colormaps are read at
# (OKLCH L = 0.55). Light enough to stay clear of the black curve.
COMPONENT_CMAP_SAMPLE = 0.6

# A faint cross for panels whose reading depends on the origin.
ORIGIN_COLOR = "0.8"
ORIGIN_LINEWIDTH = 0.6
ORIGIN_ZORDER = 0

# jax.nn activations, spelled as the surrounding prose spells them.
DISPLAY_NAMES = {
    "celu": "CELU",
    "elu": "ELU",
    "gelu": "GELU",
    "hard_sigmoid": "Hard sigmoid",
    "hard_silu": "Hard SiLU",
    "hard_tanh": "Hard tanh",
    "identity": "Identity",
    "leaky_relu": "Leaky ReLU",
    "log_sigmoid": "Log-sigmoid",
    "mish": "Mish",
    "relu": "ReLU",
    "relu6": "ReLU6",
    "selu": "SELU",
    "sigmoid": "Sigmoid",
    "silu": "SiLU",
    "soft_sign": "Softsign",
    "softplus": "Softplus",
    "sparse_plus": "Sparseplus",
    "sparse_sigmoid": "Sparse sigmoid",
    "squareplus": "Squareplus",
    "tanh": "tanh",
}


def component_color(hue: str) -> tuple[float, float, float, float]:
    """Return the color for components drawn beside a black curve."""
    return colors.get_sequential_cmap(hue)(COMPONENT_CMAP_SAMPLE)


def draw_origin(ax: mpl_axes.Axes) -> None:
    """Mark the origin with a faint cross."""
    ax.axhline(
        y=0, color=ORIGIN_COLOR, lw=ORIGIN_LINEWIDTH, zorder=ORIGIN_ZORDER
    )
    ax.axvline(
        x=0, color=ORIGIN_COLOR, lw=ORIGIN_LINEWIDTH, zorder=ORIGIN_ZORDER
    )
