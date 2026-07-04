"""Color utilities and reusable palettes for scientific plotting.

This module provides a systematic color system built on OKLCH color
space, with orthogonal hue, lightness, and chroma parameters. It
includes:

* A curated set of 8 hues with good sRGB gamut coverage
* 5 lightness levels for different use cases (dark text to light backgrounds)
* 3 chroma levels (muted, medium, vivid)
* Pre-built qualitative, sequential, and diverging palettes
* Helpers for Matplotlib integration and accessibility testing

Functions:
---------

* :func:`use`: Set a palette as the global Matplotlib default cycle.
* :func:`get_cycler`: Build a Matplotlib color cycler from a palette.
* :func:`get_color`: Get a specific color by hue, lightness, and chroma.
* :func:`get_sequential_cmap`: Build a sequential colormap from a hue.
* :func:`get_diverging_cmap`: Build a diverging colormap from two hues.
* :func:`preview`: Preview one or more palettes as horizontal color bars.
* :func:`simulate_deficiencies`: Simulate common color-vision deficiencies.

Notes:
-----
The color system uses OKLCH (Lightness, Chroma, Hue) for perceptual
uniformity. Not all (hue, lightness, chroma) combinations are in sRGB
gamut; out-of-gamut colors are automatically clipped. The provided hues
are selected to maximize gamut coverage at the defined lightness and
chroma levels.
"""

__all__ = [
    "CHROMA",
    "DIVERGING",
    "HUES",
    "LIGHTNESS",
    "PALETTES",
    "get_color",
    "get_cycler",
    "get_diverging_cmap",
    "get_sequential_cmap",
    "preview",
    "simulate_deficiencies",
    "use",
]

import io

import cycler
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from coloraide import Color
from daltonize import daltonize

# ── OKLCH Color System Parameters ─────────────────────────────────────

# ── Hues ──────────────────────────────────────────────────────────────
# 8 hues selected for:
# - Good sRGB gamut coverage at our lightness/chroma levels
# - Perceptual distinctiveness
# - Aesthetic appeal for scientific visualization
# - Reasonable colorblind discrimination
#
# Note: We skip the 180-210° cyan region which has poor gamut at extreme
# lightness values.

HUES = {
    "red": 30,
    "orange": 55,
    "yellow": 85,
    "green": 150,
    "teal": 195,
    "blue": 250,
    "purple": 320,
    "grey": None,  # achromatic, handled specially
}

# Ordered list of hue names for consistent iteration
HUE_ORDER = [
    "red",
    "orange",
    "yellow",
    "green",
    "teal",
    "blue",
    "purple",
    "grey",
]

# ── Lightness Levels ──────────────────────────────────────────────────
# 5 levels covering the practical range for visualization:
# - L1: Dark (text, lines on light backgrounds)
# - L2: Medium-dark (primary colors, good contrast)
# - L3: Medium (fills, balanced visibility)
# - L4: Light (light fills, secondary elements)
# - L5: Very light (tinted backgrounds, subtle accents)

LIGHTNESS = {
    "L1": 0.30,  # dark, for text/lines/borders
    "L2": 0.45,  # medium-dark, deep colors
    "L3": 0.60,  # medium, standard colors
    "L4": 0.75,  # medium-light, pastel/paired
    "L5": 0.88,  # light, tinted backgrounds
}

# ── Chroma Levels ─────────────────────────────────────────────────────
# 3 levels providing flexibility from subtle to vibrant:
# - muted: Subtle, works everywhere, good for backgrounds
# - medium: Balanced saturation, good all-around choice
# - vivid: High saturation, best at mid-lightness (L2-L4)
#
# Gamut note: vivid (0.15) may clip at L1/L5 for some hues

CHROMA = {
    "muted": 0.08,  # subtle, for backgrounds
    "medium": 0.15,  # rich saturation, main choice
    "vivid": 0.19,  # high saturation, vibrant
}


# ── Color Generation ──────────────────────────────────────────────────


def _oklch_to_hex(L: float, C: float, H: float | None) -> str:
    """Convert OKLCH to hex, fitting to sRGB gamut if needed."""
    # H=None means achromatic grey (chroma=0)
    c = Color("oklch", [L, 0, 0]) if H is None else Color("oklch", [L, C, H])
    return c.convert("srgb").fit("srgb").to_string(hex=True)


def get_color(
    hue: str,
    lightness: str = "L3",
    chroma: str = "medium",
) -> str:
    """Get a color by hue name, lightness level, and chroma level.

    Args:
        hue: Hue name from :data:`HUES` (e.g., "red", "blue", "green").
        lightness: Lightness level from :data:`LIGHTNESS` (e.g., "L1",
          "L3").
        chroma: Chroma level from :data:`CHROMA` (e.g., "muted", "vivid").

    Returns:
        Hex color string (e.g., "#a83240").

    Raises:
        ValueError: If any parameter is not a valid key.

    Example:
        >>> get_color("blue", "L3", "vivid")
        '#4d73c4'
    """
    if hue not in HUES:
        msg = f"Unknown hue '{hue}'. Choose from: {list(HUES.keys())}"
        raise ValueError(msg)
    if lightness not in LIGHTNESS:
        opts = list(LIGHTNESS.keys())
        msg = f"Unknown lightness '{lightness}'. Choose from: {opts}"
        raise ValueError(msg)
    if chroma not in CHROMA:
        msg = f"Unknown chroma '{chroma}'. Choose from: {list(CHROMA.keys())}"
        raise ValueError(msg)

    return _oklch_to_hex(LIGHTNESS[lightness], CHROMA[chroma], HUES[hue])


# ── Pre-built Palettes ────────────────────────────────────────────────


def _build_palette(
    lightness: str,
    chroma: str,
) -> list[str]:
    """Build a palette with all hues at given lightness and chroma."""
    return [get_color(h, lightness, chroma) for h in HUE_ORDER]


PALETTES = {
    # ── Qualitative palettes (seaborn-style naming) ───────────────────
    "deep": _build_palette("L2", "vivid"),  # dark-ish, saturated
    "bright": _build_palette("L3", "vivid"),  # medium, saturated
    "muted": _build_palette("L3", "muted"),  # medium, desaturated
    "pastel": _build_palette("L4", "medium"),  # light, soft
    "dark": _build_palette("L1", "vivid"),  # very dark, saturated
    # ── Paired palette (dark/light pairs for each hue) ────────────────
    "paired": [
        color
        for h in HUE_ORDER
        for color in [get_color(h, "L2", "vivid"), get_color(h, "L4", "medium")]
    ],
    # ── Sequential palettes (single hue, varying lightness) ───────────
    **{
        f"seq_{hue}": [
            get_color(hue, L, "medium") for L in ["L5", "L4", "L3", "L2", "L1"]
        ]
        for hue in HUE_ORDER
    },
}

# ── Diverging colormap definitions (hue pairs) ────────────────────────
DIVERGING = {
    "div_blue_red": ("blue", "red"),
    "div_blue_orange": ("blue", "orange"),
    "div_teal_red": ("teal", "red"),
    "div_purple_green": ("purple", "green"),
}

# Add reversed variants for all palettes
PALETTES.update(
    {f"{name}_r": colors[::-1] for name, colors in list(PALETTES.items())}
)


# ── Matplotlib Integration ────────────────────────────────────────────


def use(name: str = "deep") -> None:
    """Set a palette as the global default Matplotlib color cycle.

    This updates ``matplotlib.rcParams["axes.prop_cycle"]`` for the
    current Python session.

    Args:
        name: Palette name. Must be a key in :data:`PALETTES`.

    Raises:
        ValueError: If ``name`` is not a known palette.
    """
    plt.rcParams["axes.prop_cycle"] = get_cycler(name)


def get_cycler(name: str = "deep") -> cycler.Cycler:
    """Build a Matplotlib color cycler for a named palette.

    Args:
        name: Palette name. Must be a key in :data:`PALETTES`.

    Returns:
        A :class:`cycler.Cycler` that cycles the ``color`` property.

    Raises:
        ValueError: If ``name`` is not a known palette.
    """
    if name not in PALETTES:
        msg = f"Unknown palette '{name}'. Choose from: {list(PALETTES.keys())}"
        raise ValueError(msg)
    return cycler.cycler(color=PALETTES[name])


# ── Colormap Generation ───────────────────────────────────────────────


def get_sequential_cmap(
    hue: str | float,
    l_min: float = 0.30,
    l_max: float = 0.92,
    c_min: float = 0.04,
    c_max: float = 0.20,
    n: int = 256,
) -> mpl_colors.ListedColormap:
    """Create a sequential colormap from a single hue.

    Generates a perceptually uniform colormap in OKLCH space with constant
    hue, varying lightness from light to dark, and chroma that increases
    toward darker values.

    Args:
        hue: Hue name from :data:`HUES` (e.g., "blue") or numeric OKLCH
          hue in degrees (0-360).
        l_min: Minimum lightness (dark end). Default 0.30.
        l_max: Maximum lightness (light end). Default 0.92.
        c_min: Minimum chroma (at light end). Default 0.04.
        c_max: Maximum chroma (at dark end). Default 0.20.
        n: Number of discrete samples in the colormap.

    Returns:
        A :class:`matplotlib.colors.ListedColormap` going from light to
        dark.
    """
    h = HUES.get(hue) if isinstance(hue, str) else float(hue)
    is_grey = h is None

    rgba = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        L = l_max - t * (l_max - l_min)
        C = 0 if is_grey else c_min + (c_max - c_min) * t

        c = Color("oklch", [L, C, h or 0]).convert("srgb").fit("srgb")
        rgba.append([c["red"], c["green"], c["blue"], 1.0])

    name = hue if isinstance(hue, str) else f"seq_{int(hue)}"
    return mpl_colors.ListedColormap(rgba, name=name)


def get_diverging_cmap(
    hue_low: str | float,
    hue_high: str | float,
    l_min: float = 0.35,
    l_max: float = 0.97,
    c_max: float = 0.20,
    n: int = 256,
) -> mpl_colors.ListedColormap:
    """Create a diverging colormap from two hues.

    Generates a perceptually uniform colormap in OKLCH space with a
    neutral (near-white) center and two colored arms diverging to
    saturated endpoints.

    Args:
        hue_low: Hue for negative values (left arm). Can be a name from
          :data:`HUES` or numeric degrees (0-360).
        hue_high: Hue for positive values (right arm).
        l_min: Minimum lightness (at extremes). Default 0.35.
        l_max: Maximum lightness (at center). Default 0.97.
        c_max: Peak chroma (at extremes). Default 0.20.
        n: Number of discrete samples in the colormap.

    Returns:
        A :class:`matplotlib.colors.ListedColormap` diverging from
        center.
    """
    h_low = (
        HUES[hue_low]
        if isinstance(hue_low, str) and hue_low in HUES
        else float(hue_low)
    )
    h_high = (
        HUES[hue_high]
        if isinstance(hue_high, str) and hue_high in HUES
        else float(hue_high)
    )

    rgba = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.5
        d = abs(2 * t - 1)  # distance from center

        L = l_max - d * (l_max - l_min)
        C = c_max * d
        midpoint = 0.5
        H = h_low if t < midpoint else h_high

        c = Color("oklch", [L, C, H]).convert("srgb").fit("srgb")
        rgba.append([c["red"], c["green"], c["blue"], 1.0])

    name_low = hue_low if isinstance(hue_low, str) else str(int(hue_low))
    name_high = hue_high if isinstance(hue_high, str) else str(int(hue_high))
    return mpl_colors.ListedColormap(rgba, name=f"div_{name_low}_{name_high}")


# ── Visualization & Accessibility ─────────────────────────────────────


def preview(
    name: str | None = None,
    figsize_per_row: tuple[float, float] = (10, 0.5),
) -> plt.Figure:
    """Preview one or more palettes as horizontal color bars.

    Args:
        name: Optional palette name. If provided, only this palette
          is displayed. If ``None``, all non-reversed palettes are shown.
        figsize_per_row: Base figure size per palette row as (width,
          height).

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.

    Raises:
        KeyError: If ``name`` is not a valid palette.
    """
    all_names = {**PALETTES, **DIVERGING}
    if name and name not in all_names:
        msg = f"Unknown palette '{name}'. Choose from: {list(all_names.keys())}"
        raise KeyError(msg)

    if name:
        names = [name]
    else:
        # qualitative + sequential + diverging (no reversed variants)
        names = [n for n in PALETTES if not n.endswith("_r")]
        names.extend(DIVERGING.keys())

    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * len(names)),
    )
    if len(names) == 1:
        axes = [axes]

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for ax, n in zip(axes, names, strict=True):
        if n.startswith("seq_"):
            # sequential palettes as continuous colormaps
            hue_name = n[4:]  # strip "seq_" prefix
            cmap = get_sequential_cmap(hue_name)
            ax.imshow(gradient, aspect="auto", cmap=cmap)
        elif n.startswith("div_"):
            # diverging palettes as continuous colormaps
            hue_low, hue_high = DIVERGING[n]
            cmap = get_diverging_cmap(hue_low, hue_high)
            ax.imshow(gradient, aspect="auto", cmap=cmap)
        else:
            # qualitative palettes as discrete swatches
            pal_colors = PALETTES[n]
            for i, c in enumerate(pal_colors):
                ax.barh(0, 1, left=i, color=c, edgecolor="none", height=1)
            ax.set_xlim(0, len(pal_colors))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=9)
        ax.spines[:].set_visible(False)

    fig.suptitle("Color Palettes", y=1.02)
    return fig


def preview_system(figsize: tuple[float, float] = (12, 8)) -> plt.Figure:
    """Preview the entire color system as a grid.

    Shows all hues x lightness levels at medium chroma, providing
    an overview of the full color space.

    Args:
        figsize: Figure size in inches.

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.
    """
    fig, axes = plt.subplots(
        len(LIGHTNESS),
        len(HUES),
        figsize=figsize,
    )

    for i, (l_name, l_val) in enumerate(LIGHTNESS.items()):
        for j, h_name in enumerate(HUE_ORDER):
            ax = axes[i, j]
            color = get_color(h_name, l_name, "medium")
            ax.set_facecolor(color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            if i == 0:
                ax.set_title(h_name, fontsize=10)
            if j == 0:
                ax.set_ylabel(
                    f"{l_name}\n({l_val:.2f})",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )

            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Color System: Hues x Lightness (medium chroma)", y=1.02)
    return fig


def simulate_deficiencies(fig: plt.Figure) -> plt.Figure:
    """Simulate common color-vision deficiencies for a rendered figure.

    The input figure is rasterized and passed through color-vision
    deficiency simulations (deuteranopia, protanopia, and tritanopia).

    Args:
        fig: A Matplotlib figure to simulate.

    Returns:
        A new figure with the original and three CVD simulations.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = plt.imread(buf)[..., :3]

    w, h = 1.4 * fig.get_size_inches()[0], 1.4 * fig.get_size_inches()[1]
    comp_fig, axes = plt.subplots(2, 2, figsize=(w, h))
    titles = ["Original", "Deuteranopia", "Protanopia", "Tritanopia"]
    simulations = [img, *[daltonize.simulate(img, def_t) for def_t in "dpt"]]

    for ax, title, sim_img in zip(axes.flat, titles, simulations, strict=True):
        ax.imshow(sim_img)
        ax.set_title(title)
        ax.axis("off")

    return comp_fig
