r"""Bundled fonts and per-venue font configuration for the paper figures.

The appendix figures carry their fonts inside the exported PDF, so each
venue gets its own export and ``\appfigdir`` selects the matching set on
the LaTeX side. Registering the bundled faces keeps figure rendering
independent of the fonts installed on the host system.
"""

__all__ = [
    "FIGURE_FONT_CONFIGS",
    "register_bundled_fonts",
]

from pathlib import Path

from matplotlib import font_manager

_FONT_DIR = Path(__file__).resolve().parent

_FONT_PATHS = (
    _FONT_DIR / "fira" / "FiraMono-Regular.otf",
    _FONT_DIR / "libertinus" / "LibertinusMono-Regular.otf",
    _FONT_DIR / "libertinus" / "LibertinusSerif-Regular.otf",
    _FONT_DIR / "libertinus" / "LibertinusSerif-Italic.otf",
    _FONT_DIR / "libertinus" / "LibertinusSerif-Bold.otf",
    _FONT_DIR / "tex-gyre" / "TeXGyreTermes-Regular.otf",
)

# Body fonts of the two targets: Times for the conference template,
# Libertinus for the preprint.
FIGURE_FONT_CONFIGS = {
    "submission": {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes"],
        "font.monospace": ["Fira Mono"],
        "mathtext.fontset": "cm",
    },
    "preprint": {
        "font.family": "serif",
        "font.serif": ["Libertinus Serif"],
        "font.monospace": ["Libertinus Mono"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Libertinus Serif",
        "mathtext.it": "Libertinus Serif:italic",
        "mathtext.bf": "Libertinus Serif:bold",
    },
}


def register_bundled_fonts() -> None:
    """Register the bundled fonts with Matplotlib for this process."""
    for font_path in _FONT_PATHS:
        font_manager.fontManager.addfont(font_path)
