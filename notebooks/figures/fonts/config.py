"""Bundled fonts and font configuration for the paper figures.

The appendix PDFs embed their fonts. Registering the bundled faces keeps
figure rendering independent of the fonts installed on the host system.
"""

__all__ = [
    "FIGURE_FONT_CONFIG",
    "register_bundled_fonts",
]

from pathlib import Path

from matplotlib import font_manager

_FONT_DIR = Path(__file__).resolve().parent

_FONT_PATHS = (
    _FONT_DIR / "fira" / "FiraMono-Regular.otf",
    _FONT_DIR / "tex-gyre" / "TeXGyreTermes-Regular.otf",
)

# Times-compatible text with Computer Modern mathematics.
FIGURE_FONT_CONFIG = {
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes"],
    "font.monospace": ["Fira Mono"],
    "mathtext.fontset": "cm",
}


def register_bundled_fonts() -> None:
    """Register the bundled fonts with Matplotlib for this process."""
    for font_path in _FONT_PATHS:
        font_manager.fontManager.addfont(font_path)
