# Bundled figure fonts

These fonts are bundled so the paper figures render reproducibly without
depending on fonts installed on the host system.
`config.py` registers them with Matplotlib at runtime via
`matplotlib.font_manager.fontManager.addfont`, and defines the font
configuration used for the appendix PDFs: TeX Gyre Termes for text,
Fira Mono for monospace, and Computer Modern for mathematics.

## Fira Mono

- File: `fira/FiraMono-Regular.otf`
- Version: 3.201
- Upstream: <https://github.com/mozilla/Fira>
- License: SIL Open Font License 1.1 or later; see `fira/OFL.txt`

## TeX Gyre Termes

- File: `tex-gyre/TeXGyreTermes-Regular.otf`
- Version: 2.004
- Upstream: <https://www.gust.org.pl/projects/e-foundry/tex-gyre/termes>
- License: GUST Font License; see `tex-gyre/GUST-FONT-LICENSE.txt`
