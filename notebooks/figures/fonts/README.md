# Bundled figure fonts

These fonts are bundled so the paper figures render reproducibly without
depending on fonts installed on the host system.
`config.py` registers them with Matplotlib at runtime via
`matplotlib.font_manager.fontManager.addfont`, and holds the per-venue
font configuration used to export each figure twice.

## Fira Mono

- File: `fira/FiraMono-Regular.otf`
- Version: 3.201
- Upstream: <https://github.com/mozilla/Fira>
- License: SIL Open Font License 1.1 or later; see `fira/OFL.txt`

## Libertinus

- Files: `libertinus/LibertinusMono-Regular.otf` and the regular, italic, and
  bold faces of Libertinus Serif
- Version: 7.051
- Upstream: <https://github.com/alerque/libertinus>
- License: SIL Open Font License 1.1; see `libertinus/OFL.txt`

## TeX Gyre Termes

- File: `tex-gyre/TeXGyreTermes-Regular.otf`
- Version: 2.004
- Upstream: <https://www.gust.org.pl/projects/e-foundry/tex-gyre/termes>
- License: GUST Font License; see `tex-gyre/GUST-FONT-LICENSE.txt`
