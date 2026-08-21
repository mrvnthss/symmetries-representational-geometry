# Bundled figure fonts

These fonts are bundled so the paper figures render reproducibly without
depending on fonts installed on the host system.
`figures/appendix/fig_a1.py` registers them with Matplotlib at runtime via
`matplotlib.font_manager.fontManager.addfont`.

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
