r"""Page layout of the paper, for exporting figures at their final size.

The appendix figures are designed at a comfortable working size and
scaled down by ``\includegraphics[width=...\textwidth]`` in the paper.
Scaling the exported PDF to that width instead leaves the typeset
result unchanged, but gives the file the size it occupies on the page.
That matters for arXiv's HTML rendering, which sizes figures from the
file's own dimensions rather than from the LaTeX width.
"""

__all__ = [
    "TEXT_WIDTH_IN",
    "scale_pdf_to_text_width",
]

from pathlib import Path

from pypdf import PdfReader
from pypdf import PdfWriter

# \textwidth of the NeurIPS 2026 style, in inches.
TEXT_WIDTH_IN = 5.5

POINTS_PER_INCH = 72.0


def scale_pdf_to_text_width(path: Path, fraction: float) -> None:
    r"""Scale a one-page PDF in place to a fraction of the text width.

    Equivalent to ``\includegraphics[width=<fraction>\textwidth]``: the
    page and everything on it are scaled uniformly, so the aspect
    ratio, fonts and line widths come out exactly as LaTeX would
    typeset them at the design size.

    Args:
        path: The PDF to scale. It is overwritten.
        fraction: Target width as a fraction of ``TEXT_WIDTH_IN``.
    """
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        msg = f"expected a one-page PDF, got {len(reader.pages)} pages"
        raise ValueError(msg)
    page = reader.pages[0]

    target_width = fraction * TEXT_WIDTH_IN * POINTS_PER_INCH
    page.scale_by(target_width / float(page.mediabox.width))

    writer = PdfWriter()
    writer.add_page(page)
    if reader.metadata is not None:
        writer.add_metadata(reader.metadata)
    with path.open("wb") as file:
        writer.write(file)
