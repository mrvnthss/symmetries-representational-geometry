"""SVG manipulation utilities for Inkscape-based figure generation.

This module provides functions for programmatic manipulation of SVG
files, particularly those created with Inkscape. It supports:

* ID deduplication for combining multiple SVG instances
* Stroke color manipulation and colorization
* Finding elements by Inkscape labels
* Activation function visibility toggling
* Dimension parsing from SVG attributes

Functions:
---------

* :func:`make_unique_ids`: Add suffix to all IDs to avoid conflicts.
* :func:`extract_svg_body`: Extract defs and body from SVG content.
* :func:`set_stroke_color`: Replace or add stroke in a style string.
* :func:`colorize_element`: Set stroke color on a leaf element.
* :func:`colorize_group_recursive`: Recursively colorize leaf elements.
* :func:`find_labeled_elements`: Find elements by Inkscape label.
* :func:`apply_color_mapping`: Apply colors to elements by label.
* :func:`set_activation`: Set which activation function to display.
* :func:`parse_svg_dimensions`: Parse width/height from SVG content.

Notes:
-----
These utilities are designed for Inkscape SVGs that use inkscape:label
attributes for semantic grouping of elements.
"""

__all__ = [
    "NAMESPACES",
    "apply_color_mapping",
    "colorize_element",
    "colorize_group_recursive",
    "extract_svg_body",
    "find_labeled_elements",
    "make_unique_ids",
    "parse_svg_dimensions",
    "set_activation",
    "set_stroke_color",
]

import re
import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DefusedET

# Type alias for color specifications (single color or per-neuron tuple)
type ColorSpec = str | tuple[str, str, str]

# XML namespaces used in Inkscape SVGs
NAMESPACES = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "xlink": "http://www.w3.org/1999/xlink",
}

# Register namespaces to preserve them on output
for _prefix, _uri in NAMESPACES.items():
    ET.register_namespace(_prefix if _prefix != "svg" else "", _uri)


def make_unique_ids(svg_content: str, suffix: str) -> str:
    """Add suffix to all IDs and references to avoid conflicts.

    When combining multiple SVG instances into a single document, IDs
    must be unique. This function appends a suffix to all id attributes
    and updates all references (url(#...), xlink:href, href).

    Args:
        svg_content: The SVG content as a string.
        suffix: The suffix to append to each ID.

    Returns:
        Modified SVG content with unique IDs.
    """
    ids = re.findall(r'id="([^"]+)"', svg_content)
    result = svg_content
    for old_id in set(ids):
        new_id = f"{old_id}_{suffix}"
        result = result.replace(f'id="{old_id}"', f'id="{new_id}"')
        result = result.replace(f"url(#{old_id})", f"url(#{new_id})")
        result = result.replace(
            f'xlink:href="#{old_id}"', f'xlink:href="#{new_id}"'
        )
        result = result.replace(f'href="#{old_id}"', f'href="#{new_id}"')
    return result


def extract_svg_body(svg_content: str) -> tuple[str, str]:
    """Extract defs and body from SVG content.

    Parses SVG content to separate the <defs> section (containing
    reusable definitions like gradients) from the main body content
    (typically the Layer 1 group in Inkscape SVGs).

    Args:
        svg_content: The SVG content as a string.

    Returns:
        A tuple of (defs, body) where:
        - defs: The complete <defs>...</defs> section, or empty string
        - body: The main content (Layer 1 group or cleaned inner)
    """
    svg_start = svg_content.find("<svg")
    svg_open_end = svg_content.find(">", svg_start) + 1
    svg_close = svg_content.rfind("</svg>")
    inner = svg_content[svg_open_end:svg_close]

    defs_match = re.search(r"<defs[^>]*>.*?</defs>", inner, re.DOTALL)
    defs = defs_match.group(0) if defs_match else ""

    inner_clean = re.sub(
        r"<sodipodi:namedview[^>]*>.*?</sodipodi:namedview>",
        "",
        inner,
        flags=re.DOTALL,
    )

    layer_match = re.search(
        r'<g[^>]*inkscape:label="Layer 1"[^>]*>.*</g>',
        inner_clean,
        re.DOTALL,
    )
    body = layer_match.group(0) if layer_match else inner_clean

    return defs, body


def set_stroke_color(style: str, color: str) -> str:
    """Replace or add stroke color in a style string.

    Args:
        style: CSS-like style string (e.g., "stroke:#000;fill:none").
        color: The new stroke color value.

    Returns:
        Modified style string with the new stroke color.
    """
    if "stroke:" in style:
        return re.sub(r"stroke:[^;]+", f"stroke:{color}", style)
    return f"stroke:{color};" + style


def colorize_element(elem: ET.Element, color: str) -> None:
    """Set stroke color on a leaf SVG element.

    Modifies the element in place, either updating the style attribute
    or setting the stroke attribute directly.

    Args:
        elem: The SVG element to colorize.
        color: The stroke color to apply.
    """
    style = elem.get("style", "")
    if style:
        elem.set("style", set_stroke_color(style, color))
    else:
        elem.set("stroke", color)


def colorize_group_recursive(elem: ET.Element, color: str) -> None:
    """Recursively colorize all leaf elements within a group.

    Traverses the element tree and applies the specified color to all
    leaf elements (path, circle, rect, line, ellipse, polygon,
    polyline).

    Args:
        elem: The root element to start colorizing from.
        color: The stroke color to apply.
    """
    leaf_tags = {
        f"{{{NAMESPACES['svg']}}}{tag}"
        for tag in [
            "path",
            "circle",
            "rect",
            "line",
            "ellipse",
            "polygon",
            "polyline",
        ]
    }

    if elem.tag in leaf_tags:
        colorize_element(elem, color)
    else:
        for child in elem:
            colorize_group_recursive(child, color)


def find_labeled_elements(root: ET.Element, label: str) -> list[ET.Element]:
    """Find all elements with a given inkscape:label attribute.

    Recursively searches the element tree for elements matching the
    specified Inkscape label.

    Args:
        root: The root element to search from.
        label: The inkscape:label value to match.

    Returns:
        List of matching elements.
    """
    label_attr = f"{{{NAMESPACES['inkscape']}}}label"
    results: list[ET.Element] = []

    def search(elem: ET.Element) -> None:
        if elem.get(label_attr) == label:
            results.append(elem)
        for child in elem:
            search(child)

    search(root)
    return results


def _resolve_color(color_spec: ColorSpec, neuron_idx: int) -> str:
    """Resolve a color specification to a color for a neuron index."""
    if isinstance(color_spec, str):
        return color_spec
    return color_spec[neuron_idx]


def apply_color_mapping(
    svg_content: str,
    color_map: dict[str, ColorSpec],
    neuron_idx: int | None = None,
) -> str:
    """Apply stroke colors to elements/groups by their inkscape:label.

    Args:
        svg_content: The SVG content as a string.
        color_map: Dict mapping inkscape:label names to color values.
          Values can be a single color string (applied to all neurons)
          or a tuple of 3 colors (front, middle, back neuron).
        neuron_idx: If provided, resolve per-neuron colors for this
          index. If None, only single-color specs are applied.

    Returns:
        Modified SVG content with colors applied.
    """
    if not color_map:
        return svg_content

    root = DefusedET.fromstring(svg_content)

    for label, color_spec in color_map.items():
        # Resolve color for this neuron
        if isinstance(color_spec, tuple):
            if neuron_idx is None:
                continue  # skip per-neuron colors when no index provided
            color = _resolve_color(color_spec, neuron_idx)
        else:
            color = color_spec

        elements = find_labeled_elements(root, label)
        for elem in elements:
            colorize_group_recursive(elem, color)

    return ET.tostring(root, encoding="unicode")


def set_activation(svg_content: str, activation: str | None) -> str:
    """Set which activation function to display in the neuron soma.

    Controls visibility of activation function curves in the SVG by
    setting display:inline for the selected activation and display:none
    for all others.

    Args:
        svg_content: The SVG content as a string.
        activation: The activation to show ("constant", "elu", "tanh",
          "relu"), or None to hide all activations.

    Returns:
        Modified SVG content with the specified activation visible.
    """
    all_activations = ["constant", "elu", "tanh", "relu"]

    for act in all_activations:
        # Set display:none for all activations except the selected one
        show = act == activation
        display_val = "inline" if show else "none"

        # Replace display value in style for this activation's path
        # Match style attribute containing display, followed by the label
        pattern = (
            rf'(<path[^>]*style="[^"]*?)display:(none|inline)'
            rf'([^"]*"[^>]*inkscape:label="{act}")'
        )
        replacement = rf"\1display:{display_val}\3"
        svg_content = re.sub(pattern, replacement, svg_content)

        # Also handle case where label comes before style
        pattern = (
            rf'(<path[^>]*inkscape:label="{act}"[^>]*style="[^"]*?)'
            rf"display:(none|inline)"
        )
        replacement = rf"\1display:{display_val}"
        svg_content = re.sub(pattern, replacement, svg_content)

    return svg_content


def parse_svg_dimensions(svg_content: str) -> tuple[float, float]:
    """Parse width and height in mm from SVG content.

    Extracts dimensions from the SVG's width/height attributes
    (expecting mm units) or falls back to the viewBox attribute.

    Args:
        svg_content: The SVG content as a string.

    Returns:
        A tuple of (width, height) in millimeters.

    Raises:
        ValueError: If dimensions cannot be parsed from the SVG.
    """
    width_match = re.search(r'width="([\d.]+)mm"', svg_content)
    height_match = re.search(r'height="([\d.]+)mm"', svg_content)

    if width_match and height_match:
        return float(width_match.group(1)), float(height_match.group(1))

    # Fallback: try to get from viewBox
    viewbox_match = re.search(
        r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', svg_content
    )
    if viewbox_match:
        return float(viewbox_match.group(1)), float(viewbox_match.group(2))

    msg = "Could not parse SVG dimensions"
    raise ValueError(msg)
