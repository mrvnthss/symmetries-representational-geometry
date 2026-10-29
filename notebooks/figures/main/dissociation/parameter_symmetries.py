import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import math
    import re
    from dataclasses import dataclass
    from dataclasses import replace
    from pathlib import Path
    from typing import Literal

    import marimo as mo

    from symmetries import colors
    from symmetries import svg

    # ── PATHS AND TEMPLATE ────────────────────────────────────────────

    TEMPLATE_MANUAL_PATH = Path("../figures/templates/manual/hidden_unit.svg")
    OUTPUT_DIR = Path("../figures/main/dissociation/symmetries")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TEMPLATE_SVG = TEMPLATE_MANUAL_PATH.read_text()
    TEMPLATE_WIDTH_MM, TEMPLATE_HEIGHT_MM = svg.parse_svg_dimensions(
        TEMPLATE_SVG
    )

    # Re-export from svg module for use in notebook functions
    NAMESPACES = svg.NAMESPACES
    ColorSpec = svg.ColorSpec

    # ── DATACLASSES ───────────────────────────────────────────────────

    @dataclass(frozen=True)
    class LayoutParams:
        """Parameters controlling the stacked neuron layout."""

        angle_deg: float
        distance: float
        additional_gap: float
        dot_radius: float
        dot_spacing: float
        dot_offset: float
        horizontal_offset: float
        vertical_offset: float
        padding: float = 5.0

    @dataclass(frozen=True)
    class StackGeometry:
        """Computed geometry for a stack of neurons."""

        positions: list[tuple[float, float]]
        dot_positions: list[tuple[float, float]]
        min_x: float
        max_x: float
        min_y: float
        max_y: float
        width: float
        height: float

    @dataclass(frozen=True)
    class RenderConfig:
        """Configuration for rendering a single stack or neuron."""

        name: str
        filename: str
        color_map: dict[str, ColorSpec]
        activation: str | None
        is_individual: bool

    # ── COLOR PALETTE ─────────────────────────────────────────────────
    #
    # Organized by parameter type and sign:
    #   - INCOMING: weights into the neuron (w_in_1, w_in_2, w_in_3, bias)
    #   - OUTGOING: weights out of the neuron (w_out_1, w_out_2)
    #
    # Incoming parameters have four variants:
    #   - REF_POS: Reference-tied, positive sign (+w, +b) — Orange
    #   - REF_NEG: Reference-tied, negative sign (-w, -b) — Teal
    #   - ARB_POS: Arbitrary, positive sign (+w', +b')    — Green
    #   - ARB_NEG: Arbitrary, negative sign (-w', -b')    — Purple
    #   - DEAD:    Zero weights (w=0)                     — Grey
    #
    # Lightness levels: L2 (dark) → L3 (medium) → L4 (light)

    def _incoming_palette(hue: str) -> dict[str, str]:
        """Build incoming weight palette for a given hue."""
        return {
            "w_in_1": colors.get_color(hue, "L2"),
            "w_in_2": colors.get_color(hue, "L3"),
            "w_in_3": colors.get_color(hue, "L4"),
            "bias": colors.get_color(hue, "L3"),
        }

    COLORS = {
        # ── Incoming: Reference-tied, positive (+w, +b) — Orange ──────
        "in_ref_pos": _incoming_palette("orange"),
        # ── Incoming: Reference-tied, negative (-w, -b) — Teal ────────
        "in_ref_neg": _incoming_palette("teal"),
        # ── Incoming: Arbitrary, positive (+w', +b') — Green ──────────
        "in_arb_pos": _incoming_palette("green"),
        # ── Incoming: Arbitrary, negative (-w', -b') — Purple ─────────
        "in_arb_neg": _incoming_palette("purple"),
        # ── Incoming: Dead / Zero (w=0) — Grey ────────────────────────
        "in_dead": _incoming_palette("grey"),
        # ── Outgoing (individual neurons) — Blue ──────────────────────
        "out": {
            "w_out_1": colors.get_color("blue", "L2"),
            "w_out_2": colors.get_color("blue", "L4"),
        },
        # ── Outgoing (groups) — per-neuron shades of blue ─────────────
        "out_per_neuron": {
            "w_out_1": (
                colors.get_color("blue", "L2"),
                colors.get_color("blue", "L2"),
                colors.get_color("blue", "L3"),
            ),
            "w_out_2": (
                colors.get_color("blue", "L3"),
                colors.get_color("blue", "L1"),
                colors.get_color("blue", "L4"),
            ),
        },
        # ── Soma, outline, activation & dots ──────────────────────────
        "soma": colors.get_color("grey", "L3"),
        "outline": "#ffffff",
        "activation": colors.get_color("red", "L1"),
        "dots": "#000000",
    }
    COLORS["in_dead"]["bias"] = colors.get_color("green", "L3")

    # Shorthand palette references for SYMMETRY_SPECS
    INCOMING_PALETTES = {
        "ref_pos": COLORS["in_ref_pos"],
        "ref_neg": COLORS["in_ref_neg"],
        "arb_pos": COLORS["in_arb_pos"],
        "arb_neg": COLORS["in_arb_neg"],
        "dead": COLORS["in_dead"],
    }

    # ── LAYOUT CONSTANTS ──────────────────────────────────────────────

    # Layout parameters differing between single- and double-stack figures
    SINGLE_STACK_ANGLE_DEG = 52
    DOUBLE_STACK_ANGLE_DEG = 60
    SINGLE_STACK_DISTANCE = 7
    DOUBLE_STACK_DISTANCE = 9

    # Default (shared) layout parameters
    DEFAULT_LAYOUT = LayoutParams(
        angle_deg=SINGLE_STACK_ANGLE_DEG,
        distance=SINGLE_STACK_DISTANCE,
        additional_gap=12,
        dot_radius=0.4,
        dot_spacing=1.5,
        dot_offset=-1.0,
        horizontal_offset=-16,
        vertical_offset=6,
        padding=5,
    )

    # Per-variant layouts
    SINGLE_STACK_LAYOUT = DEFAULT_LAYOUT
    DOUBLE_STACK_LAYOUT = replace(
        DEFAULT_LAYOUT,
        angle_deg=DOUBLE_STACK_ANGLE_DEG,
        distance=DOUBLE_STACK_DISTANCE,
    )

    # Shorthand accessors for default layout values (used by UI sliders)
    STACK_ANGLE_DEG = DEFAULT_LAYOUT.angle_deg
    STACK_DISTANCE = DEFAULT_LAYOUT.distance
    STACK_EXTRA_GAP = DEFAULT_LAYOUT.additional_gap
    DOT_RADIUS = DEFAULT_LAYOUT.dot_radius
    DOT_SPACING = DEFAULT_LAYOUT.dot_spacing
    DOT_OFFSET = DEFAULT_LAYOUT.dot_offset
    HORIZONTAL_OFFSET = DEFAULT_LAYOUT.horizontal_offset
    VERTICAL_OFFSET = DEFAULT_LAYOUT.vertical_offset

    # ── SYMMETRY TYPES ────────────────────────────────────────────────
    #
    # Each symmetry type defines how neurons in a stack are colored.
    # For stacks (families), we have 3 neurons: front (0), middle (1), back (2).
    #
    # Activation-independent symmetries:
    #   - "reference":       Single reference neuron (w, b, a*)
    #   - "duplicate_group": K≥2 neurons, same (w,b), Σa = a*
    #   - "zero_group":      K≥1 neurons, arbitrary (w',b'), Σa = 0
    #   - "constant":        Individual neuron, w=0 (dead)
    #   - "rescaling":       Individual neuron, scaled (w, b, a)
    #
    # Activation-dependent symmetries (aligned/opposite groups):
    #   - "linear_duplicate_group":   Even-linear, replicates reference
    #   - "linear_group":             Even-linear, arbitrary incoming
    #   - "constant_duplicate_group": Constant-odd, replicates reference
    #   - "constant_group":           Constant-odd, arbitrary incoming

    SymmetryType = Literal[
        "reference",
        "duplicate_group",
        "zero_group",
        "constant",
        "rescaling",
        "linear_duplicate_group",
        "linear_group",
        "constant_duplicate_group",
        "constant_group",
    ]

    # ── SYMMETRY SPECIFICATIONS ───────────────────────────────────────
    #
    # Data-driven configuration for each symmetry type. Each entry defines:
    #   - name: Human-readable name (suffix appended for multi-stack)
    #   - filename_stem: Base filename (suffix appended for multi-stack)
    #   - activation: Which activation to display
    #   - is_individual: True if single neuron (no stack)
    #   - stacks: List of (suffix, incoming_palette) tuples

    SYMMETRY_SPECS: dict[SymmetryType, dict] = {
        "reference": {
            "name": "Reference neuron",
            "filename_stem": "reference",
            "activation": "elu",
            "is_individual": True,
            "stacks": [("", "ref_pos")],
        },
        "duplicate_group": {
            "name": "Duplicate group",
            "filename_stem": "duplicate_group",
            "activation": "elu",
            "is_individual": False,
            "stacks": [("", "ref_pos")],
        },
        "zero_group": {
            "name": "Zero group",
            "filename_stem": "zero_group",
            "activation": "elu",
            "is_individual": False,
            "stacks": [("", "arb_pos")],
        },
        "constant": {
            "name": "Constant neuron",
            "filename_stem": "constant",
            "activation": "constant",
            "is_individual": True,
            "stacks": [("", "dead")],
        },
        "rescaling": {
            "name": "Rescaling",
            "filename_stem": "rescaling",
            "activation": "relu",
            "is_individual": True,
            "stacks": [("", "ref_pos")],
        },
        "linear_duplicate_group": {
            "name": "Linear-duplicate group",
            "filename_stem": "linear_duplicate_group",
            "activation": "relu",
            "is_individual": False,
            "stacks": [("aligned", "ref_pos"), ("opposite", "ref_neg")],
        },
        "linear_group": {
            "name": "Linear group",
            "filename_stem": "linear_group",
            "activation": "relu",
            "is_individual": False,
            "stacks": [("aligned", "arb_pos"), ("opposite", "arb_neg")],
        },
        "constant_duplicate_group": {
            "name": "Constant-duplicate group",
            "filename_stem": "constant_duplicate_group",
            "activation": "tanh",
            "is_individual": False,
            "stacks": [("aligned", "ref_pos"), ("opposite", "ref_neg")],
        },
        "constant_group": {
            "name": "Constant group",
            "filename_stem": "constant_group",
            "activation": "tanh",
            "is_individual": False,
            "stacks": [("aligned", "arb_pos"), ("opposite", "arb_neg")],
        },
    }


@app.function
def compute_stack_geometry(layout: LayoutParams) -> StackGeometry:
    """Compute positions and bounds for a stack of three neurons.

    Args:
        layout: Layout parameters controlling spacing and angle.

    Returns:
        StackGeometry with positions, dot positions, and bounds.
    """
    angle_rad = math.radians(layout.angle_deg)
    dx = layout.distance * math.cos(angle_rad)
    dy = -layout.distance * math.sin(angle_rad)

    total_gap = layout.distance + layout.additional_gap
    dx_gap = total_gap * math.cos(angle_rad)
    dy_gap = -total_gap * math.sin(angle_rad)

    positions = [
        (0.0, 0.0),
        (dx, dy),
        (dx + dx_gap, dy + dy_gap),
    ]

    # Center of gap between neuron 2 and 3 (including neuron center
    # offset). ``dot_offset`` nudges the dots horizontally.
    center_x = dx + dx_gap / 2 + TEMPLATE_WIDTH_MM / 2 + layout.dot_offset
    center_y = dy + dy_gap / 2 + TEMPLATE_HEIGHT_MM / 2

    # Three dots centered at the gap
    dot_positions = [
        (
            center_x + i * layout.dot_spacing * math.cos(angle_rad),
            center_y - i * layout.dot_spacing * math.sin(angle_rad),
        )
        for i in [-1, 0, 1]
    ]

    # Compute bounding box
    all_x = [p[0] for p in positions] + [
        p[0] + TEMPLATE_WIDTH_MM for p in positions
    ]
    all_y = [p[1] for p in positions] + [
        p[1] + TEMPLATE_HEIGHT_MM for p in positions
    ]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    return StackGeometry(
        positions=positions,
        dot_positions=dot_positions,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        width=max_x - min_x,
        height=max_y - min_y,
    )


@app.function
def render_stack(
    geometry: StackGeometry,
    config: RenderConfig,
    *,
    id_prefix: str,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    dot_color: str = "#808080",
    dot_radius: float = 0.5,
) -> tuple[str, str, str]:
    """Render a single stack of neurons.

    Args:
        geometry: Precomputed stack geometry.
        config: Render configuration (colors, activation).
        id_prefix: Prefix for unique IDs (e.g., "s0" for stack 0).
        offset_x: X offset for positioning in combined SVG.
        offset_y: Y offset for positioning in combined SVG.
        dot_color: Color for continuation dots.
        dot_radius: Radius of continuation dots.

    Returns:
        Tuple of (defs_content, dots_svg, neurons_svg).
    """
    svg_base = svg.set_activation(TEMPLATE_SVG, config.activation)

    defs_content = ""
    neurons_svg = ""

    # Render neurons back-to-front (reversed order)
    for idx, pos in enumerate(reversed(geometry.positions)):
        neuron_idx = len(geometry.positions) - 1 - idx

        # Apply per-neuron color mapping
        svg_neuron = svg_base
        if config.color_map:
            svg_neuron = svg.apply_color_mapping(
                svg_base,
                config.color_map,
                neuron_idx,
            )

        svg_unique = svg.make_unique_ids(
            svg_neuron, f"{id_prefix}_n{neuron_idx}"
        )
        defs, body = svg.extract_svg_body(svg_unique)

        # Extract defs content
        match = re.search(r"<defs[^>]*>(.*?)</defs>", defs, re.DOTALL)
        if match:
            defs_content += match.group(1)

        x = pos[0] + offset_x
        y = pos[1] + offset_y
        neurons_svg += f'<g transform="translate({x}, {y})">\n{body}\n</g>\n'

    # Render dots
    dots_svg = "".join(
        f'<circle cx="{dot_x + offset_x}" cy="{dot_y + offset_y}" '
        f'r="{dot_radius}" fill="{dot_color}"/>\n'
        for dot_x, dot_y in geometry.dot_positions
    )

    return defs_content, dots_svg, neurons_svg


@app.function
def wrap_svg_document(
    *,
    width_mm: float,
    height_mm: float,
    defs_content: str,
    dots_svg: str,
    neurons_svg: str,
) -> str:
    """Wrap SVG content in a complete SVG document.

    Args:
        width_mm: Document width in millimeters.
        height_mm: Document height in millimeters.
        defs_content: Content for the <defs> section.
        dots_svg: SVG content for the dots layer.
        neurons_svg: SVG content for the neurons layer.

    Returns:
        Complete SVG document as a string.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg
   width="{width_mm}mm"
   height="{height_mm}mm"
   viewBox="0 0 {width_mm} {height_mm}"
   version="1.1"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd">
   <defs>{defs_content}</defs>
   <g inkscape:label="neurons" inkscape:groupmode="layer">
       {neurons_svg}
   </g>
   <g inkscape:label="dots" inkscape:groupmode="layer">
       {dots_svg}
   </g>
</svg>"""


@app.function
def build_single_neuron_svg(
    color_map: dict[str, str] | None = None,
    activation: str | None = "tanh",
) -> str:
    """Build SVG with a single neuron (no stacking).

    Args:
        color_map: Dict mapping inkscape:label names to stroke colors.
        activation: Which activation to display, or None to hide all.
          Options: "constant", "elu", "tanh", "relu".

    Returns:
        SVG content string.
    """
    svg_content = svg.set_activation(TEMPLATE_SVG, activation)

    # Apply color mapping (neuron_idx=0 for single neuron)
    if color_map:
        svg_content = svg.apply_color_mapping(
            svg_content,
            color_map,
            neuron_idx=0,
        )

    return svg_content


@app.function
def build_stacked_neurons_svg(
    layout: LayoutParams,
    config: RenderConfig,
    *,
    dot_color: str = "#808080",
) -> str:
    """Build SVG with stacked neurons along a diagonal.

    Args:
        layout: Layout parameters controlling spacing and angle.
        config: Render configuration (colors, activation).
        dot_color: Color of the continuation dots.

    Returns:
        Complete SVG document string.
    """
    geometry = compute_stack_geometry(layout)

    # Compute canvas dimensions with padding
    canvas_width = geometry.width + 2 * layout.padding
    canvas_height = geometry.height + 2 * layout.padding
    offset_x = -geometry.min_x + layout.padding
    offset_y = -geometry.min_y + layout.padding

    defs_content, dots_svg, neurons_svg = render_stack(
        geometry,
        config,
        id_prefix="n",
        offset_x=offset_x,
        offset_y=offset_y,
        dot_color=dot_color,
        dot_radius=layout.dot_radius,
    )

    return wrap_svg_document(
        width_mm=canvas_width,
        height_mm=canvas_height,
        defs_content=defs_content,
        dots_svg=dots_svg,
        neurons_svg=neurons_svg,
    )


@app.function
def save_figure(svg_content: str, filename: str) -> Path:
    """Save SVG content to file."""
    svg_path = OUTPUT_DIR / filename
    svg_path.write_text(svg_content)
    return svg_path


@app.function
def build_combined_stacks_svg(
    layout: LayoutParams,
    configs: list[RenderConfig],
    *,
    dot_color: str = "#808080",
) -> str:
    """Build a single SVG containing multiple stacks side by side.

    Args:
        layout: Layout parameters controlling spacing and angle.
        configs: List of RenderConfig objects from
          get_symmetry_configs().
        dot_color: Color of the continuation dots.

    Returns:
        Combined SVG document string.
    """
    if len(configs) == 1:
        return build_stacked_neurons_svg(
            layout,
            configs[0],
            dot_color=dot_color,
        )

    geometry = compute_stack_geometry(layout)

    # Canvas dimensions for combined SVG
    num_stacks = len(configs)
    canvas_width = (
        num_stacks * geometry.width
        + (num_stacks - 1) * layout.horizontal_offset
        + 2 * layout.padding
    )
    total_v_shift = (num_stacks - 1) * layout.vertical_offset
    canvas_height = geometry.height + 2 * layout.padding + abs(total_v_shift)

    # If stacks shift upward (negative offset), push all stacks down
    # so the topmost stack stays within the canvas.
    base_y_adjust = max(0.0, -total_v_shift)

    all_defs_content = ""
    all_dots_svg = ""
    all_neurons_svg = ""

    for stack_idx, config in enumerate(configs):
        # Horizontal offset for this stack
        stack_offset_x = (
            stack_idx * (geometry.width + layout.horizontal_offset)
            + layout.padding
            - geometry.min_x
        )
        stack_offset_y = (
            stack_idx * layout.vertical_offset
            + base_y_adjust
            + layout.padding
            - geometry.min_y
        )

        defs, dots, neurons = render_stack(
            geometry,
            config,
            id_prefix=f"s{stack_idx}",
            offset_x=stack_offset_x,
            offset_y=stack_offset_y,
            dot_color=dot_color,
            dot_radius=layout.dot_radius,
        )

        all_defs_content += defs
        all_dots_svg += dots
        all_neurons_svg += neurons

    return wrap_svg_document(
        width_mm=canvas_width,
        height_mm=canvas_height,
        defs_content=all_defs_content,
        dots_svg=all_dots_svg,
        neurons_svg=all_neurons_svg,
    )


@app.function
def get_symmetry_configs(symmetry: SymmetryType) -> list[RenderConfig]:
    """Get color/activation configurations for a given symmetry type.

    Each symmetry may require one or more stacks. For example:
    - "duplicate_group" needs 1 stack (all aligned)
    - "linear_duplicate_group" needs 2 stacks (aligned + opposite)

    Args:
        symmetry: The type of symmetry to visualize.

    Returns:
        List of RenderConfig objects for the symmetry's stack(s).
    """
    try:
        spec = SYMMETRY_SPECS[symmetry]
    except KeyError as e:
        msg = f"Unknown symmetry type: {symmetry}"
        raise ValueError(msg) from e

    is_individual = spec["is_individual"]
    activation = spec["activation"]

    # Outgoing colors: single color for individuals, per-neuron for groups
    outgoing = COLORS["out"] if is_individual else COLORS["out_per_neuron"]

    # Common color elements
    common = {
        **outgoing,
        "soma": COLORS["soma"],
        "outline": COLORS["outline"],
    }
    if activation is not None:
        common[activation] = COLORS["activation"]

    return [
        RenderConfig(
            name=f"{spec['name']} ({suffix})" if suffix else spec["name"],
            filename=(
                f"{spec['filename_stem']}_{suffix}.svg"
                if suffix
                else f"{spec['filename_stem']}.svg"
            ),
            color_map={**INCOMING_PALETTES[incoming], **common},
            activation=spec["activation"],
            is_individual=is_individual,
        )
        for suffix, incoming in spec["stacks"]
    ]


@app.function
def render_symmetry(
    symmetry: SymmetryType,
    layout: LayoutParams,
    *,
    dot_color: str = "#808080",
) -> tuple[str, str]:
    """Render a symmetry type and return the SVG with filename.

    This is the high-level entrypoint for rendering symmetries. It
    handles the logic of deciding whether to render a single neuron,
    single stack, or combined multi-stack visualization.

    Args:
        symmetry: The type of symmetry to render.
        layout: Layout parameters controlling spacing and angle.
        dot_color: Color for continuation dots.

    Returns:
        Tuple of (filename, svg_content).
    """
    configs = get_symmetry_configs(symmetry)

    if configs[0].is_individual:
        # Single neuron (no stacking)
        svg_content = build_single_neuron_svg(
            color_map=configs[0].color_map,
            activation=configs[0].activation,
        )
        filename = configs[0].filename
    elif len(configs) == 1:
        # Single stack
        svg_content = build_stacked_neurons_svg(
            layout,
            configs[0],
            dot_color=dot_color,
        )
        filename = configs[0].filename
    else:
        # Multi-stack: build combined SVG
        svg_content = build_combined_stacks_svg(
            layout,
            configs,
            dot_color=dot_color,
        )
        # Use symmetry name for combined filename
        filename = f"{symmetry}.svg"

    return filename, svg_content


@app.function
def generate_all_symmetry_svgs(
    single_layout: LayoutParams | None = None,
    double_layout: LayoutParams | None = None,
    *,
    dot_color: str = "#808080",
) -> dict[str, str]:
    """Generate SVGs for all 9 symmetry types.

    Single-stack figures (duplicate / zero group, plus the individual
    neurons) are rendered with ``single_layout``; double-stack figures
    (linear / constant groups) with ``double_layout``. The two layouts
    are intended to differ only in the stacking angle.

    Args:
        single_layout: Layout for single-stack and individual figures.
          Defaults to SINGLE_STACK_LAYOUT.
        double_layout: Layout for double-stack figures. Defaults to
          DOUBLE_STACK_LAYOUT.
        dot_color: Color for continuation dots.

    Returns:
        Dict mapping filename to SVG content.
    """
    if single_layout is None:
        single_layout = SINGLE_STACK_LAYOUT
    if double_layout is None:
        double_layout = DOUBLE_STACK_LAYOUT

    all_symmetries: list[SymmetryType] = [
        "reference",
        "duplicate_group",
        "zero_group",
        "constant",
        "rescaling",
        "linear_duplicate_group",
        "linear_group",
        "constant_duplicate_group",
        "constant_group",
    ]

    result: dict[str, str] = {}
    for symmetry in all_symmetries:
        # Double-stack symmetries need two stacks; everything else (single
        # stacks and individual neurons) uses the single-stack layout.
        is_double = len(get_symmetry_configs(symmetry)) > 1
        layout = double_layout if is_double else single_layout
        filename, svg_content = render_symmetry(
            symmetry, layout, dot_color=dot_color
        )
        result[filename] = svg_content
    return result


@app.cell
def _():
    # ── Layout sliders (defaults from layout constants) ───────────────
    angle_slider = mo.ui.slider(
        20,
        70,
        value=STACK_ANGLE_DEG,
        step=1,
        label="Angle (deg)",
        show_value=True,
    )
    distance_slider = mo.ui.slider(
        5,
        30,
        value=STACK_DISTANCE,
        step=0.1,
        label="Distance",
        show_value=True,
    )
    gap_slider = mo.ui.slider(
        10,
        60,
        value=STACK_EXTRA_GAP,
        step=1,
        label="Extra gap",
        show_value=True,
    )
    dot_radius_slider = mo.ui.slider(
        0.3,
        2.0,
        value=DOT_RADIUS,
        step=0.1,
        label="Dot radius",
        show_value=True,
    )
    dot_spacing_slider = mo.ui.slider(
        1,
        10,
        value=DOT_SPACING,
        step=0.5,
        label="Dot spacing",
        show_value=True,
    )
    dot_offset_slider = mo.ui.slider(
        -30,
        30,
        value=DOT_OFFSET,
        step=0.5,
        label="Dot offset",
        show_value=True,
    )
    h_offset_slider = mo.ui.slider(
        -30,
        30,
        value=HORIZONTAL_OFFSET,
        step=1,
        label="Horizontal offset",
        show_value=True,
    )
    v_offset_slider = mo.ui.slider(
        -30,
        30,
        value=VERTICAL_OFFSET,
        step=1,
        label="Vertical offset",
        show_value=True,
    )

    # ── Symmetry selector ─────────────────────────────────────────────
    symmetry_dropdown = mo.ui.dropdown(
        options={
            "Reference neuron": "reference",
            "Duplicate group": "duplicate_group",
            "Zero group": "zero_group",
            "Constant neuron (w=0)": "constant",
            "Rescaling": "rescaling",
            "Linear-duplicate (even-linear)": "linear_duplicate_group",
            "Linear (even-linear)": "linear_group",
            "Constant-duplicate (constant-odd)": "constant_duplicate_group",
            "Constant (constant-odd)": "constant_group",
        },
        value="Duplicate group",
        label="Symmetry type",
    )

    mo.vstack(
        [
            symmetry_dropdown,
            mo.hstack(
                [
                    angle_slider,
                    distance_slider,
                    gap_slider,
                    dot_radius_slider,
                ]
            ),
            mo.hstack(
                [
                    dot_spacing_slider,
                    dot_offset_slider,
                    h_offset_slider,
                    v_offset_slider,
                ]
            ),
        ]
    )
    return (
        angle_slider,
        distance_slider,
        dot_offset_slider,
        dot_radius_slider,
        dot_spacing_slider,
        gap_slider,
        h_offset_slider,
        symmetry_dropdown,
        v_offset_slider,
    )


@app.cell
def _(
    angle_slider,
    distance_slider,
    dot_offset_slider,
    dot_radius_slider,
    dot_spacing_slider,
    gap_slider,
    h_offset_slider,
    symmetry_dropdown,
    v_offset_slider,
):
    # Build layout from current slider values
    preview_layout = LayoutParams(
        angle_deg=angle_slider.value,
        distance=distance_slider.value,
        additional_gap=gap_slider.value,
        dot_radius=dot_radius_slider.value,
        dot_spacing=dot_spacing_slider.value,
        dot_offset=dot_offset_slider.value,
        horizontal_offset=h_offset_slider.value,
        vertical_offset=v_offset_slider.value,
    )

    # Render the selected symmetry
    _, preview_svg = render_symmetry(
        symmetry_dropdown.value,
        preview_layout,
        dot_color=COLORS["dots"],
    )
    mo.Html(preview_svg)
    return


@app.cell
def _():
    # ── Generate and save all symmetry figures ────────────────────────
    all_svgs = generate_all_symmetry_svgs(dot_color=COLORS["dots"])
    for filename, svg_content in all_svgs.items():
        save_figure(svg_content, filename)
    return


if __name__ == "__main__":
    app.run()
