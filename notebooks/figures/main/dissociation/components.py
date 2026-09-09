import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors as mpl_colors
    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Affine2D

    from symmetries import colors
    from symmetries import distances
    from symmetries import mlp
    from symmetries import plotting
    from symmetries import rsa
    from symmetries import xor

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    OUTPUT_PATH = Path("../figures/main/dissociation/components")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Plotting constants ────────────────────────────────────────────
    plt.style.use("./style.mplstyle")
    # Bump font sizes
    plt.rcParams.update(
        {
            "axes.labelsize": 20,  # only relevant for XOR scatter
            "ytick.labelsize": 20,  # only relevant for colorbars
            "figure.dpi": 300,  # match dpi when saving
        }
    )

    # Axis limits
    LIM = 2

    # Colormaps and color ranges
    HEATMAP_CMAP = colors.get_diverging_cmap("blue", "red")
    HEATMAP_VMIN = -2.0
    HEATMAP_VMAX = 2.0
    RSM_CMAP = colors.get_sequential_cmap("green")
    ACTIVATION_CMAP = colors.get_sequential_cmap("purple")
    ACTIVATION_VMIN = 0.0
    ACTIVATION_VMAX = 8.0

    # Data point styling
    DATA_COLOR_0 = colors.get_color("yellow", "L4", "medium")
    DATA_COLOR_1 = colors.get_color("grey", "L3", "medium")

    # L-bracket marker: vertical arm on left + horizontal arm on bottom
    # of bbox, meeting at the bottom-left corner. Arms are 30% of bbox.
    # Only horizontal+vertical edges → no clash with the ±45° decision
    # boundaries in the heatmaps. Four rotations cover the four XOR
    # corners and share identical visual area (same shape rotated).
    _BRACKET_ARM = 0.30  # arm width as fraction of bbox
    # 8-vertex polyline tracing the L outline and explicitly returning
    # to the start vertex (matplotlib's marker renderer drops CLOSEPOLY
    # for custom paths, leaving the closing edge unstroked otherwise).
    # The start/end is placed at the MIDPOINT of the left edge so the
    # butt cap of the polyline lies along an edge instead of at a sharp
    # corner — without this, two overlapping caps at a corner produce a
    # visible stub artifact.
    _BRACKET_VERTS = np.array(
        [
            (-0.5, 0.0),  # midpoint of left edge
            (-0.5, 0.5),
            (-0.5 + _BRACKET_ARM, 0.5),
            (-0.5 + _BRACKET_ARM, -0.5 + _BRACKET_ARM),
            (0.5, -0.5 + _BRACKET_ARM),
            (0.5, -0.5),
            (-0.5, -0.5),
            (-0.5, 0.0),  # back to start
        ]
    )
    _BRACKET_PATH = MplPath(
        _BRACKET_VERTS,
        [MplPath.MOVETO] + [MplPath.LINETO] * 7,
    )

    def _bracket(rotation_deg: float) -> MarkerStyle:
        """L-bracket marker rotated CCW; 0° has its corner at bottom-left.

        Uses miter joins so the sharp inner/outer corners stay crisp
        (default join style would round/clip them).
        """
        return MarkerStyle(
            _BRACKET_PATH,
            transform=Affine2D().rotate_deg(rotation_deg),
            joinstyle="miter",
        )

    # Each bracket's corner aligns with its XOR data corner.
    DATA_MARKERS = [
        _bracket(0),  # (-1,-1) bottom-left
        _bracket(90),  # ( 1,-1) bottom-right
        _bracket(270),  # (-1, 1) top-left
        _bracket(180),  # ( 1, 1) top-right
    ]
    DATA_SIZE = 250
    DATA_EDGECOLOR = "black"
    DATA_LINEWIDTH = 0.9
    LABEL_OFFSET = 0.15
    LABEL_FONTSIZE = 16

    def colors_from_labels(labels) -> list[str]:
        """Derive point colors from class labels for consistent coloring."""
        return [DATA_COLOR_0 if lbl == 0 else DATA_COLOR_1 for lbl in labels]

    # RSM styling (all RSMs normalized to [0, 1])
    RSM_VMIN = 0.0
    RSM_VMAX = 1.0

    # SVG exports
    COMPONENT_SIZE = 2.0  # physical width (inches) of inner axes box

    # ── Layout constants ──────────────────────────────────────────────
    # Pads expressed as fractions of the inner-axes-box WIDTH so that
    # all SVG inner axes have identical physical size regardless of
    # which decorations (ticks/colorbar) are present.  These mirror the
    # defaults used by the helper plotting functions in
    # ``symmetries.plotting``.
    TICK_PAD_FRAC = 0.08  # gap between axes and tick area
    TICK_W_FRAC = 0.08  # width of tick-marker area
    CBAR_PAD_FRAC = 0.08  # gap between axes and colorbar
    CBAR_W_FRAC = 0.05  # width of colorbar
    CBAR_LABEL_PAD_IN = 0.30  # extra room (inches) for cbar tick labels

    def make_component_axes(
        *,
        n_rows: int = 1,
        n_cols: int = 1,
        has_left_ticks: bool = False,
        has_bottom_ticks: bool = False,
        has_colorbar: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a figure with a fixed-size inner axes box.

        The inner axes box is always ``COMPONENT_SIZE`` inches wide;
        height scales with ``n_rows / n_cols`` (for ``aspect='equal'``
        matrix/RSM plots).  Decorations sit at constant physical offsets
        outside the axes box, so point-sized markers render at identical
        physical sizes across all SVGs and remain consistent when
        composited (as long as the inner axes box is used as the
        alignment reference).
        """
        aw = COMPONENT_SIZE
        ah = aw * n_rows / n_cols

        left_in = (TICK_PAD_FRAC + TICK_W_FRAC) * aw if has_left_ticks else 0
        bottom_in = (
            (TICK_PAD_FRAC + TICK_W_FRAC) * aw if has_bottom_ticks else 0
        )
        right_in = (
            (CBAR_PAD_FRAC + CBAR_W_FRAC) * aw + CBAR_LABEL_PAD_IN
            if has_colorbar
            else 0
        )

        fw = left_in + aw + right_in
        fh = bottom_in + ah

        fig = plt.figure(figsize=(fw, fh), layout="none")
        ax = fig.add_axes((left_in / fw, bottom_in / fh, aw / fw, ah / fh))
        return fig, ax

    def save_component(fig: plt.Figure, path: Path) -> None:
        """Save SVG with predictable dimensions matching the figure."""
        fig.savefig(path, transparent=True)
        plt.close(fig)

    # ── Kwargs dicts ──────────────────────────────────────────────────
    HEATMAP_KWARGS = {
        "cmap": HEATMAP_CMAP,
        "lim": LIM,
        "vmin": HEATMAP_VMIN,
        "vmax": HEATMAP_VMAX,
        "rasterized": True,
    }
    DATA_KWARGS = {
        "cmap": mpl_colors.ListedColormap([DATA_COLOR_0, DATA_COLOR_1]),
        "markers": DATA_MARKERS,
        "s": DATA_SIZE,
        "edgecolors": DATA_EDGECOLOR,
        "linewidths": DATA_LINEWIDTH,
    }

    def rsm_kwargs_with_labels(labels) -> dict[str, str | dict]:
        """Build RSM plotting kwargs with label-consistent tick colors."""
        return {
            "tri": "lower",
            "imshow_kwargs": {
                "cmap": RSM_CMAP,
                "vmin": RSM_VMIN,
                "vmax": RSM_VMAX,
            },
            "tick_kwargs": {
                "markers": DATA_MARKERS,
                "colors": colors_from_labels(labels),
                "size": DATA_SIZE,
                "edgecolors": DATA_EDGECOLOR,
                "linewidths": DATA_LINEWIDTH,
            },
        }

    def activation_kwargs_with_labels(labels) -> dict[str, str | dict]:
        """Build hidden activations kwargs with label-consistent colors."""
        return {
            "imshow_kwargs": {
                "cmap": ACTIVATION_CMAP,
                "vmin": ACTIVATION_VMIN,
                "vmax": ACTIVATION_VMAX,
            },
            "tick_kwargs": rsm_kwargs_with_labels(labels)["tick_kwargs"],
        }

    HYPERPLANE_KWARGS = {
        "normal_lw": 2.0,  # uniform arrow width, independent of readouts
    }


@app.cell
def _():
    # ── XOR scatter plot ──────────────────────────────────────────────
    inputs, labels = xor.xor_dataset()

    # Export XOR scatter plot. Use make_component_axes so the inner axes
    # box is exactly COMPONENT_SIZE, matching the heatmaps/RSMs; data
    # markers then composite at a consistent size when panels are
    # aligned on their plot areas. The title, axis labels, and axis-end
    # arrows live outside the axes box and are picked up by the tight
    # bbox at save time.
    _fig, _ax = make_component_axes()
    plotting.plot_data(
        (inputs, labels),
        ax=_ax,
        **DATA_KWARGS,
    )

    _ax.set_xlim(-LIM, LIM)
    _ax.set_ylim(-LIM, LIM)
    _ax.set_aspect("equal")
    _ax.set_xlabel(r"$x_1$")
    _ax.set_ylabel(r"$x_2$")
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    # Draw arrows (triangles) at axis ends
    _ax.plot(1, 0, ">k", transform=_ax.transAxes, clip_on=False)
    _ax.plot(0, 1, "^k", transform=_ax.transAxes, clip_on=False)

    # Label data points. Each L-bracket marker's corner points outward
    # (away from the origin, into its own quadrant), so place the label
    # on the diagonally opposite side, toward the center, to sit in the
    # marker's open notch.
    for (_x, _y), _lbl in zip(inputs, labels, strict=True):
        _sx = 1.0 if _x < 0 else -1.0
        _sy = 1.0 if _y < 0 else -1.0
        _ax.annotate(
            f"{int(_lbl)}",
            (_x, _y),
            xytext=(_x + _sx * LABEL_OFFSET, _y + _sy * LABEL_OFFSET),
            fontsize=LABEL_FONTSIZE,
            ha="left" if _sx > 0 else "right",
            va="bottom" if _sy > 0 else "top",
        )
    _fig.savefig(
        OUTPUT_PATH / "xor.svg",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(_fig)

    # ── Network heatmaps ──────────────────────────────────────────────
    for _idx in range(len(xor.relu_solutions)):
        _fig, _ax = make_component_axes()
        plotting.plot_mlp(
            model=xor.relu_solutions[_idx],
            data=xor.xor_dataset(),
            heatmap_kwargs=HEATMAP_KWARGS,
            hyperplane_kwargs=HYPERPLANE_KWARGS,
            data_kwargs=DATA_KWARGS,
            ax=_ax,
        )
        _ax.set_xticks([])
        _ax.set_yticks([])
        save_component(_fig, OUTPUT_PATH / f"relu_{_idx + 1}.svg")

    # ── RSMs of ReLU solutions 1 and 4 ────────────────────────────────
    for _num in (1, 4):
        _hidden = jax.vmap(xor.relu_solutions[_num - 1].hidden_activations)(
            inputs
        )
        _rsm = rsa.calculate_rdm(
            _hidden,
            distances.dot_product,
            return_matrix=True,
        )
        _rsm = _rsm / _rsm.max()

        _fig, _ax = make_component_axes(
            has_left_ticks=True, has_bottom_ticks=True
        )
        plotting.plot_rdm(_rsm, **rsm_kwargs_with_labels(labels), ax=_ax)
        save_component(_fig, OUTPUT_PATH / f"rsm_{_num}.svg")

    # ── Overparameterized heatmaps ────────────────────────────────────
    OP_IDX = [1, 4]  # see xor.py
    for _idx in range(len(xor.overparam_solutions)):
        _fig, _ax = make_component_axes()
        plotting.plot_mlp(
            model=xor.overparam_solutions[_idx],
            data=xor.xor_dataset(),
            heatmap_kwargs=HEATMAP_KWARGS,
            hyperplane_kwargs=HYPERPLANE_KWARGS,
            data_kwargs=DATA_KWARGS,
            ax=_ax,
        )
        _ax.set_xticks([])
        _ax.set_yticks([])
        save_component(_fig, OUTPUT_PATH / f"relu_{OP_IDX[_idx]}_op.svg")

    # ── Overparameterized RSMs ────────────────────────────────────────
    for _idx in range(len(xor.overparam_solutions)):
        _hidden = jax.vmap(xor.overparam_solutions[_idx].hidden_activations)(
            inputs
        )
        _rsm = rsa.calculate_rdm(
            _hidden,
            distances.dot_product,
            return_matrix=True,
        )
        _rsm = _rsm / _rsm.max()

        _fig, _ax = make_component_axes(
            has_left_ticks=True, has_bottom_ticks=True
        )
        plotting.plot_rdm(_rsm, **rsm_kwargs_with_labels(labels), ax=_ax)
        save_component(_fig, OUTPUT_PATH / f"rsm_{OP_IDX[_idx]}_op.svg")
    return


@app.cell
def _():
    # ── Analysis quantities for figure annotation ─────────────────────
    _inputs, _ = xor.xor_dataset()

    _items = [
        *[(f"relu_{_i + 1}", _m) for _i, _m in enumerate(xor.relu_solutions)],
        ("op_relu_1", xor.overparam_solutions[0]),
        ("op_relu_4", xor.overparam_solutions[1]),
    ]

    hidden_activations: dict[str, np.ndarray] = {}
    rank_one_components: dict[str, np.ndarray] = {}
    rsms: dict[str, np.ndarray] = {}
    for _name, _model in _items:
        _H = mlp.compute_hidden(_model, _inputs)  # (n_stimuli, n_hidden)
        _outer = mlp.compute_hidden(_model, _inputs, return_outer=True)
        hidden_activations[_name] = np.asarray(_H.T)
        rank_one_components[_name] = np.asarray(_outer)
        rsms[_name] = np.asarray(
            rsa.calculate_rdm(_H, distances.dot_product, return_matrix=True)
        )

    _triu = np.triu_indices(next(iter(rsms.values())).shape[0], k=1)

    def _pairwise(_labels):
        return np.array(
            [
                [
                    float(
                        distances.pearson_correlation(
                            rsms[_a][_triu], rsms[_b][_triu]
                        )
                    )
                    for _b in _labels
                ]
                for _a in _labels
            ]
        )

    rsm_similarity_standard_labels = ["relu_1", "relu_3", "relu_4", "relu_6"]
    rsm_similarity_standard = _pairwise(rsm_similarity_standard_labels)

    rsm_similarity_overparam_labels = [
        "relu_1",
        "op_relu_1",
        "relu_4",
        "op_relu_4",
    ]
    rsm_similarity_overparam = _pairwise(rsm_similarity_overparam_labels)
    return (
        hidden_activations,
        rank_one_components,
        rsm_similarity_overparam,
        rsm_similarity_overparam_labels,
        rsm_similarity_standard,
        rsm_similarity_standard_labels,
        rsms,
    )


@app.cell
def _(
    hidden_activations: dict[str, np.ndarray],
    rank_one_components: dict[str, np.ndarray],
    rsm_similarity_overparam,
    rsm_similarity_overparam_labels,
    rsm_similarity_standard,
    rsm_similarity_standard_labels,
    rsms: dict[str, np.ndarray],
):
    # ── Analysis printout: values for figure annotation ───────────────
    np.set_printoptions(precision=2)

    for _name, _H in hidden_activations.items():
        print(f"──── {_name} ────\n")
        print("Hidden activation matrix H\n")
        print(f"{_H}\n")
        print("Rank-one components zz^T of RSM\n")
        for _comp in rank_one_components[_name]:
            print(f"{_comp}\n")
        print("Full RSM\n")
        print(f"{rsms[_name]}\n\n")

    def _print_similarity(title, labels, matrix):
        _w = max(len(_n) for _n in labels)
        print(title)
        print(" " * (_w + 2) + "  ".join(f"{_n:>{_w}}" for _n in labels))
        for _i, _name in enumerate(labels):
            _row = "  ".join(f"{_v:>{_w}.3f}" for _v in matrix[_i])
            print(f"{_name:>{_w}}  {_row}")

    _print_similarity(
        "Pairwise RSM similarity (standard solutions)\n",
        rsm_similarity_standard_labels,
        rsm_similarity_standard,
    )
    print("\n")
    _print_similarity(
        "Pairwise RSM similarity (overparameterized solutions)\n",
        rsm_similarity_overparam_labels,
        rsm_similarity_overparam,
    )
    return


if __name__ == "__main__":
    app.run()
