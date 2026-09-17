import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import matplotlib.axes as mpl_axes
    import matplotlib.pyplot as plt
    import numpy as np
    from experiments.sweeps import relu_sweep_models
    from figures.appendix.xor_solutions import NEURON_COLORS
    from figures.appendix.xor_solutions import data_logit_magnitude
    from figures.fonts.config import FIGURE_FONT_CONFIGS
    from figures.fonts.config import register_bundled_fonts
    from jaxtyping import Array
    from scipy.spatial.distance import squareform

    from figures.main.dissociation.components import DATA_KWARGS
    from figures.main.dissociation.components import HEATMAP_KWARGS
    from figures.main.dissociation.components import HYPERPLANE_KWARGS
    from symmetries import clustering
    from symmetries import mlp
    from symmetries import plotting
    from symmetries import xor

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    register_bundled_fonts()

    # Importing the Figure 1 components changes a few rcParams for their
    # SVG exports, so start from the Matplotlib defaults.
    plt.style.use(["default", "./style.mplstyle"])

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STEM = "xor-sweep-clusters"

    ROW_LABEL = "Solution {number}\n($n = {size}$)"

    # Ward is defined on coordinates; the quotient metric has none.
    LINKAGE_METHOD = "average"


@app.function
def nearest_solution(
    neurons: np.ndarray,
    network: np.ndarray,
    reference_neurons: np.ndarray,
    reference_networks: np.ndarray,
) -> int:
    """Return the index of the closed-form solution a network matches.

    Compared under the same quotient the clustering uses, so a network
    matches whatever its hidden neurons can be matched onto.

    Args:
        neurons: Per-neuron features of one network.
        network: Per-network features of that network.
        reference_neurons: Per-neuron features of the closed forms.
        reference_networks: Per-network features of the closed forms.

    Returns:
        The index of the nearest closed-form solution.
    """
    distances = clustering.permutation_invariant_distances(
        np.concatenate([neurons[None], reference_neurons]),
        np.concatenate([network[None], reference_networks]),
    )
    return int(np.argmin(distances[: len(reference_neurons)]))


@app.function
def clustered_sweep() -> tuple[
    list[mlp.MLP], np.ndarray, np.ndarray, dict[int, int]
]:
    """Cluster the converged networks and match clusters to solutions.

    Networks are compared under the metric that quotients out the
    hidden-neuron permutation, so the clusters come out in one-to-one
    correspondence with the closed-form solutions.

    Returns:
        The converged networks, the square matrix of distances between
        them, one cluster label per network, and a mapping from cluster
        label to the index of the solution that cluster matches.
    """
    models, converged = relu_sweep_models()
    kept = [m for m, ok in zip(models, converged, strict=True) if ok]

    geometry = clustering.standardize_mlp_solutions(kept)
    neurons, networks = clustering.mlp_feature_blocks(
        *geometry, standardize=True, quotient_hidden_rescaling=True
    )
    condensed = clustering.permutation_invariant_distances(neurons, networks)
    distance_matrix = squareform(condensed)

    linkage = clustering.compute_linkage(condensed, method=LINKAGE_METHOD)
    thresholds, _, _ = clustering.find_threshold_by_gap(linkage, top_k=5)
    labels = clustering.cluster_from_linkage(
        linkage, distance_threshold=thresholds[0]
    )

    # Match on untouched features: an argmin over six candidates does
    # not care how they are scaled.
    raw_neurons, raw_networks = clustering.mlp_feature_blocks(
        *geometry, quotient_hidden_rescaling=True
    )
    reference_neurons, reference_networks = clustering.mlp_feature_blocks(
        *clustering.standardize_mlp_solutions(list(xor.relu_solutions)),
        quotient_hidden_rescaling=True,
    )

    matched = {}
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        medoid = members[
            np.argmin(distance_matrix[np.ix_(members, members)].sum(axis=1))
        ]
        matched[int(cluster)] = nearest_solution(
            raw_neurons[medoid],
            raw_networks[medoid],
            reference_neurons,
            reference_networks,
        )
    return kept, distance_matrix, labels, matched


@app.function
def sweep_rows(n_reps: int) -> list[tuple[str, int, list[mlp.MLP]]]:
    """Group the clusters into the rows of the figure.

    Rows run in the order of the closed-form solutions their clusters
    match.

    Args:
        n_reps: Members to draw per cluster. The first is the cluster's
          medoid, the rest the ones most dissimilar to those already
          drawn.

    Returns:
        One ``(row label, cluster size, members)`` triple per cluster.
    """
    kept, distance_matrix, labels, matched = clustered_sweep()

    rows = []
    for cluster, solution in sorted(matched.items(), key=lambda item: item[1]):
        members = np.flatnonzero(labels == cluster)
        picks = clustering.select_diverse_by_distance(
            distance_matrix, members, n_reps
        )
        rows.append(
            (
                ROW_LABEL.format(number=solution + 1, size=len(members)),
                len(members),
                [kept[int(idx)] for idx in picks],
            )
        )
    return rows


@app.function
def at_reference_scale(
    model: mlp.MLP,
    data: tuple[Array, Array],
    target_logit: float,
) -> mlp.MLP:
    """Return the network with its readout at the closed forms' scale.

    Training runs the readout out along a scaling ray, so a converged
    network's logits are tens of times the closed forms'. Rescaling the
    readout moves only the colour scale: every activation boundary and
    every readout sign stays where it is.

    Args:
        model: The trained network.
        data: The ``(inputs, labels)`` pair of the XOR task.
        target_logit: Logit magnitude to put the data points at.

    Returns:
        The rescaled network.
    """
    inputs, _ = data
    magnitude = max(abs(float(model(point)[0])) for point in inputs)
    scale = target_logit / magnitude
    return mlp.mlp_from_params(
        model.layers[0].weight,
        model.layers[0].bias,
        scale * model.layers[2].weight,
        scale * model.layers[2].bias,
        "relu",
    )


@app.function
def plot_cluster_panel(
    ax: mpl_axes.Axes,
    model: mlp.MLP,
    data: tuple[Array, Array],
    *,
    target_logit: float,
    data_size: float,
    data_edge_lw: float,
    normal_lw: float,
) -> None:
    """Draw one trained network over the input square.

    Args:
        ax: Axes to draw the panel on.
        model: The network to visualize.
        data: The ``(inputs, labels)`` pair of the XOR task.
        target_logit: Logit magnitude to draw the network at.
        data_size: Marker area of the data points, in points squared.
        data_edge_lw: Line width of the data markers' outline.
        normal_lw: Line width of the boundary normal arrows.
    """
    plotting.plot_mlp(
        model=at_reference_scale(model, data, target_logit),
        data=data,
        heatmap_kwargs=HEATMAP_KWARGS,
        hyperplane_kwargs={
            **HYPERPLANE_KWARGS,
            "neuron_colors": NEURON_COLORS,
            "normal_lw": normal_lw,
        },
        data_kwargs={
            **DATA_KWARGS,
            "s": data_size,
            "linewidths": data_edge_lw,
        },
        ax=ax,
    )
    ax.set_xticks([])
    ax.set_yticks([])


@app.function
def plot_sweep_clusters(
    *,
    fig_size: tuple[float, float],
    n_reps: int,
    data_size: float,
    data_edge_lw: float,
    boundary_lw: float,
    normal_lw: float,
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the members of each cluster the sweep matched.

    One row per matched cluster, ordered by the closed-form solution it
    matches. A row leads with its cluster's most
    representative member and continues with the members furthest from
    it, so what a row shows is how far a cluster's extremes stray from
    its type.

    Args:
        fig_size: Figure size in inches.
        n_reps: Representatives to draw per cluster.
        data_size: Marker area of the data points, in points squared.
        data_edge_lw: Line width of the data markers' outline.
        boundary_lw: Line width of the activation boundaries.
        normal_lw: Line width of the boundary normal arrows.
        output_path: If given, the path to save the figure to.
        show: Whether to display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure` instance.
    """
    data = xor.xor_dataset()
    rows = sweep_rows(n_reps)
    target_logit = data_logit_magnitude()

    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=n_reps,
        sharex=True,
        sharey=True,
        figsize=fig_size,
        squeeze=False,
    )

    # The activation boundaries take their width from the line default.
    with plt.rc_context({"lines.linewidth": boundary_lw}):
        for row, members in zip(axs, (m for _, _, m in rows), strict=True):
            for ax, model in zip(row, members, strict=True):
                plot_cluster_panel(
                    ax,
                    model,
                    data,
                    target_logit=target_logit,
                    data_size=data_size,
                    data_edge_lw=data_edge_lw,
                    normal_lw=normal_lw,
                )

    for (label, _, _), row in zip(rows, axs, strict=True):
        row[0].set_ylabel(label)

    # Save output if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@app.function
def export_sweep_clusters_variants(
    *,
    output_dir: Path,
    output_stem: str,
    fig_size: tuple[float, float],
    n_reps: int,
    data_size: float,
    data_edge_lw: float,
    boundary_lw: float,
    normal_lw: float,
) -> None:
    """Export submission and preprint variants with matched fonts."""
    for variant, font_config in FIGURE_FONT_CONFIGS.items():
        variant_output_dir = output_dir / variant
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = variant_output_dir / f"{output_stem}.pdf"
        with plt.rc_context(font_config):
            fig = plot_sweep_clusters(
                fig_size=fig_size,
                n_reps=n_reps,
                data_size=data_size,
                data_edge_lw=data_edge_lw,
                boundary_lw=boundary_lw,
                normal_lw=normal_lw,
                output_path=output_path,
            )
        plt.close(fig)


@app.cell
def _():
    export_sweep_clusters_variants(
        output_dir=OUTPUT_DIR,
        output_stem=OUTPUT_STEM,
        fig_size=(3.6, 6.2),
        n_reps=3,
        data_size=40.0,
        data_edge_lw=0.5,
        boundary_lw=0.85,
        normal_lw=0.85,
    )
    return


if __name__ == "__main__":
    app.run()
