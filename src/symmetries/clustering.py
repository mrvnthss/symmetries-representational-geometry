"""Hierarchical clustering utilities.

This module provides convenience functions for hierarchical clustering
workflows built on top of SciPy. It includes utilities for computing
linkage matrices, selecting distance thresholds, assigning cluster
labels, and visualizing dendrograms, as well as MLP-specific utilities
for canonicalizing network parameters before clustering.

The MLP canonicalization uses the same geometric parameterization as
``geometry2d.mlp_from_geometry``: each hidden neuron is represented by
its angle (normal direction), distance (signed distance from origin to
decision boundary), gain (preactivation gain), and output weight. This
representation is activation-agnostic and works for both ReLU and tanh
networks.

The module includes:

* linkage matrix computation,
* gap-based threshold selection,
* cluster label assignment,
* diverse representative selection,
* dendrogram visualization,
* MLP geometry extraction and canonicalization, and
* cluster representative visualization.

Functions:
---------

* :func:`compute_linkage`: Compute the hierarchical clustering linkage
  matrix.
* :func:`find_threshold_by_gap`: Find candidate thresholds by analyzing
  gaps in merge distances.
* :func:`cluster_from_linkage`: Assign cluster labels by cutting the
  linkage tree.
* :func:`select_diverse_representatives`: Select maximally diverse
  representatives from a cluster using a greedy maxmin algorithm.
* :func:`plot_dendrogram`: Plot a dendrogram with optional threshold
  line.
* :func:`extract_mlp_geometry`: Extract per-neuron geometric parameters
  from a single MLP.
* :func:`standardize_mlp_solutions`: Canonicalize MLP parameters into
  geometric coordinates with permutation-invariant neuron ordering.
* :func:`prepare_mlp_for_clustering`: Combine canonicalized geometric
  parameters into feature vectors for clustering.
* :func:`plot_cluster_representatives`: Plot representative models from
  each cluster.

Notes:
-----
All functions operate on NumPy arrays and are compatible with standard
SciPy hierarchical clustering workflows.

The canonical per-neuron representation is (angle, distance, gain,
output_weight), with permutation symmetry removed by lexicographic
sorting. For ReLU networks, there is an additional positive-homogeneity
symmetry (gain can be absorbed into output weights), but the
preprocessing preserves gain explicitly so that the representation
remains compatible with tanh and with the ``mlp_from_geometry`` API.
"""

__all__ = [
    "cluster_from_linkage",
    "compute_linkage",
    "extract_mlp_geometry",
    "find_threshold_by_gap",
    "plot_cluster_representatives",
    "plot_dendrogram",
    "prepare_mlp_for_clustering",
    "select_diverse_representatives",
    "standardize_mlp_solutions",
]

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

from symmetries import mlp
from symmetries import plotting

EPS = 1e-8


def compute_linkage(
    X: np.ndarray,
    method: str = "ward",
) -> np.ndarray:
    """Compute the hierarchical clustering linkage matrix.

    This provides access to the full merge history, which is needed
    for principled threshold selection.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        method: Linkage method ('ward', 'complete', 'average', 'single').

    Returns:
        Linkage matrix Z with shape (n_samples - 1, 4). Each row contains:
        [cluster_i, cluster_j, distance, n_samples_in_new_cluster].
    """
    return hierarchy.linkage(X, method=method)


def find_threshold_by_gap(
    Z: np.ndarray,
    *,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find candidate thresholds by analyzing gaps in merge distances.

    Large gaps in consecutive merge distances suggest natural cluster
    boundaries. This helper determines the top-k largest gaps, suggests
    natural thresholds (midpoint between consecutive merge distances),
    and determines the resulting number of clusters.

    Args:
        Z: Linkage matrix from ``compute_linkage``.
        top_k: Number of largest gaps to return.

    Returns:
        Tuple of (suggested_thresholds, gap_sizes, n_clusters_at_threshold).

        - suggested_thresholds: Midpoint between consecutive merge
          distances for the top-k largest gaps.
        - gap_sizes: Size of each gap.
        - n_clusters_at_threshold: Number of clusters if cutting at each
          suggested threshold.
    """
    merge_distances = Z[:, 2]
    gaps = np.diff(merge_distances)

    # Get indices of top-k largest gaps (sorted descending by gap size)
    top_gap_indices = np.argsort(gaps)[-top_k:][::-1]

    # Determine threshold as midpoint between consecutive merge distances
    suggested_thresholds = (
        merge_distances[top_gap_indices] + merge_distances[top_gap_indices + 1]
    ) / 2

    # Filter gap sizes and determine number of clusters
    gap_sizes = gaps[top_gap_indices]
    n_samples = Z.shape[0] + 1
    n_clusters = n_samples - (top_gap_indices + 1)

    return suggested_thresholds, gap_sizes, n_clusters


def cluster_from_linkage(
    Z: np.ndarray,
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> np.ndarray:
    """Assign cluster labels by cutting the linkage tree.

    Args:
        Z: Linkage matrix from ``compute_linkage``.
        n_clusters: Desired number of clusters.
        distance_threshold: Distance threshold for cutting the
          dendrogram.

    Returns:
        Cluster labels (0-indexed) for each sample.

    Raises:
        ValueError: If neither or both of n_clusters and
          distance_threshold are specified.
    """
    if (n_clusters is None) == (distance_threshold is None):
        msg = "Specify exactly one of 'n_clusters' or 'distance_threshold'"
        raise ValueError(msg)

    if distance_threshold is not None:
        t, criterion = distance_threshold, "distance"
    else:
        t, criterion = n_clusters, "maxclust"

    return hierarchy.fcluster(Z, t=t, criterion=criterion) - 1


def select_diverse_representatives(
    X: np.ndarray,
    indices: np.ndarray,
    n_reps: int,
) -> np.ndarray:
    """Select maximally diverse representatives from a subset of points.

    Uses a greedy maxmin algorithm: starts with the point closest to the
    cluster centroid, then iteratively selects the point furthest from
    all already-selected points.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        indices: Indices of points in the subset to select from.
        n_reps: Number of representatives to select.

    Returns:
        Array of selected indices (subset of ``indices``).
    """
    if len(indices) <= n_reps:
        return indices

    X_subset = X[indices]
    n_subset = len(indices)

    # Start with point closest to centroid
    centroid = X_subset.mean(axis=0)
    dists_to_centroid = np.linalg.norm(X_subset - centroid, axis=1)
    first_idx = np.argmin(dists_to_centroid)

    selected = [first_idx]
    min_dists = np.full(n_subset, np.inf)

    for _ in range(n_reps - 1):
        # Update minimum distances to selected set
        last_selected = selected[-1]
        dists_to_last = np.linalg.norm(
            X_subset - X_subset[last_selected],
            axis=1,
        )
        min_dists = np.minimum(min_dists, dists_to_last)

        # Exclude already selected
        min_dists_masked = min_dists.copy()
        min_dists_masked[selected] = -np.inf

        # Select point with maximum minimum distance
        next_idx = np.argmax(min_dists_masked)
        selected.append(next_idx)

    return indices[np.array(selected)]


def plot_dendrogram(
    Z: np.ndarray,
    *,
    p: int = 30,
    truncate_mode: str | None = "lastp",
    threshold: float | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot dendrogram with optional threshold line.

    Args:
        Z: Linkage matrix from ``compute_linkage``.
        p: Parameter for truncation (number of leaves or level).
        truncate_mode: How to truncate the dendrogram ('lastp', 'level',
          None).
        threshold: Distance threshold for visually cutting the
          dendrogram by overlaying a horizontal line at the specified
          distance. Also used to apply distinct colors to clusters
          resulting from cutting the dendrogram at this distance.
        ax: Matplotlib axes to plot on. If None, creates a new figure.

    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    hierarchy.dendrogram(
        Z,
        p=p,
        truncate_mode=truncate_mode,
        color_threshold=threshold or 0,
        ax=ax,
    )

    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="k",
            linestyle="--",
            label=f"t = {threshold:.3f}",
        )
        ax.legend(frameon=False)

    ax.set_xlabel("Sample index (or cluster size)")
    ax.set_ylabel("Distance")

    return fig, ax


def extract_mlp_geometry(
    model: mlp.MLP,
    *,
    dead_norm_tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract per-neuron geometric parameters for 2D input MLPs.

    The neuron direction is represented by an angle computed from the
    normalized weight vector. The hidden preactivation for neuron i is
    parameterized as:

        z_i(x) = gain_i * (unitvec(angle_i) @ x - distance_i)

    This matches the API of ``geometry2d.mlp_from_geometry``.

    Args:
        model: A one-hidden-layer MLP with 2D input.
        dead_norm_tol: Neurons with weight norm below this are
          considered dead.

    Returns:
        Tuple of (angles, distances, gains, output_weights, output_bias):
          - angles: shape (n_hidden,), direction of hidden normal vectors
          - distances: shape (n_hidden,), signed distance from origin to
            decision boundary
          - gains: shape (n_hidden,), hidden preactivation gain (||w||)
          - output_weights: shape (n_hidden,), outgoing weights
          - output_bias: scalar, output layer bias
    """
    W = np.asarray(model.layers[0].weight)  # (n_hidden, in_features)
    B = np.asarray(model.layers[0].bias)  # (n_hidden,)
    A = np.asarray(model.layers[2].weight).squeeze(axis=0)  # (n_hidden,)
    b2 = float(np.asarray(model.layers[2].bias).squeeze())

    gains = np.linalg.norm(W, axis=-1)  # (n_hidden,)
    neuron_is_dead = gains <= dead_norm_tol

    # Safe division for unit vector normalization (dead neurons get placeholder)
    safe_gains = np.where(neuron_is_dead, 1.0, gains)
    W_unit = W / safe_gains[:, None]

    # Compute angles from unit vectors
    angles = np.arctan2(W_unit[:, 1], W_unit[:, 0])

    # Canonicalize: compute distances and zero out dead neurons
    # From geometry2d: bias = -gain * distance, so distance = -bias / gain
    distances = np.where(neuron_is_dead, 0.0, -B / safe_gains)
    angles = np.where(neuron_is_dead, np.inf, angles)
    gains = np.where(neuron_is_dead, 0.0, gains)
    output_weights = np.where(neuron_is_dead, 0.0, A)

    return angles, distances, gains, output_weights, b2


def standardize_mlp_solutions(
    models: list[mlp.MLP],
    *,
    dead_norm_tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize one-hidden-layer MLPs into geometric coordinates.

    Extracts per-neuron geometric parameters aligned with the
    representation used by ``geometry2d.mlp_from_geometry``, then sorts
    neurons within each model to remove permutation symmetry.

    For hidden neuron i, the preactivation is parameterized as:

        z_i(x) = gain_i * (unitvec(angle_i) @ x - distance_i)

    Dead neurons (weight norm <= dead_norm_tol) are mapped to
    (angle=inf, distance=0, gain=0, output_weight=0) and sort to the end.

    Args:
        models: List of trained one-hidden-layer MLP models.
        dead_norm_tol: Neurons with weight norm below this are
          considered dead.

    Returns:
        Tuple of (angles_canon, distances_canon, gains_canon,
        output_weights_canon, output_biases):
          - angles_canon: shape (n_models, n_hidden)
          - distances_canon: shape (n_models, n_hidden)
          - gains_canon: shape (n_models, n_hidden)
          - output_weights_canon: shape (n_models, n_hidden)
          - output_biases: shape (n_models,)
    """
    n_models = len(models)

    # Extract geometry for all models
    geometries = [
        extract_mlp_geometry(model, dead_norm_tol=dead_norm_tol)
        for model in models
    ]

    angles = np.stack([g[0] for g in geometries])  # (n_models, n_hidden)
    distances = np.stack([g[1] for g in geometries])
    gains = np.stack([g[2] for g in geometries])
    output_weights = np.stack([g[3] for g in geometries])
    output_biases = np.array([g[4] for g in geometries])  # (n_models,)

    # Sort neurons on the canonical (cos, sin, d, beta=a*g) tuple so
    # the ordering is invariant under positive ReLU rescaling
    # (w, b, a) -> (c*w, c*b, a/c). Sorting on raw (gain, output_weight)
    # would tie-break differently after such a rescaling.
    n_hidden = angles.shape[1]
    angles_safe = np.where(np.isfinite(angles), angles, 0.0)
    cos_canon = np.where(np.isfinite(angles), np.cos(angles_safe), np.inf)
    sin_canon = np.where(np.isfinite(angles), np.sin(angles_safe), np.inf)
    beta_canon = output_weights * gains
    order = np.empty((n_models, n_hidden), dtype=int)
    for i in range(n_models):
        # lexsort: last key = primary; canonical order is (cos, sin, d, beta)
        keys = (beta_canon[i], distances[i], sin_canon[i], cos_canon[i])
        order[i] = np.lexsort(keys)

    angles_canon = np.take_along_axis(angles, order, axis=1)
    distances_canon = np.take_along_axis(distances, order, axis=1)
    gains_canon = np.take_along_axis(gains, order, axis=1)
    output_weights_canon = np.take_along_axis(output_weights, order, axis=1)

    return (
        angles_canon,
        distances_canon,
        gains_canon,
        output_weights_canon,
        output_biases,
    )


def prepare_mlp_for_clustering(
    angles_canon: np.ndarray,
    distances_canon: np.ndarray,
    gains_canon: np.ndarray,
    output_weights_canon: np.ndarray,
    output_biases: np.ndarray,
    *,
    tau_distance: float | None = None,
    tau_gain: float | None = None,
    tau_output_weight: float | None = None,
    tau_output_bias: float | None = None,
    standardize: bool = False,
    quotient_hidden_rescaling: bool = False,
) -> np.ndarray:
    """Combine standardized MLP geometric parameters into feature vectors.

    Concatenates geometric parameters into a single vector suitable for
    distance-based clustering, with optional ``arcsinh`` transforms to
    dampen heavy-tailed magnitudes.

    Angles are embedded as ``(cos(theta), sin(theta))`` pairs to avoid
    discontinuities at the ``atan2`` branch cut (where angles near
    ``-pi`` and ``pi`` would otherwise appear far apart in Euclidean
    distance despite representing nearly identical directions).

    All inputs are expected to come from ``standardize_mlp_solutions``.

    Args:
        angles_canon: Angles of hidden normal vectors, shape
          (n_models, n_hidden). Embedded as unit-circle coordinates.
        distances_canon: Signed distances from origin to decision
          boundaries, shape (n_models, n_hidden). Optionally compressed
          via ``arcsinh(x / tau_distance)``.
        gains_canon: Hidden preactivation gains, shape
          (n_models, n_hidden). Optionally compressed via
          ``arcsinh(x / tau_gain)``. Ignored when
          ``quotient_hidden_rescaling=True``.
        output_weights_canon: Output layer weights, shape
          (n_models, n_hidden). Optionally compressed via
          ``arcsinh(x / tau_output_weight)``.
        output_biases: Output layer biases, shape (n_models,).
          Optionally compressed via ``arcsinh(x / tau_output_bias)``.
        tau_distance: Softness parameter for the distance ``arcsinh``
          transform. When ``None``, distances are included unchanged.
        tau_gain: Softness parameter for the gain ``arcsinh``
          transform. When ``None``, gains are included unchanged.
          Ignored when ``quotient_hidden_rescaling=True``.
        tau_output_weight: Softness parameter for the output weight
          ``arcsinh`` transform. When ``None``, output weights (or
          effective output weights in quotient mode) are included
          unchanged.
        tau_output_bias: Softness parameter for the output bias
          ``arcsinh`` transform. When ``None``, output biases are
          included unchanged.
        standardize: If True, standardize each feature dimension across
          models to zero-mean and unit variance.
        quotient_hidden_rescaling: If True, quotient out the hidden
          rescaling symmetry that exists for positively homogeneous
          activations of degree 1 (e.g., ReLU, Leaky ReLU). For such
          activations, the transformation ``(w, b, alpha) -> (c*w, c*b,
          alpha/c)`` for ``c > 0`` does not change the network function.
          When enabled, the ``gain`` feature is removed and replaced
          with ``effective_output_weight = output_weight * gain``. This
          reduces the feature dimension from ``5 * n_hidden + 1`` to
          ``4 * n_hidden + 1``. Should remain ``False`` for activations
          like ``tanh`` or ``sigmoid`` where this symmetry does not hold.

    Returns:
        Feature matrix with shape ``(n_models, 5 * n_hidden + 1)`` when
        ``quotient_hidden_rescaling=False``, or ``(n_models, 4 * n_hidden
        + 1)`` when ``quotient_hidden_rescaling=True``.
    """
    # Embed angles as (cos, sin) to avoid branch-cut discontinuity
    angle_cos = np.cos(angles_canon)
    angle_sin = np.sin(angles_canon)

    # Apply optional arcsinh transforms
    distance_feature = (
        distances_canon
        if tau_distance is None
        else np.arcsinh(distances_canon / tau_distance)
    )
    output_bias_feature = (
        output_biases
        if tau_output_bias is None
        else np.arcsinh(output_biases / tau_output_bias)
    )

    if quotient_hidden_rescaling:
        # Quotient the hidden rescaling symmetry: (w, b, a) -> (c*w, c*b, a/c)
        # Compute effective output weight = output_weight * gain
        effective_output_weight = output_weights_canon * gains_canon
        effective_output_weight_feature = (
            effective_output_weight
            if tau_output_weight is None
            else np.arcsinh(effective_output_weight / tau_output_weight)
        )
        X = np.concatenate(
            [
                angle_cos,
                angle_sin,
                distance_feature,
                effective_output_weight_feature,
                output_bias_feature[:, None],
            ],
            axis=1,
        )
    else:
        # Generic mode: include gain as separate feature
        gain_feature = (
            gains_canon
            if tau_gain is None
            else np.arcsinh(gains_canon / tau_gain)
        )
        output_weight_feature = (
            output_weights_canon
            if tau_output_weight is None
            else np.arcsinh(output_weights_canon / tau_output_weight)
        )
        X = np.concatenate(
            [
                angle_cos,
                angle_sin,
                distance_feature,
                gain_feature,
                output_weight_feature,
                output_bias_feature[:, None],
            ],
            axis=1,
        )

    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.maximum(sd, EPS)
        X = (X - mu) / sd

    return X


def plot_cluster_representatives(
    models: list[mlp.MLP],
    X: np.ndarray,
    Z: np.ndarray,
    distance_threshold: float,
    *,
    n_reps: int = 1,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray, dict[int, np.ndarray]]:
    """Plot representative models from each cluster.

    Clusters the models by cutting the linkage tree at the given
    threshold, then visualizes representatives from each cluster using
    ``plotting.plot_mlp``.

    Args:
        models: List of trained MLP models (same order as used for
          clustering).
        X: Feature matrix used for clustering (needed for diverse
          selection when n_reps > 1).
        Z: Linkage matrix from ``compute_linkage``.
        distance_threshold: Distance threshold for cutting the
          dendrogram.
        n_reps: Number of representatives per cluster. When > 1, selects
          maximally diverse models within each cluster.
        **kwargs: Additional arguments passed to ``plotting.plot_mlp``.

    Returns:
        Tuple of (figure, axes_array, representatives) where
        representatives is a dict mapping cluster labels to arrays of
        model indices.
    """
    labels = cluster_from_linkage(Z, distance_threshold=distance_threshold)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Determine grid layout
    if n_reps == 1:
        cols = min(4, n_clusters)
        rows = int(np.ceil(n_clusters / cols))
    else:
        rows = n_clusters
        cols = n_reps

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        squeeze=False,
    )

    representatives: dict[int, np.ndarray] = {}

    for cluster_idx, label in enumerate(unique_labels):
        cluster_indices = np.asarray(labels == label).nonzero()[0]
        cluster_size = len(cluster_indices)
        rep_indices = select_diverse_representatives(X, cluster_indices, n_reps)
        representatives[int(label)] = rep_indices

        for rep_idx, model_idx in enumerate(rep_indices):
            if n_reps == 1:
                row, col = divmod(cluster_idx, cols)
            else:
                row, col = cluster_idx, rep_idx

            ax = axes[row, col]
            _ = plotting.plot_mlp(models[model_idx], ax=ax, **kwargs)

            if n_reps == 1 or rep_idx == 0:
                ax.set_title(f"Cluster {label} (n={cluster_size})")

    # Hide unused axes
    if n_reps == 1:
        for idx in range(n_clusters, rows * cols):
            row, col = divmod(idx, cols)
            axes[row, col].set_visible(False)

    return fig, axes, representatives
