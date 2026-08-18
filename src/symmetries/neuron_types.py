"""Detection of parameter-symmetry structure in trained ReLU MLPs.

This module detects, within a single trained one-hidden-layer MLP, the
neuron-level structures through which the parameter symmetries of the
accompanying paper can manifest. It applies to activations that are
both positively homogeneous of degree 1 and even-linear (ReLU, leaky
ReLU): such activations admit the permutation, positive scaling,
duplicate-/zero-group, and linear(-duplicate)-group symmetries, but not
the constant-odd symmetries (constant-neuron and constant-duplicate
groups).

The generic reparameterization symmetries are quotiented out before any
grouping: permutation symmetry by treating neurons as an unordered set,
and positive scaling symmetry by representing each hidden neuron by the
unit vector ``u_i = (w_i, b_i) / ||(w_i, b_i)||`` together with its
effective readout ``alpha_i = a_i * ||(w_i, b_i)||``. In these
coordinates, the neuron's contribution to the network output is
``alpha_i * relu(u_i^T (x, 1))``, so two neurons are functionally
redundant precisely when their unit vectors are aligned (duplicates,
zero groups) or anti-aligned (linear and linear-duplicate groups).

Neurons are grouped by complete-linkage hierarchical clustering of the
axial (sign-invariant) chordal distance between unit vectors, and each
group is classified via its aggregate effective readouts, following the
taxonomy of Appendix E of the paper:

* ``constant``: individual neuron with (numerically) vanishing incoming
  weights; contributes an input-independent offset (or nothing, for
  ReLU with non-positive bias).
* ``zero``: group of aligned neurons whose effective readouts sum to
  (numerically) zero; the singleton case is a neuron with vanishing
  readout.
* ``duplicate``: group of two or more aligned neurons whose effective
  readouts sum to something nonzero.
* ``linear``: mixed aligned/opposite group whose summed effective
  readout vanishes; contributes a purely linear function of the input.
* ``linear_duplicate``: mixed aligned/opposite group with nonzero
  summed effective readout; acts like a single neuron plus a linear
  residual.
* ``ordinary``: singleton group with nonzero readout (no redundancy).

The module includes:

* per-neuron canonicalization (scaling-quotient coordinates),
* axial distances and within-network neuron grouping,
* group classification into the ReLU symmetry taxonomy, and
* aggregation of per-model reports into count matrices.

Classes:
-------

* :class:`GroupRecord`: Classification record for one neuron group.
* :class:`NeuronTypeReport`: Full per-model classification report.

Functions:
---------

* :func:`canonicalize_neurons`: Extract scaling-invariant per-neuron
  coordinates from an MLP.
* :func:`axial_distances`: Pairwise sign-invariant chordal distances
  between unit vectors.
* :func:`group_neurons`: Group neurons by axial proximity via
  complete-linkage clustering.
* :func:`detect_neuron_types`: Classify all hidden neurons of a model.
* :func:`count_neuron_types`: Count neurons of each type per model.
* :func:`count_group_types`: Count groups of each type per model.
"""

__all__ = [
    "NEURON_TYPES",
    "GroupRecord",
    "NeuronTypeReport",
    "axial_distances",
    "canonicalize_neurons",
    "count_group_types",
    "count_neuron_types",
    "detect_neuron_types",
    "group_neurons",
]

import dataclasses
from collections.abc import Sequence

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance

from symmetries import mlp

EPS = 1e-12

# Canonical ordering of neuron/group types (used by the count matrices)
NEURON_TYPES: tuple[str, ...] = (
    "ordinary",
    "duplicate",
    "zero",
    "linear",
    "linear_duplicate",
    "constant",
)

# Slope of the linear odd component of ReLU: relu(x) = e(x) + x / 2
_RELU_ODD_SLOPE = 0.5


@dataclasses.dataclass(frozen=True)
class GroupRecord:
    """Classification record for one group of aligned/opposite neurons.

    Attributes:
        indices: Indices of the hidden neurons in the group.
        neuron_type: One of ``NEURON_TYPES`` except ``"constant"``
          (constant neurons are individual and never form groups).
        axis: Unit vector in (w, b)-space shared by the group, taken
          from the group member with the largest absolute effective
          readout.
        signs: Sign (+1/-1) of each member's unit vector relative to
          ``axis``, in the order of ``indices``.
        readout_sum: Sum of effective readouts over all members
          (``a^+ + a^-`` in the notation of the paper).
        readout_diff: Aligned-minus-opposite sum of effective readouts
          (``a^+ - a^-``); equals ``readout_sum`` for groups without
          opposite members.
    """

    indices: np.ndarray
    neuron_type: str
    axis: np.ndarray
    signs: np.ndarray
    readout_sum: float
    readout_diff: float


@dataclasses.dataclass(frozen=True)
class NeuronTypeReport:
    """Full neuron-type classification of a one-hidden-layer MLP.

    Attributes:
        neuron_types: Per-neuron type labels, shape (n_hidden,), each
          one of ``NEURON_TYPES``.
        group_labels: Per-neuron group index, shape (n_hidden,);
          constant neurons carry the label -1.
        groups: Classification records of all non-constant groups,
          ordered by group index.
        units: Scaling-invariant unit vectors in (w, b)-space, shape
          (n_hidden, n_in + 1).
        effective_readouts: Scaling-invariant effective readouts
          ``a_i * ||(w_i, b_i)||``, shape (n_hidden,).
        silent: Boolean mask, shape (n_hidden,), marking neurons whose
          contribution to the network output is (numerically) exactly
          zero on all inputs (constant neurons with non-positive bias
          or vanishing readout).
        n_effective: Number of hidden neurons remaining after
          collapsing all detected symmetry structure (one per
          ``ordinary``, ``duplicate``, and ``linear_duplicate`` group).
        residual_linear_norm: Norm of the input-weight part of the
          summed linear residuals left behind when collapsing all
          mixed aligned/opposite groups. Zero (up to tolerance) means
          the model reduces exactly to a width-``n_effective`` ReLU
          network plus an output-bias shift.
    """

    neuron_types: np.ndarray
    group_labels: np.ndarray
    groups: tuple[GroupRecord, ...]
    units: np.ndarray
    effective_readouts: np.ndarray
    silent: np.ndarray
    n_effective: int
    residual_linear_norm: float


def canonicalize_neurons(
    model: mlp.MLP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract scaling-invariant per-neuron coordinates from an MLP.

    Each hidden neuron ``(w_i, b_i, a_i)`` is represented by the unit
    vector ``u_i = (w_i, b_i) / ||(w_i, b_i)||`` and the effective
    readout ``alpha_i = a_i * ||(w_i, b_i)||``. For positively
    homogeneous activations of degree 1, both quantities are invariant
    under the positive scaling symmetry ``(w, b, a) -> (cw, cb, a/c)``,
    and the neuron's contribution to the output is
    ``alpha_i * sigma(u_i^T (x, 1))``.

    Neurons with ``||(w_i, b_i)|| = 0`` are mapped to the zero vector
    and effective readout 0.

    Args:
        model: A one-hidden-layer MLP with scalar output.

    Returns:
        Tuple of (units, effective_readouts, scales, output_bias):
          - units: shape (n_hidden, n_in + 1), unit vectors in
            (w, b)-space (rows of norm 1, or 0 for degenerate neurons)
          - effective_readouts: shape (n_hidden,)
          - scales: shape (n_hidden,), the norms ``||(w_i, b_i)||``
          - output_bias: scalar output-layer bias
    """
    W = np.asarray(model.layers[0].weight)  # (n_hidden, n_in)
    b = np.asarray(model.layers[0].bias)  # (n_hidden,)
    a = np.asarray(model.layers[2].weight).squeeze(axis=0)  # (n_hidden,)
    output_bias = float(np.asarray(model.layers[2].bias).squeeze())

    V = np.concatenate([W, b[:, None]], axis=1)  # (n_hidden, n_in + 1)
    scales = np.linalg.norm(V, axis=-1)
    units = V / np.maximum(scales, EPS)[:, None]
    effective_readouts = a * scales

    return units, effective_readouts, scales, output_bias


def axial_distances(units: np.ndarray) -> np.ndarray:
    """Pairwise sign-invariant chordal distances between unit vectors.

    The axial distance between unit vectors ``u`` and ``v`` is
    ``min(||u - v||, ||u + v||) = sqrt(2 - 2 |u . v|)``, i.e., the
    chordal distance between the corresponding points in projective
    space. It vanishes both for duplicates (``u = v``) and for
    sign-flipped pairs (``u = -v``), matching the aligned/opposite
    group structure of the ReLU symmetries.

    Args:
        units: Unit vectors, shape (n, d).

    Returns:
        Symmetric distance matrix, shape (n, n).
    """
    cos = np.clip(np.abs(units @ units.T), 0.0, 1.0)
    return np.sqrt(np.maximum(2.0 - 2.0 * cos, 0.0))


def group_neurons(
    units: np.ndarray,
    *,
    align_tol: float,
) -> np.ndarray:
    """Group neurons by axial proximity via complete-linkage clustering.

    Two neurons end up in the same group when their unit vectors are
    aligned or anti-aligned up to the tolerance: complete linkage on
    the axial distance guarantees that every within-group pair
    satisfies ``sqrt(2 - 2 |u_i . u_j|) <= align_tol``. For small
    angles ``theta`` between the axes, the axial distance is
    approximately ``theta``, so ``align_tol`` can be read as an angular
    tolerance in radians.

    Args:
        units: Unit vectors, shape (n, d).
        align_tol: Distance threshold below which neurons are grouped.

    Returns:
        Integer group labels (0-indexed), shape (n,).
    """
    n = units.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)

    condensed = distance.squareform(axial_distances(units), checks=False)
    Z = hierarchy.linkage(condensed, method="complete")
    return hierarchy.fcluster(Z, t=align_tol, criterion="distance") - 1


def _classify_group(
    signs: np.ndarray,
    readouts: np.ndarray,
    readout_tol: float,
) -> tuple[str, float, float]:
    """Classify one neuron group from its signs and effective readouts.

    Args:
        signs: Sign (+1/-1) of each member relative to the group axis.
        readouts: Effective readouts of the members.
        readout_tol: Absolute tolerance below which aggregate readouts
          are treated as zero.

    Returns:
        Tuple of (neuron_type, readout_sum, readout_diff).
    """
    aligned_sum = float(readouts[signs > 0].sum())
    opposite_sum = float(readouts[signs < 0].sum())
    readout_sum = aligned_sum + opposite_sum
    readout_diff = aligned_sum - opposite_sum

    mixed = bool((signs > 0).any() and (signs < 0).any())
    if mixed:
        if abs(readout_sum) <= readout_tol:
            neuron_type = (
                "zero" if abs(readout_diff) <= readout_tol else "linear"
            )
        else:
            neuron_type = "linear_duplicate"
    elif abs(readout_sum) <= readout_tol:
        neuron_type = "zero"
    elif len(signs) >= 2:  # noqa: PLR2004
        neuron_type = "duplicate"
    else:
        neuron_type = "ordinary"

    return neuron_type, readout_sum, readout_diff


def detect_neuron_types(
    model: mlp.MLP,
    *,
    align_tol: float = 1e-3,
    constant_tol: float = 1e-6,
    zero_tol: float = 1e-6,
    odd_slope: float = _RELU_ODD_SLOPE,
) -> NeuronTypeReport:
    """Classify all hidden neurons of a model into symmetry types.

    Applies the taxonomy described in the module docstring: constant
    neurons are split off first, the remaining neurons are grouped by
    axial proximity of their unit vectors (quotienting permutation and
    positive scaling), and each group is classified via its aggregate
    effective readouts.

    Args:
        model: A one-hidden-layer MLP with scalar output and a
          positively homogeneous, even-linear activation (ReLU family).
        align_tol: Axial-distance threshold for grouping neurons;
          approximately an angular tolerance in radians.
        constant_tol: Threshold on the input-weight fraction
          ``||w_i|| / ||(w_i, b_i)||`` below which a neuron counts as
          constant. Scale-free by construction.
        zero_tol: Relative tolerance for aggregate effective readouts:
          readouts are treated as zero below ``zero_tol`` times the
          largest absolute effective readout in the model.
        odd_slope: Slope ``m`` of the linear odd component of the
          activation (``1/2`` for ReLU); only used for the linear
          residual diagnostic.

    Returns:
        A :class:`NeuronTypeReport` for the model.
    """
    units, effective_readouts, _scales, _ = canonicalize_neurons(model)
    n_hidden = units.shape[0]
    n_in = units.shape[1] - 1

    readout_scale = float(np.max(np.abs(effective_readouts), initial=0.0))
    readout_tol = zero_tol * max(readout_scale, EPS)

    # Constant neurons: vanishing input weights (scale-free criterion).
    # Degenerate neurons with (w, b) = 0 have units = 0 and are constant.
    weight_fraction = np.linalg.norm(units[:, :n_in], axis=-1)
    is_constant = weight_fraction <= constant_tol

    # A constant ReLU neuron contributes eff * relu(u_bias); it is
    # silent when the bias is non-positive or the readout vanishes
    bias_component = units[:, n_in]
    constant_contribution = effective_readouts * np.maximum(bias_component, 0.0)
    silent = is_constant & (np.abs(constant_contribution) <= readout_tol)

    neuron_types = np.full(n_hidden, "", dtype=object)
    neuron_types[is_constant] = "constant"
    group_labels = np.full(n_hidden, -1, dtype=int)

    # Group and classify the non-constant neurons
    active_indices = np.asarray(~is_constant).nonzero()[0]
    labels_active = group_neurons(units[active_indices], align_tol=align_tol)

    groups: list[GroupRecord] = []
    residual = np.zeros(n_in + 1)
    n_groups = int(labels_active.max()) + 1 if labels_active.size else 0
    for group_idx in range(n_groups):
        member_indices = active_indices[labels_active == group_idx]
        group_labels[member_indices] = group_idx

        # Orient the group along its functionally dominant member
        member_readouts = effective_readouts[member_indices]
        axis = units[member_indices[np.argmax(np.abs(member_readouts))]]
        signs = np.sign(units[member_indices] @ axis).astype(int)

        neuron_type, readout_sum, readout_diff = _classify_group(
            signs, member_readouts, readout_tol
        )
        neuron_types[member_indices] = neuron_type
        groups.append(
            GroupRecord(
                indices=member_indices,
                neuron_type=neuron_type,
                axis=axis,
                signs=signs,
                readout_sum=readout_sum,
                readout_diff=readout_diff,
            )
        )

        # Collapsing a mixed group to a single neuron at +axis leaves a
        # linear residual m * (readout_diff - readout_sum) * z; same-sign
        # groups contribute exactly zero since readout_diff == readout_sum
        residual += odd_slope * (readout_diff - readout_sum) * axis

    n_effective = sum(
        g.neuron_type in {"ordinary", "duplicate", "linear_duplicate"}
        for g in groups
    )
    residual_linear_norm = float(np.linalg.norm(residual[:n_in]))

    return NeuronTypeReport(
        neuron_types=neuron_types,
        group_labels=group_labels,
        groups=tuple(groups),
        units=units,
        effective_readouts=effective_readouts,
        silent=silent,
        n_effective=n_effective,
        residual_linear_norm=residual_linear_norm,
    )


def count_neuron_types(reports: Sequence[NeuronTypeReport]) -> np.ndarray:
    """Count neurons of each type per model.

    Args:
        reports: Per-model reports from :func:`detect_neuron_types`.

    Returns:
        Integer count matrix of shape (n_models, len(NEURON_TYPES)),
        with columns ordered as in ``NEURON_TYPES``.
    """
    counts = np.zeros((len(reports), len(NEURON_TYPES)), dtype=int)
    for row, report in enumerate(reports):
        for col, neuron_type in enumerate(NEURON_TYPES):
            counts[row, col] = int((report.neuron_types == neuron_type).sum())
    return counts


def count_group_types(reports: Sequence[NeuronTypeReport]) -> np.ndarray:
    """Count groups of each type per model.

    Constant neurons are individual neurons rather than groups; they
    are counted one each in the ``"constant"`` column.

    Args:
        reports: Per-model reports from :func:`detect_neuron_types`.

    Returns:
        Integer count matrix of shape (n_models, len(NEURON_TYPES)),
        with columns ordered as in ``NEURON_TYPES``.
    """
    counts = np.zeros((len(reports), len(NEURON_TYPES)), dtype=int)
    type_to_col = {t: c for c, t in enumerate(NEURON_TYPES)}
    for row, report in enumerate(reports):
        for group in report.groups:
            counts[row, type_to_col[group.neuron_type]] += 1
        counts[row, type_to_col["constant"]] = int(
            (report.neuron_types == "constant").sum()
        )
    return counts
