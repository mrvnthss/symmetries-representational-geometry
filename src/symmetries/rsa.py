"""Representational Similarity Analysis (RSA) utilities.

This module provides functions for Representational Similarity Analysis,
a method for comparing neural representations by analyzing the structure
of representational dissimilarity matrices (RDMs).

The module includes:

* RDM computation from activation matrices,
* RSA comparison functions,
* pre-configured RDM functions for common distance measures, and
* pre-configured RSA measures.

Functions:
---------

* :func:`calculate_rdm`: Compute the RDM from activations using a
  dissimilarity measure.
* :func:`calculate_rsa`: Compare two representations using RSA.
* :func:`dot_product_rsm`: Compute RSM using dot product.
* :func:`correlation_rdm`: Compute RDM using correlation distance.
* :func:`euclidean_rdm`: Compute RDM using scaled squared Euclidean
  distance.
* :func:`dot_product_rsm_correlation_rsa`: RSA with Pearson correlation
  on dot product RSMs.
* :func:`correlation_rdm_correlation_rsa`: RSA with Pearson correlation
  on correlation RDMs.
* :func:`correlation_rdm_cosine_rsa`: RSA with cosine similarity on
  correlation RDMs.
* :func:`euclidean_rdm_correlation_rsa`: RSA with Pearson correlation
  on Euclidean RDMs.
* :func:`euclidean_rdm_cosine_rsa`: RSA with cosine similarity on
  Euclidean RDMs.

Notes:
-----
All functions operate on activation matrices and are compatible with
JAX transformations such as ``jit`` and ``vmap``.
"""

__all__ = [
    "calculate_rdm",
    "calculate_rsa",
    "correlation_rdm",
    "correlation_rdm_correlation_rsa",
    "correlation_rdm_cosine_rsa",
    "dot_product_rsm",
    "dot_product_rsm_correlation_rsa",
    "euclidean_rdm",
    "euclidean_rdm_correlation_rsa",
    "euclidean_rdm_cosine_rsa",
]

from collections.abc import Callable
from typing import Literal
from typing import overload

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from symmetries import distances


@overload
def calculate_rdm(
    activations: Float[Array, "n_stimuli n_features"],
    rdm_measure: Callable[[Array, Array], Array],
    *,
    return_matrix: Literal[False] = False,
) -> Float[Array, " n_pairs"]: ...


@overload
def calculate_rdm(
    activations: Float[Array, "n_stimuli n_features"],
    rdm_measure: Callable[[Array, Array], Array],
    *,
    return_matrix: Literal[True],
) -> Float[Array, "n_stimuli n_stimuli"]: ...


def calculate_rdm(
    activations: Float[Array, "n_stimuli n_features"],
    rdm_measure: Callable[[Array, Array], Array],
    *,
    return_matrix: bool = False,
) -> Float[Array, " n_pairs"] | Float[Array, "n_stimuli n_stimuli"]:
    """Compute representational dissimilarities for activations.

    Computes pairwise dissimilarities between all stimulus
    representations. By default, returns only the upper triangular
    elements (excluding diagonal). When ``return_matrix=True``, returns
    the full symmetric matrix.

    Assumes that ``rdm_measure`` is symmetric:
        rdm_measure(x, y) == rdm_measure(y, x)

    Args:
        activations: Activation matrix of shape (n_stimuli, n_features).
        rdm_measure: Function that computes dissimilarity between two
          feature vectors.
        return_matrix: If True, return full symmetric matrix including
          diagonal. If False (default), return only upper triangular
          elements as a flattened array.

    Returns:
        If ``return_matrix=False``: Upper triangular elements of the RDM,
        flattened to a 1D array of shape (n_pairs,) where
        n_pairs = n_stimuli * (n_stimuli - 1) / 2.

        If ``return_matrix=True``: Full symmetric matrix of shape
        (n_stimuli, n_stimuli).
    """
    n = activations.shape[0]
    rows, cols = jnp.triu_indices(n, k=1)
    triu_vals = jax.vmap(rdm_measure)(activations[rows], activations[cols])

    if not return_matrix:
        return triu_vals

    # Build full symmetric matrix with diagonal
    matrix = jnp.zeros((n, n))
    matrix = matrix.at[rows, cols].set(triu_vals)
    matrix = matrix + matrix.T

    # Compute diagonal
    diag_vals = jax.vmap(lambda x: rdm_measure(x, x))(activations)
    return matrix.at[jnp.diag_indices(n)].set(diag_vals)


def calculate_rsa(
    x: Float[Array, "n_stimuli n_features_x"],
    y: Float[Array, "n_stimuli n_features_y"],
    rdm_measure: Callable[[Array, Array], Array],
    rsa_measure: Callable[[Array, Array], Array],
) -> Float[Array, ""]:
    """Compare two representations using RSA.

    Computes RDMs for both representations and compares them using
    the specified comparison measure.

    Assumes that ``rdm_measure`` is symmetric:
        rdm_measure(x, y) == rdm_measure(y, x)

    Args:
        x: First activation matrix of shape (n_stimuli, n_features_x).
        y: Second activation matrix of shape (n_stimuli, n_features_y).
        rdm_measure: Symmetric dissimilarity measure for computing RDMs.
        rsa_measure: Similarity measure for comparing the two RDMs.

    Returns:
        Scalar comparison value between the two RDMs.
    """
    rdm_x = calculate_rdm(x, rdm_measure)
    rdm_y = calculate_rdm(y, rdm_measure)
    return rsa_measure(rdm_x, rdm_y)


def dot_product_rsm(
    activations: Float[Array, "n_stimuli n_features"],
) -> Float[Array, " n_pairs"]:
    """Compute RSM using dot product.

    Args:
        activations: Activation matrix of shape (n_stimuli, n_features).

    Returns:
        Upper triangular elements of the dot product RSM.
    """
    return calculate_rdm(activations, distances.dot_product)


def correlation_rdm(
    activations: Float[Array, "n_stimuli n_features"],
) -> Float[Array, " n_pairs"]:
    """Compute RDM using correlation distance.

    Correlation distance is defined as 1 - Pearson correlation, ranging
    from 0 (identical patterns) to 2 (perfectly anti-correlated).

    Args:
        activations: Activation matrix of shape (n_stimuli, n_features).

    Returns:
        Upper triangular elements of the correlation RDM.
    """
    return calculate_rdm(activations, distances.correlation_distance)


def euclidean_rdm(
    activations: Float[Array, "n_stimuli n_features"],
) -> Float[Array, " n_pairs"]:
    """Compute RDM using scaled squared Euclidean distance.

    The squared Euclidean distance is scaled by the number of feature
    dimensions to make distances comparable across representations of
    different sizes.

    Args:
        activations: Activation matrix of shape (n_stimuli, n_features).

    Returns:
        Upper triangular elements of the Euclidean RDM.
    """
    return calculate_rdm(activations, distances.scaled_squared_euclidean)


def dot_product_rsm_correlation_rsa(
    x: Float[Array, "n_stimuli n_features_x"],
    y: Float[Array, "n_stimuli n_features_y"],
) -> Float[Array, ""]:
    """RSA using Pearson correlation on dot product RSMs.

    Computes dot product-based RSMs for both representations and
    compares them using Pearson correlation.

    Args:
        x: First activation matrix of shape (n_stimuli, n_features_x).
        y: Second activation matrix of shape (n_stimuli, n_features_y).

    Returns:
        Pearson correlation between the two RSMs.
    """
    return calculate_rsa(
        x=x,
        y=y,
        rdm_measure=distances.dot_product,
        rsa_measure=distances.pearson_correlation,
    )


def correlation_rdm_correlation_rsa(
    x: Float[Array, "n_stimuli n_features_x"],
    y: Float[Array, "n_stimuli n_features_y"],
) -> Float[Array, ""]:
    """RSA using Pearson correlation on correlation RDMs.

    Computes correlation-based RDMs for both representations and
    compares them using Pearson correlation.

    Args:
        x: First activation matrix of shape (n_stimuli, n_features_x).
        y: Second activation matrix of shape (n_stimuli, n_features_y).

    Returns:
        Pearson correlation between the two correlation RDMs.
    """
    return calculate_rsa(
        x=x,
        y=y,
        rdm_measure=distances.correlation_distance,
        rsa_measure=distances.pearson_correlation,
    )


def correlation_rdm_cosine_rsa(
    x: Float[Array, "n_stimuli n_features_x"],
    y: Float[Array, "n_stimuli n_features_y"],
) -> Float[Array, ""]:
    """RSA using cosine similarity on correlation RDMs.

    Computes correlation-based RDMs for both representations and
    compares them using cosine similarity.

    Args:
        x: First activation matrix of shape (n_stimuli, n_features_x).
        y: Second activation matrix of shape (n_stimuli, n_features_y).

    Returns:
        Cosine similarity between the two correlation RDMs.
    """
    return calculate_rsa(
        x=x,
        y=y,
        rdm_measure=distances.correlation_distance,
        rsa_measure=distances.cosine_similarity,
    )


def euclidean_rdm_correlation_rsa(
    x: Float[Array, "n_stimuli n_features_x"],
    y: Float[Array, "n_stimuli n_features_y"],
) -> Float[Array, ""]:
    """RSA using Pearson correlation on Euclidean RDMs.

    Computes Euclidean-based RDMs for both representations and
    compares them using Pearson correlation.

    Args:
        x: First activation matrix of shape (n_stimuli, n_features_x).
        y: Second activation matrix of shape (n_stimuli, n_features_y).

    Returns:
        Pearson correlation between the two Euclidean RDMs.
    """
    return calculate_rsa(
        x=x,
        y=y,
        rdm_measure=distances.scaled_squared_euclidean,
        rsa_measure=distances.pearson_correlation,
    )


def euclidean_rdm_cosine_rsa(
    x: Float[Array, "n_stimuli n_features_x"],
    y: Float[Array, "n_stimuli n_features_y"],
) -> Float[Array, ""]:
    """RSA using cosine similarity on Euclidean RDMs.

    Computes Euclidean-based RDMs for both representations and
    compares them using cosine similarity.

    Args:
        x: First activation matrix of shape (n_stimuli, n_features_x).
        y: Second activation matrix of shape (n_stimuli, n_features_y).

    Returns:
        Cosine similarity between the two Euclidean RDMs.
    """
    return calculate_rsa(
        x=x,
        y=y,
        rdm_measure=distances.scaled_squared_euclidean,
        rsa_measure=distances.cosine_similarity,
    )
