"""Distance and similarity measures for JAX arrays.

This module provides a collection of functions for computing distances
and similarities between arrays treated as flattened vectors. All
functions are compatible with JAX transformations such as ``jit``, ``vmap``,
and ``grad``.

The module includes:

* vector norms (L0, L1, L2, and general p-norms),
* difference-based distances,
* dot products,
* cosine similarity and distance, and
* Pearson correlation and correlation distance.

Functions:
---------

* :func:`vector_norm`: Compute the p-norm of an array as a vector.
* :func:`difference_norm`: Compute the p-norm of the difference.
* :func:`dot_product`: Compute the dot product of two arrays.
* :func:`scaled_squared_euclidean`: Compute dimension-scaled squared
  Euclidean distance.
* :func:`cosine_similarity`: Compute safe cosine similarity.
* :func:`cosine_distance`: Compute safe cosine distance.
* :func:`pearson_correlation`: Compute Pearson correlation coefficient.
* :func:`correlation_distance`: Compute correlation distance.

Notes:
-----
All similarity and distance functions handle zero-norm inputs gracefully,
returning zero instead of NaN when one or both inputs have zero norm.
"""

__all__ = [
    "correlation_distance",
    "cosine_distance",
    "cosine_similarity",
    "difference_norm",
    "dot_product",
    "pearson_correlation",
    "scaled_squared_euclidean",
    "vector_norm",
]

import jax.numpy as jnp
from jaxtyping import Array


def _constant_like(x: Array, const: float | Array) -> Array:
    """Return a constant array with the same shape and dtype as ``x``."""
    return jnp.array(const, dtype=x.dtype)


def vector_norm(x: Array, p: float) -> Array:
    """Compute the vector norm of a tensor ``x`` treated as a vector."""
    if x.size == 0:
        msg = "Cannot compute norm of an empty array"
        raise ValueError(msg)

    axes = tuple(range(x.ndim))

    if p == jnp.inf:
        return jnp.amax(jnp.abs(x), axis=axes, keepdims=False)
    if p == -jnp.inf:
        return jnp.amin(jnp.abs(x), axis=axes, keepdims=False)

    match p:
        case 2:
            return jnp.sqrt(
                jnp.sum(
                    jnp.real(x * jnp.conj(x)),
                    axis=axes,
                    keepdims=False,
                )
            )

        case 1:
            return jnp.sum(
                jnp.abs(x),
                axis=axes,
                keepdims=False,
            )

        case 0:
            return jnp.sum(
                x != 0,
                dtype=x.dtype,
                axis=axes,
                keepdims=False,
            )

        case _:
            if x.dtype in (jnp.int32, jnp.int64):
                msg = "Cannot compute general p-norm for integer tensors"
                raise ValueError(msg)
            abs_x = jnp.abs(x)
            p_arr = _constant_like(abs_x, p)
            p_inv = _constant_like(abs_x, 1.0 / p_arr)
            out = jnp.sum(abs_x**p_arr, axis=axes, keepdims=False)
            return jnp.power(out, p_inv)


def difference_norm(x: Array, y: Array, p: int) -> Array:
    """Compute the vector norm of the difference between ``x`` and ``y``."""
    return vector_norm(x - y, p=p)


def dot_product(x: Array, y: Array) -> Array:
    """Compute the dot product of ``x`` and ``y`` matched as vectors."""
    axes = (tuple(range(x.ndim)), tuple(range(y.ndim)))
    return jnp.tensordot(x, y, axes=axes)


def scaled_squared_euclidean(x: Array, y: Array) -> Array:
    """Compute dimension-scaled squared Euclidean distance.

    The squared Euclidean distance is scaled by the number of elements
    to make distances comparable across representations of different
    sizes.
    """
    diff = x - y
    squared_dist = dot_product(diff, diff)
    return squared_dist / x.size


def cosine_similarity(x: Array, y: Array) -> Array:
    """Compute safe cosine similarity of ``x`` and ``y`` as vectors."""
    norm_x = vector_norm(x, p=2)
    norm_y = vector_norm(y, p=2)

    nonzero_norm_x = jnp.greater(norm_x, 0.0)
    nonzero_norm_y = jnp.greater(norm_y, 0.0)
    valid_norm = jnp.logical_and(nonzero_norm_x, nonzero_norm_y)

    cosine_similarity = dot_product(x, y) / (norm_x * norm_y)
    return jnp.where(valid_norm, cosine_similarity, 0.0)


def cosine_distance(x: Array, y: Array) -> Array:
    """Compute safe cosine distance of ``x`` and ``y`` as vectors."""
    return 1 - cosine_similarity(x, y)


def pearson_correlation(x: Array, y: Array) -> Array:
    """Compute the Pearson correlation of ``x`` and ``y`` matched as vectors."""
    return cosine_similarity(
        x - jnp.mean(x, keepdims=True),
        y - jnp.mean(y, keepdims=True),
    )


def correlation_distance(x: Array, y: Array) -> Array:
    """Compute correlation distance of ``x`` and ``y`` as vectors."""
    return 1 - pearson_correlation(x, y)
