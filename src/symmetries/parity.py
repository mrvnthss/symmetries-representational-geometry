"""Parity classification task in arbitrary input dimension.

This module provides the d-dimensional parity binary classification
task, the natural generalization of the XOR task to higher input
dimensions: inputs are the ``2**n_dims`` corners of the hypercube
``{-1, +1}**n_dims``, and the label of a corner is the parity of its
number of positive coordinates. For ``n_dims=2``, the task coincides
with the XOR task provided by :mod:`symmetries.xor` (including the
ordering of the four corners).

The module includes:

* parity dataset generation, and
* parity network constructors.

Functions:
---------

* :func:`parity_dataset`: Return the full parity dataset in a given
  dimension.
* :func:`paritynet`: Create a parity network with random
  initialization.
"""

__all__ = [
    "parity_dataset",
    "paritynet",
]

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PRNGKeyArray

from symmetries import activations
from symmetries import mlp


def parity_dataset(
    n_dims: int,
) -> tuple[Float[Array, "n_points n_dims"], Float[Array, " n_points"]]:
    """Return the full parity dataset in ``n_dims`` dimensions.

    The inputs are all ``2**n_dims`` corners of the hypercube
    ``{-1, +1}**n_dims``, enumerated in binary counting order with the
    first coordinate varying fastest. The label of a corner is the
    parity of its number of ``+1`` coordinates. For ``n_dims=2``, this
    reproduces ``xor.xor_dataset`` exactly.

    Args:
        n_dims: Input dimension. Must be a positive integer.

    Returns:
        Tuple of (inputs, labels) where inputs has shape
        (2**n_dims, n_dims) with corners at (+/-1, ..., +/-1) and
        labels are 0 for corners with an even number of positive
        coordinates and 1 otherwise.

    Raises:
        ValueError: If ``n_dims`` is not a positive integer.
    """
    if n_dims < 1:
        msg = f"'n_dims' must be a positive integer, got {n_dims}"
        raise ValueError(msg)

    indices = jnp.arange(2**n_dims)
    bits = (indices[:, None] >> jnp.arange(n_dims)[None, :]) & 1
    inputs = 2.0 * bits - 1.0
    labels = (bits.sum(axis=-1) % 2).astype(inputs.dtype)
    return inputs, labels


def paritynet(
    n_dims: int,
    n_hidden: int,
    activation: activations.ActivationSpec = "relu",
    *,
    use_bias: bool = True,
    key: PRNGKeyArray,
) -> mlp.MLP:
    """Create a parity network (``n_dims``-D input, 1D output).

    Convenience wrapper around ``mlp.MLP`` with parity-specific
    dimensions, generalizing ``xor.xornet`` to arbitrary input
    dimension.

    Args:
        n_dims: Input dimension.
        n_hidden: Number of hidden units.
        activation: Specification of the activation function to use.
        use_bias: Whether to include bias terms in linear layers.
        key: A ``jax.random.PRNGKey`` used to provide randomness for
          parameter initialization.

    Returns:
        An MLP with ``n_dims`` input features, ``n_hidden`` hidden
        units, and 1 output.
    """
    return mlp.MLP(
        in_features=n_dims,
        hidden_features=n_hidden,
        out_features=1,
        activation=activation,
        use_bias=use_bias,
        key=key,
    )
