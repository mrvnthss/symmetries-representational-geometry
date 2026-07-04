"""Simple multilayer perceptron (MLP) utilities.

This module provides a minimal MLP implementation built on Equinox,
along with helper functions for constructing networks from explicit
parameters, batching/unbatching collections of models, and extracting
hidden layer activations.

The module includes:

* a two-layer (one hidden layer) MLP class,
* parameter-based model construction,
* utilities for stacking models into batched pytrees, and
* hidden activation extraction.

Classes:
-------

* :class:`MLP`: A simple two-layer neural network with configurable
  activation.

Functions:
---------

* :func:`mlp_from_params`: Create an MLP from explicit weight and bias
  arrays.
* :func:`batch_models`: Stack multiple models into a single batched
  pytree.
* :func:`unbatch_models`: Unstack a batched pytree into individual
  models.
* :func:`compute_hidden`: Compute hidden layer activations for a batch
  of inputs.

Notes:
-----
The MLP class is compatible with JAX transformations such as ``jit``,
``vmap``, and ``grad`` via Equinox's PyTree integration.
"""

__all__ = [
    "MLP",
    "batch_models",
    "compute_hidden",
    "mlp_from_params",
    "unbatch_models",
]

from collections.abc import Sequence
from typing import Literal
from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PRNGKeyArray

from symmetries import activations


class MLP(eqx.Module):
    """A simple two-layer neural network (one hidden layer)."""

    layers: tuple

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: activations.ActivationSpec = "relu",
        *,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ) -> None:
        """Initialize the MLP.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden units.
            out_features: Number of output features.
            activation: Specification of the activation function to use.
            use_bias: Whether to include bias terms in linear layers.
            key: A ``jax.random.PRNGKey`` used to provide randomness for
              parameter initialization.
        """
        k1, k2 = jax.random.split(key, 2)
        self.layers = (
            eqx.nn.Linear(in_features, hidden_features, use_bias, key=k1),
            activations.get_activation(activation),
            eqx.nn.Linear(hidden_features, out_features, use_bias, key=k2),
        )

    def __call__(
        self,
        x: Float[Array, " in_features"],
    ) -> Float[Array, " out_features"]:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x

    def hidden_activations(
        self,
        x: Float[Array, " in_features"],
    ) -> Float[Array, " hidden_features"]:
        """Compute hidden layer activations."""
        for layer in self.layers[:-1]:
            x = layer(x)
        return x


def mlp_from_params(
    W1: Float[Array, "hidden_features in_features"],
    b1: Float[Array, " hidden_features"],
    W2: Float[Array, "out_features hidden_features"],
    b2: Float[Array, " out_features"],
    activation: activations.ActivationSpec = "relu",
) -> MLP:
    """Create an MLP with specific weights and biases.

    Args:
        W1: Hidden layer weight matrix, shape (hidden_features,
          in_features).
        b1: Hidden layer bias vector, shape (hidden_features,).
        W2: Output layer weight matrix, shape (out_features,
          hidden_features).
        b2: Output layer bias vector, shape (out_features,).
        activation: Specification of the activation function to use.

    Returns:
        An MLP with the specified parameters.
    """
    hidden_features, in_features = W1.shape
    out_features = W2.shape[0]

    # Create model with dummy weights
    model = MLP(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        activation=activation,
        key=jax.random.PRNGKey(0),
    )
    # Replace with specified parameters
    return eqx.tree_at(
        lambda m: (
            m.layers[0].weight,
            m.layers[0].bias,
            m.layers[2].weight,
            m.layers[2].bias,
        ),
        model,
        (W1, b1, W2, b2),
    )


def batch_models(models: Sequence[MLP]) -> MLP:
    """Stack a sequence of models into a batched pytree.

    Combines multiple models into a single pytree where each array leaf
    has an additional leading batch dimension.

    Args:
        models: Sequence of MLP models with identical structure.

    Returns:
        A single MLP where array leaves have shape (n_models, ...).
    """
    params_list = [eqx.partition(m, eqx.is_array)[0] for m in models]
    _, static = eqx.partition(models[0], eqx.is_array)
    stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *params_list)
    return eqx.combine(stacked_params, static)


def unbatch_models(batched_model: MLP) -> list[MLP]:
    """Unstack a batched pytree into a list of models.

    Inverse of ``batch_models``.

    Args:
        batched_model: A single MLP where array leaves have shape
          (n_models, ...).

    Returns:
        List of individual MLP models.
    """
    batched_params, static = eqx.partition(batched_model, eqx.is_array)
    batch_size = jax.tree.leaves(batched_params)[0].shape[0]

    models = []
    for idx in range(batch_size):
        params = jax.tree.map(lambda x, i=idx: x[i], batched_params)
        model = eqx.combine(params, static)
        models.append(model)
    return models


@overload
def compute_hidden(
    model: MLP,
    inputs: Float[Array, "n_stimuli in_features"],
    *,
    return_outer: Literal[False] = False,
    triu_k: int | None = None,
) -> Float[Array, "n_stimuli n_hidden"]: ...


@overload
def compute_hidden(
    model: MLP,
    inputs: Float[Array, "n_stimuli in_features"],
    *,
    return_outer: Literal[True],
    triu_k: None = None,
) -> Float[Array, "n_hidden n_stimuli n_stimuli"]: ...


@overload
def compute_hidden(
    model: MLP,
    inputs: Float[Array, "n_stimuli in_features"],
    *,
    return_outer: Literal[True],
    triu_k: int,
) -> Float[Array, "n_hidden n_pairs"]: ...


def compute_hidden(
    model: MLP,
    inputs: Float[Array, "n_stimuli in_features"],
    *,
    return_outer: bool = False,
    triu_k: int | None = None,
) -> (
    Float[Array, "n_stimuli n_hidden"]
    | Float[Array, "n_hidden n_stimuli n_stimuli"]
    | Float[Array, "n_hidden n_pairs"]
):
    """Compute hidden layer activations for a batch of inputs.

    Args:
        model: The MLP model.
        inputs: Input stimuli of shape (n_stimuli, in_features).
        return_outer: If True, return rank-one matrices h h^T for each
          hidden neuron instead of the activation matrix.
        triu_k: When not None and return_outer is True, extract only the
          upper triangular part of the outer product and return it as a
          flattened vector. The value specifies the diagonal offset
          following NumPy/JAX conventions: k=0 includes the diagonal,
          k=1 excludes it. Ignored when return_outer is False.

    Returns:
        If return_outer is False: Hidden activations of shape
          (n_stimuli, n_hidden).
        If return_outer is True and triu_k is None: Array of shape
          (n_hidden, n_stimuli, n_stimuli) containing n_hidden rank-one
          matrices, where the i-th matrix of shape (n_stimuli, n_stimuli)
          is the outer product of the i-th neuron's activations.
        If return_outer is True and triu_k is not None: Array of shape
          (n_hidden, n_pairs) containing the flattened upper triangular
          part of each outer product. n_pairs depends on triu_k and
          n_stimuli.
    """
    H = jax.vmap(model.hidden_activations)(inputs)

    if not return_outer:
        return H

    outer = jax.vmap(jnp.outer)(H.T, H.T)

    if triu_k is None:
        return outer

    # Extract upper triangular indices and flatten
    n_stimuli = inputs.shape[0]
    triu_indices = jnp.triu_indices(n_stimuli, k=triu_k)
    return outer[:, triu_indices[0], triu_indices[1]]
