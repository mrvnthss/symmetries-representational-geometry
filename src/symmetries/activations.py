"""Utilities for working with JAX activation functions."""

__all__ = [
    "ACTIVATIONS",
    "ActivationSpec",
    "get_activation",
]

from collections.abc import Callable
from collections.abc import Mapping
from functools import partial
from typing import Any

import jax.nn as jnn

ActivationSpec = str | tuple[str, Mapping[str, Any]]

ACTIVATIONS = frozenset(
    {
        "celu",
        "elu",
        "gelu",
        "glu",
        "hard_sigmoid",
        "hard_silu",
        "hard_swish",
        "hard_tanh",
        "identity",
        "leaky_relu",
        "log_sigmoid",
        "mish",
        "relu",
        "relu6",
        "selu",
        "sigmoid",
        "silu",
        "soft_sign",
        "softplus",
        "sparse_plus",
        "sparse_sigmoid",
        "squareplus",
        "swish",
        "tanh",
    }
)


def _get_jnn_activation(name: str) -> Callable:
    """Get an activation function from jax.nn by name, with validation."""
    if not hasattr(jnn, name):
        available = ", ".join(sorted(ACTIVATIONS))
        msg = f"Unknown activation '{name}'. Available activations: {available}"
        raise ValueError(msg)
    return getattr(jnn, name)


def get_activation(spec: ActivationSpec) -> Callable:
    """Create a JAX activation function from an ActivationSpec.

    Args:
        spec: Either a string naming the activation (e.g., "relu") or a tuple
            of (name, kwargs) for parameterized activations (e.g.,
            ("leaky_relu", {"negative_slope": 0.1})).

    Returns:
        A callable activation function from jax.nn.

    Raises:
        ValueError: If the activation name is not found in jax.nn.

    Examples:
        >>> fn = get_activation("relu")
        >>> fn = get_activation(("leaky_relu", {"negative_slope": 0.05}))
    """
    if isinstance(spec, str):
        return _get_jnn_activation(spec)
    name, kwargs = spec
    return partial(_get_jnn_activation(name), **kwargs)
