"""XOR classification task and analytical solutions.

This module provides the XOR binary classification task and pre-computed
analytical solutions for two-hidden-unit networks with ReLU activations.

The module includes:

* XOR dataset generation,
* XOR network constructors, and
* analytical solution specifications and pre-built models.

Functions:
---------

* :func:`xor_dataset`: Return the 4-point XOR dataset.
* :func:`xornet`: Create an XOR network with random initialization.
* :func:`xornet_from_params`: Create an XOR network from explicit
  parameters.

Data:
----

* :data:`relu_solution_specs`: Geometric specifications for 6 ReLU
  solutions.
* :data:`relu_solutions`: Pre-built ReLU solution networks.
* :data:`overparam_solution_specs`: Geometric specifications for
  overparameterized networks (double dissociation demonstration).
* :data:`overparam_solutions`: Overparameterized ReLU solution networks.
"""

__all__ = [
    "overparam_solution_specs",
    "overparam_solutions",
    "relu_solution_specs",
    "relu_solutions",
    "xor_dataset",
    "xornet",
    "xornet_from_params",
]

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import PRNGKeyArray

from symmetries import activations
from symmetries import geometry2d
from symmetries import mlp

# ── Constants ─────────────────────────────────────────────────────────

_SQRT2 = jnp.sqrt(2.0)
_PI_OVER_4 = jnp.pi / 4

_ZERO_NEURON_GAIN = 3.0


# ── Dataset ───────────────────────────────────────────────────────────


def xor_dataset() -> tuple[Float[Array, "4 2"], Float[Array, " 4"]]:
    """Return the XOR dataset.

    Returns:
        Tuple of (inputs, labels) where inputs has shape (4, 2) with
        corners at (+/-1, +/-1) and labels are 0 for same-sign inputs
        and 1 for opposite-sign inputs.
    """
    inputs = jnp.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    labels = jnp.array([0.0, 1.0, 1.0, 0.0])
    return inputs, labels


# ── Network constructors ──────────────────────────────────────────────


def xornet(
    n_hidden: int,
    activation: activations.ActivationSpec = "relu",
    *,
    use_bias: bool = True,
    key: PRNGKeyArray,
) -> mlp.MLP:
    """Create an XOR network (2D input, 1D output).

    Convenience wrapper around ``mlp.MLP`` with XOR-specific dimensions.

    Args:
        n_hidden: Number of hidden units.
        activation: Specification of the activation function to use.
        use_bias: Whether to include bias terms in linear layers.
        key: A ``jax.random.PRNGKey`` used to provide randomness for
          parameter initialization.

    Returns:
        An MLP with 2 input features, n_hidden hidden units, and 1 output.
    """
    return mlp.MLP(
        in_features=2,
        hidden_features=n_hidden,
        out_features=1,
        activation=activation,
        use_bias=use_bias,
        key=key,
    )


def xornet_from_params(
    W1: Float[Array, "n_hidden 2"],
    b1: Float[Array, " n_hidden"],
    W2: Float[Array, "1 n_hidden"],
    b2: Float[Array, ""],
    activation: activations.ActivationSpec = "relu",
) -> mlp.MLP:
    """Create an XORNet with specific weights and biases.

    Convenience wrapper around ``mlp.mlp_from_params`` that ensures the
    bias parameter has the correct shape for a scalar output.
    """
    b2_arr = jnp.atleast_1d(jnp.asarray(b2))
    return mlp.mlp_from_params(W1, b1, W2, b2_arr, activation)


# ── Analytical solutions ──────────────────────────────────────────────

# ReLU network solutions (6 total: 2 families of 3)
relu_solution_specs: Sequence[dict] = (
    # "Diagonal" solutions
    {
        "angles": (3 * _PI_OVER_4, 3 * _PI_OVER_4),
        "distances": (0.0, -_SQRT2),
        "output_weights": (2.0, -1.0),
        "output_bias": 1 / _SQRT2,
    },
    {
        "angles": (3 * _PI_OVER_4, 7 * _PI_OVER_4),
        "distances": (0.0, 0.0),
        "output_weights": (1.0, 1.0),
        "output_bias": -1 / _SQRT2,
    },
    {
        "angles": (7 * _PI_OVER_4, 7 * _PI_OVER_4),
        "distances": (-_SQRT2, 0.0),
        "output_weights": (-1.0, 2.0),
        "output_bias": 1 / _SQRT2,
    },
    # "Anti-diagonal" solutions
    {
        "angles": (_PI_OVER_4, _PI_OVER_4),
        "distances": (-_SQRT2, 0.0),
        "output_weights": (1.0, -2.0),
        "output_bias": -1 / _SQRT2,
    },
    {
        "angles": (_PI_OVER_4, 5 * _PI_OVER_4),
        "distances": (0.0, 0.0),
        "output_weights": (-1.0, -1.0),
        "output_bias": 1 / _SQRT2,
    },
    {
        "angles": (5 * _PI_OVER_4, 5 * _PI_OVER_4),
        "distances": (0.0, -_SQRT2),
        "output_weights": (-2.0, 1.0),
        "output_bias": -1 / _SQRT2,
    },
)

# Pre-constructed solution networks
relu_solutions: tuple[mlp.MLP, ...] = tuple(
    geometry2d.mlp_from_geometry(**spec, activation="relu")
    for spec in relu_solution_specs
)


# ── Overparameterized solutions ───────────────────────────────────────

overparam_solution_specs: Sequence[dict] = (
    # "Diagonal" solution, corresponds to relu_solutions[0]
    {
        "angles": (3 * _PI_OVER_4, 3 * _PI_OVER_4, _PI_OVER_4),
        "distances": (0.0, -_SQRT2, -_SQRT2),
        "gains": (1.0, 1.0, _ZERO_NEURON_GAIN),
        "output_weights": (2.0, -1.0, 0.0),
        "output_bias": 1 / _SQRT2,
    },
    # "Anti-diagonal" solution, corresponds to relu_solutions[3]
    {
        "angles": (_PI_OVER_4, _PI_OVER_4, 3 * _PI_OVER_4),
        "distances": (-_SQRT2, 0.0, -_SQRT2),
        "gains": (1.0, 1.0, _ZERO_NEURON_GAIN),
        "output_weights": (1.0, -2.0, 0.0),
        "output_bias": -1 / _SQRT2,
    },
)

overparam_solutions: tuple[mlp.MLP, ...] = tuple(
    geometry2d.mlp_from_geometry(**spec, activation="relu")
    for spec in overparam_solution_specs
)
