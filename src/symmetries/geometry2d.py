"""2D geometric construction utilities for neural networks.

This module provides functions for constructing neural network layers
from intuitive geometric parameters in 2D. Instead of specifying raw
weight matrices, users can define hidden neurons by their decision
boundary orientation (angle) and position (offset).

The module includes:

* unit vector construction from angles,
* single-neuron parameter generation,
* full hidden layer construction, and
* complete MLP construction from geometry.

Functions:
---------

* :func:`unitvec`: Return a 2D unit vector from an angle.
* :func:`neuron_from_geometry`: Construct weight and bias for a single
  neuron from angle, offset, and scale.
* :func:`hidden_layer_from_geometry`: Construct hidden layer parameters
  (W1, b1) from per-neuron geometric specifications.
* :func:`mlp_from_geometry`: Create a complete one-hidden-layer MLP
  from geometric parameters.

Notes:
-----
All angles are measured in radians from the positive x-axis. The distance
parameter controls how far the decision boundary is shifted from the
origin along the normal direction.
"""

__all__ = [
    "hidden_layer_from_geometry",
    "mlp_from_geometry",
    "neuron_from_geometry",
    "unitvec",
]

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from symmetries import activations
from symmetries import mlp


def unitvec(angle: float) -> Float[Array, " 2"]:
    """Return the unit vector [cos(angle), sin(angle)]."""
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def neuron_from_geometry(
    angle: float,
    distance: float,
    gain: float = 1.0,
) -> tuple[Float[Array, " 2"], Float[Array, ""]]:
    """Construct a single hidden neuron from geometric parameters.

    The neuron preactivation is

        z(x) = weight @ x + bias
             = gain * (unitvec(angle) @ x - distance)

    where

        unitvec(angle) = (cos(angle), sin(angle)).

    Geometrically, the zero level set z(x) = 0 is the line

        unitvec(angle) @ x = distance.

    This gives a direct geometric interpretation of the parameters:

      - angle:
          Direction of the unit normal vector to the line.
          The line itself is perpendicular to this vector.

      - distance:
          Signed distance from the origin to the line, measured along
          the normal direction unitvec(angle).

          In particular:
            * distance > 0: the line lies in the +unitvec(angle)
              direction from the origin,
            * distance < 0: the line lies in the -unitvec(angle)
              direction,
            * distance = 0: the line passes through the origin.

          The point on the line closest to the origin is exactly

              distance * unitvec(angle).

      - gain:
          Overall gain factor applied to both weight and bias.
          This does not change the decision boundary z(x) = 0, only the
          magnitude of the preactivation away from the boundary.

    Equivalently, in the usual affine form

        z(x) = w @ x + b,

    we have

        w = gain * unitvec(angle)
        b = -gain * distance.

    Args:
        angle: Angle of the unit normal vector of the neuron's decision
          boundary in radians.
        distance: Signed distance from the origin to the decision
          boundary.
        gain: Positive gain factor.

    Returns:
        Tuple (weight, bias), where:
          - weight has shape (2,)
          - bias is a scalar
    """
    weight = gain * unitvec(angle)
    bias = -gain * jnp.asarray(distance)
    return weight, bias


def hidden_layer_from_geometry(
    angles: tuple[float, ...],
    distances: tuple[float, ...],
    gains: tuple[float, ...] | None = None,
) -> tuple[Float[Array, "n_hidden 2"], Float[Array, " n_hidden"]]:
    """Construct (W1, b1) for a hidden layer from geometric parameters.

    Args:
        angles: Angles of the unit normal vectors of the neurons'
          decision boundaries in radians.
        distances: Signed distances from the origin to the decision
          boundaries.
        gains: Optional positive gain factors. If omitted, all gain
          factors are 1.

    Returns:
        Tuple of (W1, b1) where W1 has shape (n_hidden, 2) and b1 has
        shape (n_hidden,).
    """
    n_hidden = len(angles)
    if len(distances) != n_hidden:
        msg = "angles and distances must have the same length"
        raise ValueError(msg)
    if gains is None:
        gains = (1.0,) * n_hidden
    elif len(gains) != n_hidden:
        msg = "angles and gains must have the same length"
        raise ValueError(msg)

    rows = [
        neuron_from_geometry(angle=a, distance=d, gain=g)
        for a, d, g in zip(angles, distances, gains, strict=True)
    ]
    W1 = jnp.stack([w for w, _ in rows])
    b1 = jnp.stack([b for _, b in rows])
    return W1, b1


def mlp_from_geometry(
    angles: tuple[float, ...],
    distances: tuple[float, ...],
    output_weights: tuple[float, ...] | Float[Array, " n_hidden"],
    output_bias: float,
    *,
    gains: tuple[float, ...] | None = None,
    activation: activations.ActivationSpec = "relu",
) -> mlp.MLP:
    """Create a one-hidden-layer network from geometric parameters.

    The i-th hidden neuron has preactivation

        z_i(x) = gains[i] * (unitvec(angles[i]) @ x - distances[i])

    and the output layer is given by

        y = output_weights @ h + output_bias

    where h is the hidden activation vector.

    Args:
        angles: Angles of the unit normal vectors of the neurons'
          decision boundaries in radians.
        distances: Signed distances from the origin to the decision
          boundaries.
        output_weights: Output layer weights as a tuple or 1D array of
          length n_hidden.
        output_bias: Output layer bias (scalar).
        gains: Optional positive gain factors. If omitted, all gain
          factors are 1.
        activation: Specification of the activation function to use.

    Returns:
        An MLP model with the specified geometry.
    """
    W1, b1 = hidden_layer_from_geometry(
        angles=angles,
        distances=distances,
        gains=gains,
    )

    W2 = jnp.asarray(output_weights)
    if W2.ndim != 1:
        msg = "output_weights must be a 1D array or tuple of length n_hidden"
        raise ValueError(msg)
    if W2.shape[0] != W1.shape[0]:
        msg = "output_weights must have length n_hidden"
        raise ValueError(msg)

    return mlp.mlp_from_params(
        W1=W1,
        b1=b1,
        W2=W2[None, :],
        b2=jnp.atleast_1d(jnp.asarray(output_bias)),
        activation=activation,
    )
