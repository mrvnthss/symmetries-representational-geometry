"""Training utilities for neural networks.

This module provides training functions for binary classification tasks,
including single-model training and multi-seed parameter sweeps using
JAX and Equinox.

The module includes:

* JIT-compiled training loop, and
* parallel multi-seed training sweeps.

Functions:
---------

* :func:`train`: Train a binary classifier for a fixed number of steps.
* :func:`seed_sweep`: Train XOR networks across multiple random seeds
  in parallel.
"""

__all__ = [
    "seed_sweep",
    "train",
]

import equinox as eqx
import jax
import optax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PRNGKeyArray
from jaxtyping import PyTree

from symmetries import activations
from symmetries import mlp
from symmetries import xor


@eqx.filter_jit
def train(
    model: mlp.MLP,
    inputs: Float[Array, "batch d"],
    labels: Int[Array, " batch"],
    lr: float,
    n_steps: int,
) -> tuple[mlp.MLP, Float[Array, " n_steps"]]:
    """Train a binary classifier for a fixed number of gradient steps.

    Args:
        model: Model to train. For each input of shape (d,), the model
          must return a single logit (shape (1,) or scalar).
        inputs: Training inputs of shape (batch, d).
        labels: Binary training labels of shape (batch,). Values are
          expected to be in {0, 1}.
        lr: Optimizer learning rate.
        n_steps: Number of optimization steps.

    Returns:
        A tuple `(trained_model, loss_hist)`, where `loss_hist` contains
        the mean binary cross-entropy loss at each training step.
    """
    params, static = eqx.partition(model, eqx.is_array)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_from_params(p: PyTree[Array]) -> Float[Array, ""]:
        m = eqx.combine(p, static)
        logits = jax.vmap(m)(inputs)
        logits = logits.reshape(-1)
        return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    loss_and_grad = eqx.filter_value_and_grad(loss_from_params)

    def step(
        carry: tuple[PyTree[Array], optax.OptState], _: None
    ) -> tuple[tuple[PyTree[Array], optax.OptState], Float[Array, ""]]:
        p, s = carry
        loss_val, grads = loss_and_grad(p)
        updates, s = optimizer.update(grads, s, p)
        p = optax.apply_updates(p, updates)
        return (p, s), loss_val

    (params, _), loss_hist = jax.lax.scan(
        step,
        (params, opt_state),
        xs=None,
        length=n_steps,
    )
    trained_model = eqx.combine(params, static)

    return trained_model, loss_hist


def seed_sweep(
    n_seeds: int,
    n_hidden: int = 2,
    activation: activations.ActivationSpec = "relu",
    lr: float = 0.1,
    n_steps: int = int(1e6),
    target_loss: float = 1e-12,
    *,
    key: PRNGKeyArray,
) -> tuple[
    list[mlp.MLP],
    list[Float[Array, " n_steps"]],
    list[bool],
]:
    """Train XORNet models across multiple seeds in parallel.

    Args:
        n_seeds: Number of random seeds to sweep over.
        n_hidden: Number of hidden neurons.
        activation: Specification of the activation function to use.
        lr: Learning rate.
        n_steps: Number of optimization steps.
        target_loss: Convergence threshold.
        key: A ``jax.random.PRNGKey`` used to provide randomness for
          parameter initialization.

    Returns:
        Tuple of (trained_models, loss_hists, converged).
    """

    def init_model(k: PRNGKeyArray) -> mlp.MLP:
        return xor.xornet(
            n_hidden=n_hidden,
            activation=activation,
            use_bias=True,
            key=k,
        )

    inputs, labels = xor.xor_dataset()

    def _train(m: mlp.MLP) -> tuple[mlp.MLP, Float[Array, " n_steps"]]:
        return train(
            m,
            inputs,
            labels,
            lr,
            n_steps,
        )

    # Initialize and train models
    keys = jax.random.split(key, n_seeds)
    models = eqx.filter_vmap(init_model)(keys)
    trained_model_batched, loss_hist_batched = eqx.filter_vmap(_train)(models)

    # Unbatch results
    trained_models = mlp.unbatch_models(trained_model_batched)
    loss_hists = [loss_hist_batched[idx] for idx in range(n_seeds)]
    converged = (loss_hist_batched[:, -1] < target_loss).tolist()

    return trained_models, loss_hists, converged
