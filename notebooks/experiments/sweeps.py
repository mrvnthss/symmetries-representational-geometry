"""Cached training sweeps for the XOR clustering experiment and figures.

``mo.persistent_cache`` stores results under ``__marimo__/cache``,
relative to the working directory, so every notebook launched from
``notebooks/`` shares one copy. The cache keys on the code and on the
constants below, so changing either retrains.

Defined in a module rather than in a notebook's setup block:
``persistent_cache`` needs the defining module's ``__builtins__`` to be a
mapping, which holds on import but not in ``__main__``.
"""

__all__ = [
    "LOSS_EVERY",
    "N_SEEDS",
    "N_STEPS",
    "SEED",
    "TARGET_LOSS",
    "relu_sweep_models",
    "run_relu_sweep",
]

import jax
import marimo as mo
import numpy as np

from symmetries import mlp
from symmetries import training

N_SEEDS = 1000
N_STEPS = int(1e7)
LOSS_EVERY = int(1e4)
TARGET_LOSS = 1e-12
SEED = 0


@mo.persistent_cache
def run_relu_sweep() -> tuple[tuple[np.ndarray, ...], list[bool]]:
    """Train two-hidden-unit ReLU networks on XOR across seeds.

    Returns:
        The stacked ``(W1, b1, W2, b2)`` of every trained network, and
        one flag per network saying whether it converged.
    """
    models, _, converged = training.seed_sweep(
        n_seeds=N_SEEDS,
        n_hidden=2,
        activation="relu",
        n_steps=N_STEPS,
        target_loss=TARGET_LOSS,
        loss_every=LOSS_EVERY,
        key=jax.random.PRNGKey(SEED),
    )
    # Cache raw parameter arrays: the activation function inside the
    # Equinox models is not picklable
    return (
        np.stack([np.asarray(m.layers[0].weight) for m in models]),
        np.stack([np.asarray(m.layers[0].bias) for m in models]),
        np.stack([np.asarray(m.layers[2].weight) for m in models]),
        np.stack([np.asarray(m.layers[2].bias) for m in models]),
    ), converged


def relu_sweep_models() -> tuple[list[mlp.MLP], list[bool]]:
    """Return the swept networks, rebuilt from the cached parameters.

    Returns:
        Every trained network, and one flag per network saying whether
        it converged.
    """
    (W1, b1, W2, b2), converged = run_relu_sweep()
    models = [
        mlp.mlp_from_params(W1[idx], b1[idx], W2[idx], b2[idx], "relu")
        for idx in range(len(converged))
    ]
    return models, converged
