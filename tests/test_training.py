"""Tests for generalized training sweeps and loss-history subsampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from symmetries import training


class TestSeedSweepDataset:
    """Test the dataset generalization of the training sweep."""

    def test_custom_dataset_sweep_shapes(self) -> None:
        """A custom 3D dataset yields models with three input features."""
        n_seeds = 3
        n_steps = 200
        inputs = jnp.array(
            [
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [0.0, -1.0, 1.0],
                [0.0, 1.0, -1.0],
            ]
        )
        labels = jnp.array([0, 1, 0, 1])

        models, loss_hists, converged = training.seed_sweep(
            n_seeds=n_seeds,
            n_hidden=4,
            n_steps=n_steps,
            dataset=(inputs, labels),
            key=jax.random.PRNGKey(0),
        )

        assert len(models) == n_seeds
        assert len(converged) == n_seeds
        assert models[0].layers[0].weight.shape == (4, 3)
        assert loss_hists[0].shape == (n_steps,)

    def test_loss_every_subsamples_history(self) -> None:
        """loss_every reduces history length without changing training."""
        kwargs = {
            "n_seeds": 2,
            "n_hidden": 2,
            "n_steps": 200,
            "key": jax.random.PRNGKey(0),
        }

        _, hists_dense, _ = training.seed_sweep(**kwargs)
        _, hists_sparse, _ = training.seed_sweep(loss_every=50, **kwargs)

        assert hists_dense[0].shape == (200,)
        assert hists_sparse[0].shape == (4,)

        # The sparse history records every 50th step of the dense one
        np.testing.assert_allclose(
            hists_sparse[0],
            hists_dense[0][49::50],
            rtol=1e-5,
        )

    def test_loss_every_must_divide_n_steps(self) -> None:
        """Indivisible loss_every raises a ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            training.seed_sweep(
                n_seeds=1,
                n_steps=100,
                loss_every=3,
                key=jax.random.PRNGKey(0),
            )
