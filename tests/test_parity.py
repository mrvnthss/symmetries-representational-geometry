"""Tests for the parity task and generalized training sweep."""

import jax
import numpy as np
import pytest

from symmetries import parity
from symmetries import training
from symmetries import xor


class TestParityDataset:
    """Test the parity dataset construction."""

    def test_reduces_to_xor_in_two_dimensions(self) -> None:
        """The 2D parity dataset must coincide with the XOR dataset."""
        parity_inputs, parity_labels = parity.parity_dataset(2)
        xor_inputs, xor_labels = xor.xor_dataset()

        np.testing.assert_array_equal(parity_inputs, xor_inputs)
        np.testing.assert_array_equal(parity_labels, xor_labels)

    def test_shapes_and_values(self) -> None:
        """Inputs cover all hypercube corners with parity labels."""
        n_dims = 4
        inputs, labels = parity.parity_dataset(n_dims)

        assert inputs.shape == (2**n_dims, n_dims)
        assert labels.shape == (2**n_dims,)
        assert set(np.unique(inputs).tolist()) == {-1.0, 1.0}

        # No duplicate corners
        assert len({tuple(row.tolist()) for row in inputs}) == 2**n_dims

        # Label = parity of the number of positive coordinates
        expected = ((inputs > 0).sum(axis=1) % 2).astype(float)
        np.testing.assert_array_equal(labels, expected)

    def test_balanced_labels(self) -> None:
        """Half of all corners have odd parity."""
        for n_dims in (1, 2, 3, 5):
            _, labels = parity.parity_dataset(n_dims)
            assert labels.sum() == 2 ** (n_dims - 1)

    def test_rejects_non_positive_dimension(self) -> None:
        """Non-positive dimensions raise a ValueError."""
        with pytest.raises(ValueError, match="n_dims"):
            parity.parity_dataset(0)


class TestSeedSweepDataset:
    """Test the dataset generalization of the training sweep."""

    def test_parity_dataset_sweep_shapes(self) -> None:
        """Sweeping on a 3D parity dataset yields 3D-input models."""
        n_seeds = 3
        n_steps = 200

        models, loss_hists, converged = training.seed_sweep(
            n_seeds=n_seeds,
            n_hidden=4,
            n_steps=n_steps,
            dataset=parity.parity_dataset(3),
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
