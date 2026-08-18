"""Tests for neuron-type detection in trained ReLU MLPs."""

import jax.numpy as jnp
import numpy as np

from symmetries import mlp
from symmetries import neuron_types
from symmetries import xor


def _relu_mlp(W1, b1, a, b2=0.0) -> mlp.MLP:
    """Build a ReLU MLP from per-neuron parameters."""
    return mlp.mlp_from_params(
        jnp.asarray(W1),
        jnp.asarray(b1),
        jnp.asarray(a)[None, :],
        jnp.asarray([b2]),
        activation="relu",
    )


class TestCanonicalizeNeurons:
    """Test the scaling-quotient canonicalization."""

    def test_units_and_effective_readouts(self) -> None:
        """Unit vectors and effective readouts match the definition."""
        W1 = [[3.0, 0.0], [0.0, 1.0]]
        b1 = [4.0, 0.0]
        a = [2.0, -1.5]

        model = _relu_mlp(W1, b1, a, b2=0.25)
        units, eff, scales, output_bias = neuron_types.canonicalize_neurons(
            model
        )

        np.testing.assert_allclose(scales, [5.0, 1.0])
        np.testing.assert_allclose(units[0], [0.6, 0.0, 0.8])
        np.testing.assert_allclose(units[1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(eff, [10.0, -1.5])
        assert output_bias == 0.25

    def test_rescaling_invariance(self) -> None:
        """Positive rescaling (cw, cb, a/c) leaves coordinates unchanged."""
        c = 3.7
        model1 = _relu_mlp([[1.0, 2.0]], [0.5], [0.8])
        model2 = _relu_mlp([[c * 1.0, c * 2.0]], [c * 0.5], [0.8 / c])

        units1, eff1, _, _ = neuron_types.canonicalize_neurons(model1)
        units2, eff2, _, _ = neuron_types.canonicalize_neurons(model2)

        np.testing.assert_allclose(units1, units2, rtol=1e-6)
        np.testing.assert_allclose(eff1, eff2, rtol=1e-6)

    def test_degenerate_neuron_maps_to_zero(self) -> None:
        """Neurons with (w, b) = 0 map to zero unit vector and readout."""
        model = _relu_mlp([[0.0, 0.0]], [0.0], [1.0])
        units, eff, scales, _ = neuron_types.canonicalize_neurons(model)

        np.testing.assert_allclose(units, 0.0)
        np.testing.assert_allclose(eff, 0.0)
        np.testing.assert_allclose(scales, 0.0)


class TestAxialDistances:
    """Test the sign-invariant chordal distance."""

    def test_duplicates_and_sign_flips_at_zero_distance(self) -> None:
        """Aligned and anti-aligned unit vectors have distance zero."""
        u = np.array([0.6, 0.0, 0.8])
        units = np.stack([u, u, -u])
        D = neuron_types.axial_distances(units)

        np.testing.assert_allclose(D, 0.0, atol=1e-12)

    def test_orthogonal_vectors_at_sqrt_two(self) -> None:
        """Orthogonal axes are sqrt(2) apart."""
        units = np.eye(3)[:2]
        D = neuron_types.axial_distances(units)

        np.testing.assert_allclose(D[0, 1], np.sqrt(2.0), rtol=1e-12)


class TestDetectNeuronTypes:
    """Test classification of planted symmetry structures."""

    def test_ordinary_neurons(self) -> None:
        """Distinct, well-separated neurons are all ordinary."""
        model = _relu_mlp(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.5],
            [1.0, -2.0],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == ["ordinary", "ordinary"]
        assert report.n_effective == 2
        assert report.residual_linear_norm < 1e-12

    def test_duplicate_group(self) -> None:
        """Rescaled copies of one neuron form a duplicate group."""
        # Second neuron is the first rescaled by c = 2; readouts do not
        # cancel, so together they act as a single reference neuron
        model = _relu_mlp(
            [[1.0, 1.0], [2.0, 2.0], [0.0, 1.0]],
            [0.5, 1.0, 0.0],
            [1.0, 0.5, -1.0],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == [
            "duplicate",
            "duplicate",
            "ordinary",
        ]
        assert report.group_labels[0] == report.group_labels[1]
        assert report.n_effective == 2

    def test_zero_group(self) -> None:
        """Aligned neurons with cancelling readouts form a zero group."""
        # Effective readouts: 1 * s and (-0.5) * (2 s) sum to zero
        model = _relu_mlp(
            [[1.0, 1.0], [2.0, 2.0], [0.0, 1.0]],
            [0.5, 1.0, 0.0],
            [1.0, -0.5, -1.0],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == ["zero", "zero", "ordinary"]
        assert report.n_effective == 1

    def test_singleton_zero_readout_neuron(self) -> None:
        """A neuron with vanishing readout is a singleton zero group."""
        model = _relu_mlp(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.5],
            [1.0, 0.0],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == ["ordinary", "zero"]
        assert report.n_effective == 1

    def test_linear_group(self) -> None:
        """A sign-flipped pair with cancelling readouts is linear."""
        # relu(z) - relu(-z) = z: purely linear contribution
        model = _relu_mlp(
            [[1.0, 1.0], [-1.0, -1.0], [0.0, 1.0]],
            [0.5, -0.5, 0.0],
            [1.0, -1.0, 2.0],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == ["linear", "linear", "ordinary"]
        assert report.n_effective == 1
        assert report.residual_linear_norm > 0.1

    def test_linear_duplicate_group(self) -> None:
        """A sign-flipped pair with non-cancelling readouts."""
        model = _relu_mlp(
            [[1.0, 1.0], [-1.0, -1.0]],
            [0.5, -0.5],
            [2.0, 1.0],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == [
            "linear_duplicate",
            "linear_duplicate",
        ]
        assert report.n_effective == 1

    def test_constant_neuron(self) -> None:
        """Neurons with vanishing incoming weights are constant."""
        model = _relu_mlp(
            [[1.0, 0.0], [0.0, 0.0]],
            [0.0, 2.0],
            [1.0, 0.5],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == ["ordinary", "constant"]
        assert report.group_labels[1] == -1
        # Positive bias and nonzero readout: contributes a constant
        assert not report.silent[1]

    def test_silent_constant_neuron(self) -> None:
        """Constant ReLU neurons with non-positive bias contribute zero."""
        model = _relu_mlp(
            [[1.0, 0.0], [0.0, 0.0]],
            [0.0, -2.0],
            [1.0, 0.5],
        )
        report = neuron_types.detect_neuron_types(model)

        assert list(report.neuron_types) == ["ordinary", "constant"]
        assert report.silent[1]

    def test_permutation_invariance(self) -> None:
        """Neuron order does not affect per-type counts."""
        W1 = [[1.0, 1.0], [0.0, 1.0], [2.0, 2.0]]
        b1 = [0.5, 0.0, 1.0]
        a = [1.0, -1.0, 0.5]
        perm = [2, 0, 1]

        model1 = _relu_mlp(W1, b1, a)
        model2 = _relu_mlp(
            [W1[i] for i in perm],
            [b1[i] for i in perm],
            [a[i] for i in perm],
        )

        counts1 = neuron_types.count_neuron_types(
            [neuron_types.detect_neuron_types(model1)]
        )
        counts2 = neuron_types.count_neuron_types(
            [neuron_types.detect_neuron_types(model2)]
        )

        np.testing.assert_array_equal(counts1, counts2)

    def test_rescaling_invariance_of_classification(self) -> None:
        """Positive rescaling does not change the classification."""
        c = 5.0
        model1 = _relu_mlp(
            [[1.0, 1.0], [1.0, 1.0]],
            [0.5, 0.5],
            [1.0, -1.0],
        )
        model2 = _relu_mlp(
            [[1.0, 1.0], [c * 1.0, c * 1.0]],
            [0.5, c * 0.5],
            [1.0, -1.0 / c],
        )

        report1 = neuron_types.detect_neuron_types(model1)
        report2 = neuron_types.detect_neuron_types(model2)

        assert list(report1.neuron_types) == list(report2.neuron_types)
        assert report1.n_effective == report2.n_effective


class TestAnalyticalXORSolutions:
    """Test detection on the analytical XOR solutions from the paper."""

    def test_overparam_solution_contains_zero_group(self) -> None:
        """The overparameterized solutions plant a zero-readout neuron."""
        for model in xor.overparam_solutions:
            report = neuron_types.detect_neuron_types(model)
            group_counts = neuron_types.count_group_types([report])[0]

            counts = dict(
                zip(neuron_types.NEURON_TYPES, group_counts, strict=True)
            )
            assert counts["ordinary"] == 2
            assert counts["zero"] == 1
            assert report.n_effective == 2

    def test_parallel_hyperplane_solution_is_ordinary(self) -> None:
        """Same angle but different distances: no symmetry structure."""
        report = neuron_types.detect_neuron_types(xor.relu_solutions[0])

        assert list(report.neuron_types) == ["ordinary", "ordinary"]
        assert report.n_effective == 2

    def test_absolute_value_solution_is_linear_duplicate(self) -> None:
        """The symmetric XOR solution computes |z| via a sign-flipped pair.

        relu_solutions[1] has angles (3 pi / 4, 7 pi / 4) and distances
        (0, 0): the two hidden neurons share the same hyperplane with
        opposite orientation, i.e., they form an aligned/opposite pair
        with non-cancelling readouts. Its linear residual cannot cancel
        against other groups, which is exactly why the solution is
        irreducible despite carrying group structure.
        """
        report = neuron_types.detect_neuron_types(xor.relu_solutions[1])

        assert list(report.neuron_types) == [
            "linear_duplicate",
            "linear_duplicate",
        ]
        assert report.n_effective == 1
        assert report.residual_linear_norm > 0.1


class TestCountMatrices:
    """Test the aggregation helpers."""

    def test_shapes_and_column_order(self) -> None:
        """Count matrices have one column per type in NEURON_TYPES."""
        model = _relu_mlp([[1.0, 0.0]], [0.0], [1.0])
        reports = [neuron_types.detect_neuron_types(model)] * 3

        neuron_counts = neuron_types.count_neuron_types(reports)
        group_counts = neuron_types.count_group_types(reports)

        expected_shape = (3, len(neuron_types.NEURON_TYPES))
        assert neuron_counts.shape == expected_shape
        assert group_counts.shape == expected_shape

        ordinary_col = neuron_types.NEURON_TYPES.index("ordinary")
        assert (neuron_counts[:, ordinary_col] == 1).all()
        assert (group_counts[:, ordinary_col] == 1).all()

    def test_neuron_counts_sum_to_width(self) -> None:
        """Per-model neuron counts sum to the hidden width."""
        model = _relu_mlp(
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 1.0]],
            [0.5, 1.0, 1.0, 0.0],
            [1.0, 0.5, 1.0, -1.0],
        )
        report = neuron_types.detect_neuron_types(model)
        counts = neuron_types.count_neuron_types([report])

        assert counts.sum() == 4
