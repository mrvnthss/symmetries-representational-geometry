"""Tests for clustering canonicalization functions."""

import jax.numpy as jnp
import numpy as np

from symmetries import clustering
from symmetries import geometry2d
from symmetries import mlp


class TestHiddenPermutationInvariance:
    """Test that hidden neuron permutation doesn't affect canonical output."""

    def test_two_hidden_neurons_swapped(self) -> None:
        """Swapping hidden neurons should produce identical canonical output."""
        # Create a model with specific geometry
        angles = (0.5, 1.2)
        distances = (0.3, -0.7)
        gains = (1.5, 2.0)
        output_weights = (0.8, -0.4)
        output_bias = 0.1

        model1 = geometry2d.mlp_from_geometry(
            angles=angles,
            distances=distances,
            output_weights=output_weights,
            output_bias=output_bias,
            gains=gains,
            activation="relu",
        )

        # Create model with swapped neurons
        model2 = geometry2d.mlp_from_geometry(
            angles=(angles[1], angles[0]),
            distances=(distances[1], distances[0]),
            output_weights=(output_weights[1], output_weights[0]),
            output_bias=output_bias,
            gains=(gains[1], gains[0]),
            activation="relu",
        )

        # Canonicalize both
        (
            angles1,
            distances1,
            gains1,
            output_weights1,
            output_biases1,
        ) = clustering.standardize_mlp_solutions([model1])

        (
            angles2,
            distances2,
            gains2,
            output_weights2,
            output_biases2,
        ) = clustering.standardize_mlp_solutions([model2])

        # Should match up to numerical tolerance
        np.testing.assert_allclose(angles1, angles2, rtol=1e-6)
        np.testing.assert_allclose(distances1, distances2, rtol=1e-6)
        np.testing.assert_allclose(gains1, gains2, rtol=1e-6)
        np.testing.assert_allclose(output_weights1, output_weights2, rtol=1e-6)
        np.testing.assert_allclose(output_biases1, output_biases2, rtol=1e-6)

    def test_four_hidden_neurons_permuted(self) -> None:
        """Arbitrary permutation of 4 hidden neurons should give same output."""
        angles = (0.1, 0.5, 1.0, 2.0)
        distances = (0.2, -0.3, 0.4, -0.1)
        gains = (1.0, 1.5, 2.0, 0.5)
        output_weights = (0.5, -0.5, 0.3, -0.8)
        output_bias = -0.2

        # Original model
        model1 = geometry2d.mlp_from_geometry(
            angles=angles,
            distances=distances,
            output_weights=output_weights,
            output_bias=output_bias,
            gains=gains,
            activation="relu",
        )

        # Apply a permutation: [3, 1, 0, 2]
        perm = [3, 1, 0, 2]
        model2 = geometry2d.mlp_from_geometry(
            angles=tuple(angles[i] for i in perm),
            distances=tuple(distances[i] for i in perm),
            output_weights=tuple(output_weights[i] for i in perm),
            output_bias=output_bias,
            gains=tuple(gains[i] for i in perm),
            activation="relu",
        )

        # Canonicalize both
        result1 = clustering.standardize_mlp_solutions([model1])
        result2 = clustering.standardize_mlp_solutions([model2])

        for arr1, arr2 in zip(result1, result2, strict=True):
            np.testing.assert_allclose(arr1, arr2, rtol=1e-6)


class TestDeadNeuronCanonicalization:
    """Test that dead neurons are properly canonicalized."""

    def test_dead_neuron_mapped_to_canonical_values(self) -> None:
        """Dead neurons: angle=inf, distance=0, gain=0, output_weight=0."""
        # Create a model with one dead neuron (zero weight)
        W1 = jnp.array(
            [
                [1.0, 0.5],  # Normal neuron
                [0.0, 0.0],  # Dead neuron
            ]
        )
        b1 = jnp.array([0.3, 0.5])
        W2 = jnp.array([[0.8, 0.4]])
        b2 = jnp.array([0.1])

        model = mlp.mlp_from_params(W1, b1, W2, b2, activation="relu")

        angles, distances, gains, output_weights, _output_biases = (
            clustering.standardize_mlp_solutions([model])
        )

        # The dead neuron should be at the end (angle=inf sorts last)
        # and have canonical values
        assert np.isinf(angles[0, -1])
        assert distances[0, -1] == 0.0
        assert gains[0, -1] == 0.0
        assert output_weights[0, -1] == 0.0

    def test_dead_neuron_sorts_to_end(self) -> None:
        """Dead neurons should sort to the end of the neuron list."""
        # Create model with dead neuron in the middle
        W1 = jnp.array(
            [
                [1.0, 0.0],  # Normal neuron, angle=0
                [0.0, 0.0],  # Dead neuron
                [0.0, 1.0],  # Normal neuron, angle=pi/2
            ]
        )
        b1 = jnp.array([0.3, 0.5, 0.1])
        W2 = jnp.array([[0.8, 0.4, 0.2]])
        b2 = jnp.array([0.1])

        model = mlp.mlp_from_params(W1, b1, W2, b2, activation="relu")

        angles, _distances, _gains, _output_weights, _output_biases = (
            clustering.standardize_mlp_solutions([model])
        )

        # Dead neuron should be last
        assert np.isinf(angles[0, -1])
        assert not np.isinf(angles[0, 0])
        assert not np.isinf(angles[0, 1])


class TestRoundTripConsistency:
    """Test round-trip consistency with mlp_from_geometry."""

    def test_round_trip_preserves_geometry(self) -> None:
        """Constructing then extracting should recover original geometry."""
        angles = (0.3, 1.5)
        distances = (0.5, -0.2)
        gains = (1.2, 0.8)
        output_weights = (0.7, -0.3)
        output_bias = 0.15

        model = geometry2d.mlp_from_geometry(
            angles=angles,
            distances=distances,
            output_weights=output_weights,
            output_bias=output_bias,
            gains=gains,
            activation="relu",
        )

        (
            angles_canon,
            distances_canon,
            gains_canon,
            output_weights_canon,
            output_biases,
        ) = clustering.standardize_mlp_solutions([model])

        # Compare as multisets sorted by angle, since the canonical sort
        # used by standardize_mlp_solutions is implementation-defined.
        expected_order = np.argsort(np.array(angles))
        actual_order = np.argsort(angles_canon[0])

        np.testing.assert_allclose(
            angles_canon[0][actual_order],
            np.array(angles)[expected_order],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            distances_canon[0][actual_order],
            np.array(distances)[expected_order],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            gains_canon[0][actual_order],
            np.array(gains)[expected_order],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            output_weights_canon[0][actual_order],
            np.array(output_weights)[expected_order],
            rtol=1e-6,
        )
        np.testing.assert_allclose(output_biases[0], output_bias, rtol=1e-6)


class TestReLUScalingRelation:
    """Test that ReLU scaling is represented correctly."""

    def test_scaling_changes_gain_not_angle_or_distance(self) -> None:
        """Scaling a hidden neuron should only change the gain parameter."""
        angle = 0.7
        distance = 0.4
        gain = 1.0
        output_weight = 0.5
        output_bias = 0.1

        # Original model
        model1 = geometry2d.mlp_from_geometry(
            angles=(angle,),
            distances=(distance,),
            output_weights=(output_weight,),
            output_bias=output_bias,
            gains=(gain,),
            activation="relu",
        )

        # Model with scaled hidden neuron (2x gain)
        c = 2.0
        model2 = geometry2d.mlp_from_geometry(
            angles=(angle,),
            distances=(distance,),
            output_weights=(output_weight,),
            output_bias=output_bias,
            gains=(gain * c,),
            activation="relu",
        )

        # Extract geometry
        angles1, distances1, gains1, ow1, _ob1 = (
            clustering.standardize_mlp_solutions([model1])
        )
        angles2, distances2, gains2, ow2, _ob2 = (
            clustering.standardize_mlp_solutions([model2])
        )

        # Same angle and distance
        np.testing.assert_allclose(angles1, angles2, rtol=1e-6)
        np.testing.assert_allclose(distances1, distances2, rtol=1e-6)

        # Different gain
        np.testing.assert_allclose(gains1[0, 0] * c, gains2[0, 0], rtol=1e-6)

        # Same output weight (intentionally not compensated)
        np.testing.assert_allclose(ow1, ow2, rtol=1e-6)


class TestExtractMlpGeometry:
    """Test the extract_mlp_geometry helper function."""

    def test_extracts_correct_shapes(self) -> None:
        """Extracted arrays should have correct shapes."""
        n_hidden = 3
        model = geometry2d.mlp_from_geometry(
            angles=(0.1, 0.5, 1.0),
            distances=(0.2, -0.3, 0.4),
            output_weights=(0.5, -0.5, 0.3),
            output_bias=-0.2,
            gains=(1.0, 1.5, 2.0),
            activation="relu",
        )

        angles, distances, gains, output_weights, output_bias = (
            clustering.extract_mlp_geometry(model)
        )

        assert angles.shape == (n_hidden,)
        assert distances.shape == (n_hidden,)
        assert gains.shape == (n_hidden,)
        assert output_weights.shape == (n_hidden,)
        assert isinstance(output_bias, float)

    def test_dead_neuron_canonical_values(self) -> None:
        """Dead neurons should have canonical values in extraction."""
        W1 = jnp.array(
            [
                [0.0, 0.0],  # Dead
                [1.0, 0.0],  # Alive
            ]
        )
        b1 = jnp.array([0.5, 0.3])
        W2 = jnp.array([[0.4, 0.8]])
        b2 = jnp.array([0.1])

        model = mlp.mlp_from_params(W1, b1, W2, b2, activation="relu")

        angles, distances, gains, output_weights, _output_bias = (
            clustering.extract_mlp_geometry(model)
        )

        # First neuron (dead)
        assert np.isinf(angles[0])
        assert distances[0] == 0.0
        assert gains[0] == 0.0
        assert output_weights[0] == 0.0


class TestPrepareMLPForClustering:
    """Test the prepare_mlp_for_clustering function."""

    def test_output_shape(self) -> None:
        """Output feature matrix should have correct shape."""
        n_models = 5
        n_hidden = 3

        rng = np.random.default_rng(42)
        angles = rng.standard_normal((n_models, n_hidden))
        distances = rng.standard_normal((n_models, n_hidden))
        gains = np.abs(rng.standard_normal((n_models, n_hidden)))
        output_weights = rng.standard_normal((n_models, n_hidden))
        output_biases = rng.standard_normal(n_models)

        X = clustering.prepare_mlp_for_clustering(
            angles,
            distances,
            gains,
            output_weights,
            output_biases,
        )

        # 5 features per neuron (cos, sin, distance, gain, output_weight)
        # + 1 output bias
        expected_features = 5 * n_hidden + 1
        assert X.shape == (n_models, expected_features)

    def test_arcsinh_transform_applied(self) -> None:
        """Arcsinh transform should be applied when tau is specified."""
        n_models = 3
        n_hidden = 2

        angles = np.array([[0.5, 1.0], [0.3, 0.8], [0.1, 0.4]])
        distances = np.array([[10.0, -10.0], [5.0, -5.0], [20.0, -20.0]])
        gains = np.ones((n_models, n_hidden))
        output_weights = np.ones((n_models, n_hidden))
        output_biases = np.zeros(n_models)

        tau_distance = 1.0

        X = clustering.prepare_mlp_for_clustering(
            angles,
            distances,
            gains,
            output_weights,
            output_biases,
            tau_distance=tau_distance,
        )

        # The distance features should be transformed. Distance features are in
        # positions 2*n_hidden : 3*n_hidden (after cos, sin)
        distance_features = X[:, 2 * n_hidden : 3 * n_hidden]
        expected = np.arcsinh(distances / tau_distance)
        np.testing.assert_allclose(distance_features, expected, rtol=1e-6)

    def test_standardization(self) -> None:
        """Standardization should produce zero mean and unit variance."""
        n_models = 100
        n_hidden = 2

        rng = np.random.default_rng(42)
        angles = rng.standard_normal((n_models, n_hidden))
        distances = rng.standard_normal((n_models, n_hidden)) * 5
        gains = np.abs(rng.standard_normal((n_models, n_hidden))) + 0.1
        output_weights = rng.standard_normal((n_models, n_hidden)) * 2
        output_biases = rng.standard_normal(n_models)

        X = clustering.prepare_mlp_for_clustering(
            angles,
            distances,
            gains,
            output_weights,
            output_biases,
            standardize=True,
        )

        # Should have approximately zero mean and unit variance per feature
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-10)

    def test_angle_wraparound_robustness(self) -> None:
        """Angles near -pi and pi should map to nearby feature vectors."""
        eps = 1e-3
        # Two models with angles on opposite sides of the branch cut
        angles_canon = np.array(
            [
                [np.pi - eps],
                [-np.pi + eps],
            ]
        )
        distances_canon = np.array([[0.0], [0.0]])
        gains_canon = np.array([[1.0], [1.0]])
        output_weights_canon = np.array([[1.0], [1.0]])
        output_biases = np.array([0.0, 0.0])

        X = clustering.prepare_mlp_for_clustering(
            angles_canon,
            distances_canon,
            gains_canon,
            output_weights_canon,
            output_biases,
            standardize=False,
        )

        # The two models should be very close in feature space
        dist = np.linalg.norm(X[0] - X[1])
        assert dist < 1e-2

    def test_quotient_mode_output_shape(self) -> None:
        """Quotient mode should produce 4 * n_hidden + 1 features."""
        n_models = 5
        n_hidden = 3

        rng = np.random.default_rng(42)
        angles = rng.standard_normal((n_models, n_hidden))
        distances = rng.standard_normal((n_models, n_hidden))
        gains = np.abs(rng.standard_normal((n_models, n_hidden))) + 0.1
        output_weights = rng.standard_normal((n_models, n_hidden))
        output_biases = rng.standard_normal(n_models)

        X = clustering.prepare_mlp_for_clustering(
            angles,
            distances,
            gains,
            output_weights,
            output_biases,
            quotient_hidden_rescaling=True,
        )

        # 4 features per neuron (cos, sin, distance, effective_output_weight)
        # + 1 output bias
        expected_features = 4 * n_hidden + 1
        assert X.shape == (n_models, expected_features)

    def test_quotient_mode_rescaling_invariance(self) -> None:
        """Rescaled networks should produce identical features in quotient mode.

        For positively homogeneous activations, (w, b, alpha) -> (c*w, c*b,
        alpha/c) does not change the network function. In quotient mode,
        this symmetry should be removed.
        """
        # Original parameters
        angle = 0.7
        distance = 0.5
        gain = 1.5
        output_weight = 0.8
        output_bias = 0.1

        # Rescaling factor
        c = 2.5

        # Original model parameters (as arrays for prepare_mlp_for_clustering)
        angles1 = np.array([[angle]])
        distances1 = np.array([[distance]])
        gains1 = np.array([[gain]])
        output_weights1 = np.array([[output_weight]])
        output_biases1 = np.array([output_bias])

        # Rescaled model: (c*w, c*b, alpha/c). In geometric params:
        # gain -> c*gain, output_weight -> output_weight/c
        angles2 = np.array([[angle]])
        distances2 = np.array([[distance]])  # distance = -b / ||w||, unchanged
        gains2 = np.array([[c * gain]])
        output_weights2 = np.array([[output_weight / c]])
        output_biases2 = np.array([output_bias])

        X1 = clustering.prepare_mlp_for_clustering(
            angles1,
            distances1,
            gains1,
            output_weights1,
            output_biases1,
            quotient_hidden_rescaling=True,
        )

        X2 = clustering.prepare_mlp_for_clustering(
            angles2,
            distances2,
            gains2,
            output_weights2,
            output_biases2,
            quotient_hidden_rescaling=True,
        )

        # Should be identical in quotient mode
        np.testing.assert_allclose(X1, X2, rtol=1e-10)

    def test_generic_mode_rescaling_distinction(self) -> None:
        """Rescaled networks should produce different features in generic mode.

        Without quotient_hidden_rescaling, the gain and output_weight
        are separate features, so rescaling should produce different vectors.
        """
        # Original parameters
        angle = 0.7
        distance = 0.5
        gain = 1.5
        output_weight = 0.8
        output_bias = 0.1

        # Rescaling factor
        c = 2.5

        # Original model parameters
        angles1 = np.array([[angle]])
        distances1 = np.array([[distance]])
        gains1 = np.array([[gain]])
        output_weights1 = np.array([[output_weight]])
        output_biases1 = np.array([output_bias])

        # Rescaled model
        angles2 = np.array([[angle]])
        distances2 = np.array([[distance]])
        gains2 = np.array([[c * gain]])
        output_weights2 = np.array([[output_weight / c]])
        output_biases2 = np.array([output_bias])

        X1 = clustering.prepare_mlp_for_clustering(
            angles1,
            distances1,
            gains1,
            output_weights1,
            output_biases1,
            quotient_hidden_rescaling=False,
        )

        X2 = clustering.prepare_mlp_for_clustering(
            angles2,
            distances2,
            gains2,
            output_weights2,
            output_biases2,
            quotient_hidden_rescaling=False,
        )

        # Should be different in generic mode
        dist = np.linalg.norm(X1 - X2)
        assert dist > 0.1  # Noticeably different
