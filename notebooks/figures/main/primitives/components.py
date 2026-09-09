import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from figures.main.dissociation.components import DATA_KWARGS
    from figures.main.dissociation.components import HEATMAP_KWARGS
    from figures.main.dissociation.components import HYPERPLANE_KWARGS
    from figures.main.dissociation.components import (
        activation_kwargs_with_labels,
    )
    from figures.main.dissociation.components import make_component_axes
    from figures.main.dissociation.components import rsm_kwargs_with_labels
    from figures.main.dissociation.components import save_component
    from symmetries import distances
    from symmetries import geometry2d
    from symmetries import mlp
    from symmetries import plotting
    from symmetries import rsa
    from symmetries import xor

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    OUTPUT_PATH = Path("../figures/main/primitives/components")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # ── Numerical constants ───────────────────────────────────────────

    _SQRT2 = jnp.sqrt(2.0)
    _SQRT5 = jnp.sqrt(5.0)
    _PI_OVER_4 = jnp.pi / 4

    # ── Network constants ─────────────────────────────────────────────

    # Index of ReLU solution to use for visualizations
    RELU_SOLUTION_IDX = 0

    # Extra neuron geometry (same boundary as overparam_solutions[0])
    ZERO_ANGLE = _PI_OVER_4
    ZERO_DISTANCE = -_SQRT2
    ZERO_GAIN_EQUAL = 1.5

    # Gains for the extra zero-output neurons
    #
    # The extra neuron in xor.overparam_solutions[0] has gain g=3.
    # Its RSM contribution is g**2 * phi @ phi.T, where phi is the
    # activation pattern.  Splitting into two neurons with gains g1, g2
    # gives (g1**2 + g2**2) * phi @ phi.T.  To match, we need:
    # g1**2 + g2**2 = g**2 = 9.
    ZERO_GAIN_A = 3.0 / _SQRT5
    ZERO_GAIN_B = 6.0 / _SQRT5

    # ── Plotting constants ────────────────────────────────────────────
    plt.style.use("./style.mplstyle")
    plt.rcParams.update({"figure.dpi": 300})


@app.cell
def _():
    # ── Constructing overparameterized solutions ──────────────────────
    _base = xor.relu_solution_specs[RELU_SOLUTION_IDX]
    _n_base = len(_base["angles"])
    _base_gains = (1.0,) * _n_base  # corresponds to default gains

    # Task-linked: irreducible solution (2 hidden units)
    solution_task_linked = xor.relu_solutions[RELU_SOLUTION_IDX]

    # Addition: +1 zero-output neuron
    solution_addition = geometry2d.mlp_from_geometry(
        angles=_base["angles"] + (ZERO_ANGLE,),
        distances=_base["distances"] + (ZERO_DISTANCE,),
        gains=(*_base_gains, ZERO_GAIN_EQUAL),
        output_weights=_base["output_weights"] + (0.0,),
        output_bias=_base["output_bias"],
    )

    # Duplication: +2 zero-output neurons, equal gains
    solution_duplication = geometry2d.mlp_from_geometry(
        angles=_base["angles"] + (ZERO_ANGLE, ZERO_ANGLE),
        distances=_base["distances"] + (ZERO_DISTANCE, ZERO_DISTANCE),
        gains=(*_base_gains, ZERO_GAIN_EQUAL, ZERO_GAIN_EQUAL),
        output_weights=_base["output_weights"] + (0.0, 0.0),
        output_bias=_base["output_bias"],
    )

    # Scaling: +2 zero-output neurons, rescaled gains (RSM matches base)
    solution_scaling = geometry2d.mlp_from_geometry(
        angles=_base["angles"] + (ZERO_ANGLE, ZERO_ANGLE),
        distances=_base["distances"] + (ZERO_DISTANCE, ZERO_DISTANCE),
        gains=(*_base_gains, ZERO_GAIN_A, ZERO_GAIN_B),
        output_weights=_base["output_weights"] + (0.0, 0.0),
        output_bias=_base["output_bias"],
    )
    return (
        solution_addition,
        solution_duplication,
        solution_scaling,
        solution_task_linked,
    )


@app.cell
def _(solution_addition, solution_duplication, solution_scaling):
    # ── Panel A: Addition, duplication & scaling ──────────────────────
    _inputs, _labels = xor.xor_dataset()

    _solutions = [
        ("addition", solution_addition),
        ("duplication", solution_duplication),
        ("scaling", solution_scaling),
    ]

    for _name, _model in _solutions:
        _hidden = jax.vmap(_model.hidden_activations)(_inputs)

        # Hidden activations
        _activations = _hidden.T

        _fig, _ax = make_component_axes(
            n_rows=_activations.shape[0],
            n_cols=_activations.shape[1],
            has_bottom_ticks=True,
        )
        plotting.plot_matrix(
            _activations,
            **activation_kwargs_with_labels(_labels),
            ax=_ax,
        )
        save_component(_fig, OUTPUT_PATH / f"{_name}_activations.svg")

        # RSM
        _rsm = rsa.calculate_rdm(
            _hidden,
            distances.dot_product,
            return_matrix=True,
        )
        _rsm = _rsm / _rsm.max()

        _fig, _ax = make_component_axes(
            has_left_ticks=True, has_bottom_ticks=True
        )
        plotting.plot_rdm(
            _rsm,
            **rsm_kwargs_with_labels(_labels),
            ax=_ax,
        )
        save_component(_fig, OUTPUT_PATH / f"{_name}_rsm.svg")
    return


@app.cell
def _(solution_scaling, solution_task_linked):
    # ── Panel B: Decomposition & dissociation ─────────────────────────
    _inputs, _labels = xor.xor_dataset()

    # ── Task-linked component (irreducible solution) ──────────────────
    # Heatmap
    _fig, _ax = make_component_axes()
    plotting.plot_mlp(
        model=solution_task_linked,
        data=xor.xor_dataset(),
        heatmap_kwargs=HEATMAP_KWARGS,
        hyperplane_kwargs=HYPERPLANE_KWARGS,
        data_kwargs=DATA_KWARGS,
        ax=_ax,
    )
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_component(_fig, OUTPUT_PATH / "task-linked_heatmap.svg")

    # Hidden activations
    _hidden_task_linked = jax.vmap(solution_task_linked.hidden_activations)(
        _inputs
    )

    _activations_task_linked = _hidden_task_linked.T
    _fig, _ax = make_component_axes(
        n_rows=_activations_task_linked.shape[0],
        n_cols=_activations_task_linked.shape[1],
        has_bottom_ticks=True,
    )
    plotting.plot_matrix(
        _activations_task_linked,
        **activation_kwargs_with_labels(_labels),
        ax=_ax,
    )
    save_component(_fig, OUTPUT_PATH / "task-linked_activations.svg")

    # RSM
    _rsm = rsa.calculate_rdm(
        _hidden_task_linked,
        distances.dot_product,
        return_matrix=True,
    )
    _rsm = _rsm / _rsm.max()

    _fig, _ax = make_component_axes(has_left_ticks=True, has_bottom_ticks=True)
    plotting.plot_rdm(
        _rsm,
        **rsm_kwargs_with_labels(_labels),
        ax=_ax,
    )
    save_component(_fig, OUTPUT_PATH / "task-linked_rsm.svg")

    # ── Symmetry-induced component (zero-output neurons) ──────────────
    # Standalone sub-network of just the zero-output neurons
    _solution_symmetry_induced = geometry2d.mlp_from_geometry(
        angles=(ZERO_ANGLE, ZERO_ANGLE),
        distances=(ZERO_DISTANCE, ZERO_DISTANCE),
        gains=(ZERO_GAIN_A, ZERO_GAIN_B),
        output_weights=(0.0, 0.0),
        output_bias=0.0,
    )

    # Heatmap
    _fig, _ax = make_component_axes()
    plotting.plot_mlp(
        model=_solution_symmetry_induced,
        data=xor.xor_dataset(),
        heatmap_kwargs=HEATMAP_KWARGS,
        hyperplane_kwargs=HYPERPLANE_KWARGS,
        data_kwargs=DATA_KWARGS,
        ax=_ax,
    )
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_component(_fig, OUTPUT_PATH / "symmetry-induced_heatmap.svg")

    # Hidden activations
    _hidden_symmetry_induced = jax.vmap(
        _solution_symmetry_induced.hidden_activations
    )(_inputs)

    _activations_symmetry_induced = _hidden_symmetry_induced.T
    _fig, _ax = make_component_axes(
        n_rows=_activations_symmetry_induced.shape[0],
        n_cols=_activations_symmetry_induced.shape[1],
        has_bottom_ticks=True,
    )
    plotting.plot_matrix(
        _activations_symmetry_induced,
        **activation_kwargs_with_labels(_labels),
        ax=_ax,
    )
    save_component(_fig, OUTPUT_PATH / "symmetry-induced_activations.svg")

    # RSM
    _rsm = rsa.calculate_rdm(
        _hidden_symmetry_induced,
        distances.dot_product,
        return_matrix=True,
    )
    _rsm = _rsm / _rsm.max()

    _fig, _ax = make_component_axes(has_left_ticks=True, has_bottom_ticks=True)
    plotting.plot_rdm(
        _rsm,
        **rsm_kwargs_with_labels(_labels),
        ax=_ax,
    )
    save_component(_fig, OUTPUT_PATH / "symmetry-induced_rsm.svg")

    # ── Heatmap of overparameterized solution ─────────────────────────
    _fig, _ax = make_component_axes()
    plotting.plot_mlp(
        model=solution_scaling,
        data=xor.xor_dataset(),
        heatmap_kwargs=HEATMAP_KWARGS,
        hyperplane_kwargs=HYPERPLANE_KWARGS,
        data_kwargs=DATA_KWARGS,
        ax=_ax,
    )
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_component(_fig, OUTPUT_PATH / "scaling_heatmap.svg")

    # ── Opposite ReLU family (relu_solutions[3] = relu_4) ─────────────
    _solution_opp_family = xor.relu_solutions[3]

    # Heatmap
    _fig, _ax = make_component_axes()
    plotting.plot_mlp(
        model=_solution_opp_family,
        data=xor.xor_dataset(),
        heatmap_kwargs=HEATMAP_KWARGS,
        hyperplane_kwargs=HYPERPLANE_KWARGS,
        data_kwargs=DATA_KWARGS,
        ax=_ax,
    )
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_component(_fig, OUTPUT_PATH / "opp-family_heatmap.svg")

    # Hidden activations
    _hidden_opp_family = jax.vmap(_solution_opp_family.hidden_activations)(
        _inputs
    )

    _activations_opp_family = _hidden_opp_family.T
    _fig, _ax = make_component_axes(
        n_rows=_activations_opp_family.shape[0],
        n_cols=_activations_opp_family.shape[1],
        has_bottom_ticks=True,
    )
    plotting.plot_matrix(
        _activations_opp_family,
        **activation_kwargs_with_labels(_labels),
        ax=_ax,
    )
    save_component(_fig, OUTPUT_PATH / "opp-family_activations.svg")

    # RSM
    _rsm = rsa.calculate_rdm(
        _hidden_opp_family,
        distances.dot_product,
        return_matrix=True,
    )
    _rsm = _rsm / _rsm.max()

    _fig, _ax = make_component_axes(has_left_ticks=True, has_bottom_ticks=True)
    plotting.plot_rdm(
        _rsm,
        **rsm_kwargs_with_labels(_labels),
        ax=_ax,
    )
    save_component(_fig, OUTPUT_PATH / "opp-family_rsm.svg")
    return


@app.cell
def _(
    solution_addition,
    solution_duplication,
    solution_scaling,
    solution_task_linked,
):
    # ── Analysis quantities for figure annotation ─────────────────────
    _inputs, _ = xor.xor_dataset()

    _items = [
        ("task-linked", solution_task_linked),
        ("addition", solution_addition),
        ("duplication", solution_duplication),
        ("scaling", solution_scaling),
        ("opp-family", xor.relu_solutions[3]),
    ]

    hidden_activations: dict[str, np.ndarray] = {}
    rank_one_components: dict[str, np.ndarray] = {}
    rsms: dict[str, np.ndarray] = {}
    for _name, _model in _items:
        _H = mlp.compute_hidden(_model, _inputs)  # (n_stimuli, n_hidden)
        _outer = mlp.compute_hidden(_model, _inputs, return_outer=True)
        hidden_activations[_name] = np.asarray(_H.T)
        rank_one_components[_name] = np.asarray(_outer)
        rsms[_name] = np.asarray(
            rsa.calculate_rdm(_H, distances.dot_product, return_matrix=True)
        )

    _names = list(rsms.keys())
    _triu = np.triu_indices(rsms[_names[0]].shape[0], k=1)
    rsm_similarity = np.array(
        [
            [
                float(
                    distances.pearson_correlation(
                        rsms[_a][_triu], rsms[_b][_triu]
                    )
                )
                for _b in _names
            ]
            for _a in _names
        ]
    )
    rsm_similarity_labels = _names
    return (
        hidden_activations,
        rank_one_components,
        rsm_similarity,
        rsm_similarity_labels,
        rsms,
    )


@app.cell
def _(
    hidden_activations: dict[str, np.ndarray],
    rank_one_components: dict[str, np.ndarray],
    rsm_similarity,
    rsm_similarity_labels,
    rsms: dict[str, np.ndarray],
):
    # ── Analysis printout: values for figure annotation ───────────────
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    for _name, _H in hidden_activations.items():
        print(f"──── {_name} ────\n")
        print("Hidden activation matrix H\n")
        print(f"{_H}\n")
        print("Rank-one components zz^T of RSM\n")
        for _comp in rank_one_components[_name]:
            print(f"{_comp}\n")
        print("Full RSM\n")
        print(f"{rsms[_name]}\n\n")

    _w = max(len(_n) for _n in rsm_similarity_labels)
    print("Pairwise RSM similarity\n")
    print(
        " " * (_w + 2)
        + "  ".join(f"{_n:>{_w}}" for _n in rsm_similarity_labels)
    )
    for _i, _name in enumerate(rsm_similarity_labels):
        _row = "  ".join(f"{_v:>{_w}.4f}" for _v in rsm_similarity[_i])
        print(f"{_name:>{_w}}  {_row}")
    return


if __name__ == "__main__":
    app.run()
