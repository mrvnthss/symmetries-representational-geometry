import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import resource
    import time

    import jax
    import marimo as mo
    import matplotlib.colors as mpl_colors
    import matplotlib.pyplot as plt
    import numpy as np

    from symmetries import mlp
    from symmetries import neuron_types
    from symmetries import parity
    from symmetries import plotting
    from symmetries import training
    from symmetries import xor

    plt.style.use("./style.mplstyle")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Parity neuron types

    Train shallow ReLU networks of varying width on the
    $d$-dimensional parity task ($d = 2$ is XOR) across many random
    seeds, then detect whether solutions trained to near-zero loss
    naturally inherit the parameter-symmetry structures catalogued in
    the paper: duplicate-neuron groups, zero(-readout) neurons,
    constant neurons, and linear(-duplicate) aligned/opposite pairs.

    Unlike `xor_clustering`, which clusters whole networks by their
    geometry, this notebook clusters *neurons within each network*
    (after quotienting the permutation and positive scaling
    symmetries) and classifies the resulting groups; see
    `symmetries.neuron_types` for the detection logic.

    Sweeps are cached on disk via `mo.persistent_cache`; the first
    full run trains $12$ configurations and takes on the order of
    hours, subsequent runs are instantaneous.
    """)
    return


@app.cell
def _():
    N_SEEDS = 250
    N_STEPS = int(1e7)
    LOSS_EVERY = int(1e4)
    TARGET_LOSS = 1e-12
    SEED = 0

    # Widths per input dimension: near-minimal to 4x overparameterized
    CONFIGS = {
        n_dims: (n_dims, n_dims + 1, 2 * n_dims, 4 * n_dims)
        for n_dims in (2, 3, 4)
    }

    # Detection tolerances (see symmetries.neuron_types)
    ALIGN_TOL = 1e-3
    CONSTANT_TOL = 1e-6
    ZERO_TOL = 1e-6
    return (
        ALIGN_TOL,
        CONFIGS,
        CONSTANT_TOL,
        LOSS_EVERY,
        N_SEEDS,
        N_STEPS,
        SEED,
        TARGET_LOSS,
        ZERO_TOL,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training sweeps
    """)
    return


@app.cell
def _(CONFIGS, LOSS_EVERY, N_SEEDS, N_STEPS, SEED, TARGET_LOSS):
    @mo.persistent_cache
    def run_sweep(n_dims, n_hidden):
        models, _, converged = training.seed_sweep(
            n_seeds=N_SEEDS,
            n_hidden=n_hidden,
            activation="relu",
            n_steps=N_STEPS,
            target_loss=TARGET_LOSS,
            loss_every=LOSS_EVERY,
            dataset=parity.parity_dataset(n_dims),
            key=jax.random.fold_in(
                jax.random.PRNGKey(SEED), 1000 * n_dims + n_hidden
            ),
        )
        # Cache raw parameter arrays: the activation function inside
        # the Equinox models is not picklable
        W1 = np.stack([np.asarray(m.layers[0].weight) for m in models])
        b1 = np.stack([np.asarray(m.layers[0].bias) for m in models])
        W2 = np.stack([np.asarray(m.layers[2].weight) for m in models])
        b2 = np.stack([np.asarray(m.layers[2].bias) for m in models])
        return (W1, b1, W2, b2), converged

    sweeps = {}
    for _n_dims, _widths in CONFIGS.items():
        for _n_hidden in _widths:
            _t0 = time.perf_counter()
            (_W1, _b1, _W2, _b2), _converged = run_sweep(_n_dims, _n_hidden)
            _models = [
                mlp.mlp_from_params(
                    _W1[_idx],
                    _b1[_idx],
                    _W2[_idx],
                    _b2[_idx],
                    activation="relu",
                )
                for _idx in range(len(_converged))
            ]
            _minutes = (time.perf_counter() - _t0) / 60
            print(
                f"d={_n_dims}, width={_n_hidden}: {_minutes:.2f} min, "
                f"converged: {np.mean(_converged) * 100:.1f} %"
            )
            sweeps[(_n_dims, _n_hidden)] = (_models, _converged)
    # ru_maxrss is bytes on macOS, KiB on Linux
    sweep_peak_rss_gb = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    )
    print(f"peak RSS: {sweep_peak_rss_gb:.2f} GB")
    return (sweeps,)


@app.cell(hide_code=True)
def _(sweeps):
    _lines = [
        "**Convergence rates** (loss below target within the step budget)",
        "",
        "| $d$ | Width | Converged |",
        "| --- | ----- | --------- |",
    ]
    _lines += [
        f"| {_n_dims} | {_n_hidden} | {np.mean(_converged) * 100:.1f} % |"
        for (_n_dims, _n_hidden), (_, _converged) in sweeps.items()
    ]
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Neuron-type detection

    Each hidden neuron $(\mathbf{w}_i, b_i, a_i)$ is reduced to
    scaling-invariant coordinates: the unit vector $\mathbf{u}_i =
    (\mathbf{w}_i, b_i) / \lVert (\mathbf{w}_i, b_i) \rVert$ and the
    effective readout $\alpha_i = a_i \lVert (\mathbf{w}_i, b_i)
    \rVert$. Within each converged network, neurons are grouped by
    complete-linkage clustering of the sign-invariant (axial) distance
    between unit vectors, and groups are classified via their
    aggregate effective readouts into the ReLU taxonomy: duplicate
    groups, zero groups, linear and linear-duplicate groups, plus
    individual constant neurons ($\mathbf{w} \approx \mathbf{0}$).
    Constant-odd symmetries do not apply to ReLU and are not searched
    for. Only converged runs are analyzed.
    """)
    return


@app.cell
def _(ALIGN_TOL, CONSTANT_TOL, ZERO_TOL, sweeps):
    converged_models = {
        cfg: [model for model, ok in zip(models, converged, strict=True) if ok]
        for cfg, (models, converged) in sweeps.items()
    }

    reports = {
        cfg: [
            neuron_types.detect_neuron_types(
                model,
                align_tol=ALIGN_TOL,
                constant_tol=CONSTANT_TOL,
                zero_tol=ZERO_TOL,
            )
            for model in models
        ]
        for cfg, models in converged_models.items()
    }
    return converged_models, reports


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tolerance diagnostics

    The grouping and zero-readout thresholds are only meaningful if
    the corresponding quantities are bimodal: near-exact structure
    well below threshold, everything else well above. Left: pairwise
    axial distances between neurons within a network, pooled over all
    converged networks. Right: per-neuron effective readout magnitudes
    relative to the largest readout in the network. Dashed lines mark
    the detection thresholds.
    """)
    return


@app.cell
def _(ALIGN_TOL, CONFIGS, ZERO_TOL, reports):
    _FLOOR = 1e-16

    _fig = plt.figure(figsize=(8, 4))
    _ax0 = _fig.add_subplot(1, 2, 1)
    _ax1 = _fig.add_subplot(1, 2, 2)

    for _n_dims in CONFIGS:
        _pair_dists = []
        _rel_readouts = []
        for (_d, _), _reps in reports.items():
            if _d != _n_dims:
                continue
            for _rep in _reps:
                _iu = np.triu_indices(len(_rep.units), k=1)
                _pair_dists.append(
                    neuron_types.axial_distances(_rep.units)[_iu]
                )
                _scale = np.abs(_rep.effective_readouts).max()
                if _scale > 0:
                    _rel_readouts.append(
                        np.abs(_rep.effective_readouts) / _scale
                    )
        if not _pair_dists:
            continue
        _ax0.hist(
            np.log10(np.maximum(np.concatenate(_pair_dists), _FLOOR)),
            bins=60,
            histtype="step",
            label=f"$d = {_n_dims}$",
        )
        _ax1.hist(
            np.log10(np.maximum(np.concatenate(_rel_readouts), _FLOOR)),
            bins=60,
            histtype="step",
            label=f"$d = {_n_dims}$",
        )

    _ax0.axvline(np.log10(ALIGN_TOL), color="k", ls="--", lw=1)
    _ax0.set_xlabel(r"$\log_{10}$ axial distance")
    _ax0.set_ylabel("Count")
    _ax0.set_yscale("log")
    _ax0.legend(frameon=False)

    _ax1.axvline(np.log10(ZERO_TOL), color="k", ls="--", lw=1)
    _ax1.set_xlabel(r"$\log_{10} \, |\alpha| \, / \max |\alpha|$")
    _ax1.set_yscale("log")

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prevalence of symmetry structures

    Fraction of converged networks containing at least one group of
    each type, as a function of the overparameterization factor
    (width / $d$). The last panel shows the fraction of networks
    carrying *any* symmetry structure (i.e., at least one non-ordinary
    neuron).
    """)
    return


@app.cell
def _(CONFIGS, reports):
    _TYPES = ("duplicate", "zero", "linear", "linear_duplicate", "constant")

    _fig, _axes = plt.subplots(2, 3, figsize=(9, 5.5), sharey=True)

    for _panel, _type in enumerate(_TYPES):
        _ax = _axes.flat[_panel]
        _col = neuron_types.NEURON_TYPES.index(_type)
        for _n_dims, _widths in CONFIGS.items():
            _x = [_n_hidden / _n_dims for _n_hidden in _widths]
            _rates = []
            for _n_hidden in _widths:
                _counts = neuron_types.count_group_types(
                    reports[(_n_dims, _n_hidden)]
                )
                _rates.append(
                    float((_counts[:, _col] > 0).mean())
                    if _counts.shape[0]
                    else np.nan
                )
            _ax.plot(_x, _rates, marker="o", label=f"$d = {_n_dims}$")
        _ax.set_title(_type.replace("_", " "))

    _ax_any = _axes.flat[-1]
    for _n_dims, _widths in CONFIGS.items():
        _x = [_n_hidden / _n_dims for _n_hidden in _widths]
        _rates = []
        for _n_hidden in _widths:
            _reps = reports[(_n_dims, _n_hidden)]
            _rates.append(
                float(
                    np.mean(
                        [
                            (_rep.neuron_types != "ordinary").any()
                            for _rep in _reps
                        ]
                    )
                )
                if _reps
                else np.nan
            )
        _ax_any.plot(_x, _rates, marker="o", label=f"$d = {_n_dims}$")
    _ax_any.set_title("any structure")

    for _ax in _axes.flat:
        _ax.set_ylim(-0.05, 1.05)
    for _ax in _axes[-1, :]:
        _ax.set_xlabel("Width / $d$")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("Fraction of networks")
    _axes.flat[0].legend(frameon=False)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Redundant width

    Number of hidden neurons that collapsing all detected structure
    would remove (group of $k$ duplicates $\to$ 1 neuron, zero groups,
    linear groups, and constant neurons $\to$ 0 neurons). Collapsing a
    linear or linear-duplicate group leaves a linear residual; the
    reduction is only exact when these residuals cancel across groups,
    which `residual_linear_norm` quantifies (see summary below the
    figure).
    """)
    return


@app.cell
def _(CONFIGS, reports):
    _fig, _ax = plt.subplots(figsize=(4.5, 3.5))

    for _n_dims, _widths in CONFIGS.items():
        _x = [_n_hidden / _n_dims for _n_hidden in _widths]
        _means = []
        _stds = []
        for _n_hidden in _widths:
            _reps = reports[(_n_dims, _n_hidden)]
            if not _reps:
                _means.append(np.nan)
                _stds.append(np.nan)
                continue
            _excess = [
                len(_rep.neuron_types) - _rep.n_effective for _rep in _reps
            ]
            _means.append(float(np.mean(_excess)))
            _stds.append(float(np.std(_excess)))
        _ax.errorbar(
            _x,
            _means,
            yerr=_stds,
            marker="o",
            capsize=3,
            label=f"$d = {_n_dims}$",
        )

    _ax.set_xlabel("Width / $d$")
    _ax.set_ylabel("Collapsible neurons")
    _ax.legend(frameon=False)

    plt.show()
    return


@app.cell(hide_code=True)
def _(CONFIGS, reports):
    _RESIDUAL_TOL = 1e-6

    _lines = [
        "**Linear residuals.** Fraction of converged networks whose",
        "mixed aligned/opposite groups leave a non-cancelling linear",
        f"residual (relative norm above ${_RESIDUAL_TOL:g}$):",
        "",
    ]
    for _n_dims in CONFIGS:
        _fracs = []
        for (_d, _), _reps in reports.items():
            if _d != _n_dims:
                continue
            _fracs += [
                float(
                    _rep.residual_linear_norm
                    > _RESIDUAL_TOL
                    * max(np.abs(_rep.effective_readouts).max(), 1e-12)
                )
                for _rep in _reps
            ]
        if _fracs:
            _lines.append(f"- $d = {_n_dims}$: {np.mean(_fracs) * 100:.1f} %")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example solutions ($d = 2$)

    One converged XOR network per detected structure type (where
    found), with hidden hyperplanes overlaid on the output landscape.
    """)
    return


@app.cell
def _(converged_models, reports):
    _TYPES = ("duplicate", "zero", "linear", "linear_duplicate", "constant")
    _PLOT_DIM = 2  # only 2D networks can be visualized
    _D2_CONFIGS = [_cfg for _cfg in reports if _cfg[0] == _PLOT_DIM]

    _examples = []
    for _type in _TYPES:
        for _cfg in reversed(_D2_CONFIGS):  # widest first
            _found = False
            for _idx, _rep in enumerate(reports[_cfg]):
                if (_rep.neuron_types == _type).any():
                    _examples.append((_type, _cfg, _idx))
                    _found = True
                    break
            if _found:
                break

    if _examples:
        _fig, _axes = plt.subplots(
            1,
            len(_examples),
            figsize=(3 * len(_examples), 3),
            squeeze=False,
        )
        for _ax, (_type, _cfg, _idx) in zip(_axes.flat, _examples, strict=True):
            plotting.plot_mlp(
                converged_models[_cfg][_idx],
                ax=_ax,
                data=xor.xor_dataset(),
                data_kwargs={"cmap": mpl_colors.ListedColormap(["C0", "C1"])},
                hyperplane_kwargs={"normal_lw": 1.0},
            )
            _ax.set_title(f"{_type.replace('_', ' ')} (width {_cfg[1]})")
        plt.show()
    else:
        print("No symmetry structures detected in d = 2 networks.")
    return


if __name__ == "__main__":
    app.run()
