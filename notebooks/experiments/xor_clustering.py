import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import resource
    import time

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.colors as mpl_colors
    import matplotlib.pyplot as plt
    import numpy as np

    from symmetries import clustering
    from symmetries import training
    from symmetries import xor

    plt.style.use("./style.mplstyle")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # XOR solution clustering

    Train two-hidden-unit networks on the XOR task across many random seeds,
    then cluster the converged solutions by their geometry.
    """)
    return


@app.cell
def _():
    N_SEEDS = 1000
    N_STEPS = int(1e7)
    TARGET_LOSS = 1e-12
    SEED = 0
    return N_SEEDS, N_STEPS, SEED, TARGET_LOSS


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ReLU networks
    """)
    return


@app.cell
def _(N_SEEDS, N_STEPS, SEED, TARGET_LOSS):
    _t0 = time.perf_counter()
    relu_trained, _relu_losses, relu_converged = training.seed_sweep(
        n_seeds=N_SEEDS,
        n_hidden=2,
        activation="relu",
        n_steps=N_STEPS,
        target_loss=TARGET_LOSS,
        key=jax.random.PRNGKey(SEED),
    )
    jax.tree.map(
        lambda y: (
            y.block_until_ready() if hasattr(y, "block_until_ready") else y
        ),
        (relu_trained, _relu_losses, relu_converged),
    )
    relu_sweep_minutes = (time.perf_counter() - _t0) / 60
    # ru_maxrss is bytes on macOS, KiB on Linux
    relu_peak_rss_gb = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    )
    print(
        f"ReLU sweep: {relu_sweep_minutes:.2f} min, "
        f"peak RSS: {relu_peak_rss_gb:.2f} GB"
    )
    return relu_converged, relu_trained


@app.cell(hide_code=True)
def _(relu_converged):
    mo.md(rf"""
    Converged runs: {jnp.array(relu_converged).mean() * 100:.2f} %
    """)
    return


@app.cell
def _(relu_converged, relu_trained):
    relu_converged_models = [
        m for m, ok in zip(relu_trained, relu_converged, strict=True) if ok
    ]

    (
        relu_angles,
        relu_distances,
        relu_gains,
        relu_output_weights,
        relu_output_biases,
    ) = clustering.standardize_mlp_solutions(relu_converged_models)

    relu_angles_flat = relu_angles.reshape(-1)
    relu_distances_flat = relu_distances.reshape(-1)

    RELU_N_BINS = 36

    _fig = plt.figure(figsize=(8, 4))
    _ax0 = _fig.add_subplot(1, 2, 1)
    _ax1 = _fig.add_subplot(1, 2, 2, projection="polar")

    _ax0.hist(relu_distances_flat)
    _ax0.set_xlabel("Distance")

    _ax1.hist(
        relu_angles_flat,
        bins=np.linspace(-np.pi, np.pi, RELU_N_BINS + 1),
    )
    _ax1.set_yticklabels([])

    plt.show()
    return (
        relu_angles,
        relu_converged_models,
        relu_distances,
        relu_distances_flat,
        relu_gains,
        relu_output_biases,
        relu_output_weights,
    )


@app.cell(hide_code=True)
def _(relu_distances_flat):
    mo.md(rf"""
    **Distribution of distances (ReLU)**

    - Minimum: ${np.min(relu_distances_flat):.2f}$
    - Median: ${np.median(relu_distances_flat):.2f}$
    - Maximum: ${np.max(relu_distances_flat):.2f}$
    """)
    return


@app.cell
def _(
    relu_angles,
    relu_distances,
    relu_gains,
    relu_output_biases,
    relu_output_weights,
):
    RELU_THRESHOLD_IDX = 0

    relu_X = clustering.prepare_mlp_for_clustering(
        relu_angles,
        relu_distances,
        relu_gains,
        relu_output_weights,
        relu_output_biases,
        standardize=True,
        quotient_hidden_rescaling=True,
    )
    relu_Z = clustering.compute_linkage(relu_X)

    relu_thresholds, _relu_gaps, _relu_n_clusters = (
        clustering.find_threshold_by_gap(relu_Z, top_k=5)
    )

    _fig, _ax = clustering.plot_dendrogram(
        relu_Z, threshold=relu_thresholds[RELU_THRESHOLD_IDX]
    )
    plt.show()
    return RELU_THRESHOLD_IDX, relu_X, relu_Z, relu_thresholds


@app.cell
def _(
    RELU_THRESHOLD_IDX,
    relu_X,
    relu_Z,
    relu_converged_models,
    relu_thresholds,
):
    _fig, _ax, _relu_reps = clustering.plot_cluster_representatives(
        relu_converged_models,
        relu_X,
        relu_Z,
        distance_threshold=relu_thresholds[RELU_THRESHOLD_IDX],
        n_reps=4,
        data=xor.xor_dataset(),
        data_kwargs={"cmap": mpl_colors.ListedColormap(["C0", "C1"])},
        hyperplane_kwargs={"normal_lw": 1.0},
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
