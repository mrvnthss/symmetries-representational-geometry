import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import cycler
    import matplotlib.pyplot as plt
    from fig_a1 import plot_activation_components_grid

    from symmetries import colors

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    # Plotting defaults
    plt.style.use("./style.mplstyle")

    # Sample orange sequential colormap at discrete values
    cmap = colors.get_sequential_cmap("orange")
    orange_colors = [cmap(t) for t in [0.9, 0.6, 0.4]]
    plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=orange_colors)

    OUTPUT_PATH = Path("../figures/appendix/fig_a2.pdf")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Constant-odd activation functions
    ACTIVATIONS = [
        "hard_sigmoid",
        "hard_tanh",
        "sigmoid",
        "soft_sign",
        "sparse_sigmoid",
        "tanh",
    ]


@app.cell
def _():
    plot_activation_components_grid(
        ACTIVATIONS,
        fig_size=(8.0, 4.2),
        n_rows=2,
        n_cols=3,
        y_range=3.0,
        x_to_y_ratio=1.4,
        n_points=1001,
        z_orders=(2, 1, 3),
        output_path=OUTPUT_PATH,
    )
    return


if __name__ == "__main__":
    app.run()
