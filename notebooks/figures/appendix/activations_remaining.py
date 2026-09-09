import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import matplotlib.pyplot as plt
    from activations_even_linear import (
        export_activation_components_grid_variants,
    )

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    # Plotting defaults
    plt.style.use(["./style.mplstyle", "./appendix.mplstyle"])

    OUTPUT_DIR = Path("../figures/appendix")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Other activation functions (not even-linear or constant-odd)
    ACTIVATIONS = [
        "identity",
        ("celu", {"alpha": 3.0}),  # default: 1.0
        ("elu", {"alpha": 3.0}),  # default: 1.0
        "mish",
        "relu6",
        "selu",
    ]


@app.cell
def _():
    export_activation_components_grid_variants(
        ACTIVATIONS,
        output_dir=OUTPUT_DIR,
        output_stem="activations-remaining",
        fig_size=(8.0, 4.02),
        n_rows=2,
        n_cols=3,
        y_range=7.0,
        x_to_y_ratio=1.4,
        n_points=1001,
        z_orders=(2, 1, 3),
    )
    return


if __name__ == "__main__":
    app.run()
