import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    from functools import partial
    from pathlib import Path

    import jax.nn as jnn
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    OUTPUT_DIR = Path("../figures/templates/generated")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.cell
def _():
    y_range = 5.0
    x_to_y_ratio = 1.0
    x_range = x_to_y_ratio * y_range

    xs = jnp.linspace(-(x_range + 1.0), x_range + 1.0, num=1001)

    for name, fn in [
        ("relu", jnn.relu),  # even-linear
        ("tanh", jnn.tanh),  # constant-odd
        ("elu", partial(jnn.elu, alpha=3.0)),  # neither
    ]:
        ys = fn(xs)

        fig, ax = plt.subplots()
        ax.plot(xs, ys, c="k")
        plt.axvline(x=0, c="0.5", ls="--")
        plt.axhline(y=0, c="0.5", ls="--")

        ax.set_aspect("equal")
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(-y_range, y_range)
        ax.axis("off")
        fig.tight_layout(pad=0)

        fname = f"{name}.svg"
        fig.savefig(
            OUTPUT_DIR / fname,
            dpi="figure",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    return


if __name__ == "__main__":
    app.run()
