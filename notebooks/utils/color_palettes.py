import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt

    from symmetries import colors

    plt.style.use("./style.mplstyle")
    return colors, plt


@app.cell
def _(colors, plt):
    palettes = colors.preview()
    plt.show()
    return (palettes,)


@app.cell
def _(colors, plt):
    system = colors.preview_system()
    plt.show()
    return (system,)


@app.cell
def _(colors, palettes):
    colors.simulate_deficiencies(palettes)
    return


@app.cell
def _(colors, system):
    colors.simulate_deficiencies(system)
    return


if __name__ == "__main__":
    app.run()
