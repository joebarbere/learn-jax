import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from learnjax import rosetta
    return mo, rosetta


@app.cell
def _(mo, rosetta):
    pick = mo.ui.dropdown({f"{e.topic} · {e.name}": e.name for e in rosetta.ENTRIES},
                          value=f"{rosetta.ENTRIES[0].topic} · {rosetta.ENTRIES[0].name}", label="Operation")
    mo.vstack([mo.md("# Math ↔ code\n\nEvery operation the course introduces, as the formula, plain Python, "
                     "NumPy, JAX and PyTorch. The lessons show the same tabs where each is first taught; "
                     "docs/ROSETTA.md has them all on one page."), pick])
    return (pick,)


@app.cell
def _(mo, pick, rosetta):
    rosetta.tabs(mo, pick.value)
    return


if __name__ == "__main__":
    app.run()
