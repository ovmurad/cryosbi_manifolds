import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from src.geometry_analysis.io import Database
from src.geometry_analysis.utils import create_grid_1d

sns.set_style("darkgrid")

TS_LASSO_PCTS = create_grid_1d(start=0, stop=95, step_size=5, scale="int")
TS_LASSO_FUNCS = (
    "conf",
    "snr",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
    "shift_x",
    "shift_y",
    "sigma",
    "defoc",
)
DATASET = Database(database_name="igg_latent_vecs_256_final")

# real_results = {p: DATASET["tslasso"][f"real_{p}"] for p in TS_LASSO_PCTS}
sim_results = {p: DATASET["tslasso"][f"sim_{p}"] for p in TS_LASSO_PCTS}


def plot_tslasso_norms(df, selections, name):

    nrows, ncols = 2, 5

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=True
    )

    axes = axes.flatten()

    for i, (ax, func) in enumerate(zip(axes, TS_LASSO_FUNCS)):

        data = df[df["func"] == func]
        sns.lineplot(
            data,
            x="lambda",
            y="Bk_norm",
            style="selected",
            style_order=(True, False),
            hue="snr",
            ax=ax,
            palette="bwr",
            lw=1,
        )

        sel_pct = np.sum(selections == i) / selections.shape[0]
        ax.set_title(
            f"Function {func} (selected {sel_pct:.2f} pct of the time)",
            fontsize=8,
        )

        ax.set_xlim(0, 12)
        ax.set_ylim(-0.1, 80)

    # fig.suptitle(f"TSLasso on {name} data", fontsize=10)

    plt.tight_layout()
    plt.show()


for name, results in (("sim", sim_results),):

    selections = []
    for solution_idx, _, _, beta_norms in results.values():
        selections.append(np.flatnonzero(beta_norms[solution_idx]))
    selections = np.array(selections)

    print(f"{name.capitalize()} TSLasso Selections:")
    for snr_p, sel in zip(TS_LASSO_PCTS, selections):
        print(f"For snr percentile {snr_p}, the selections were: {tuple(sel)}.")

    print("\n")

    for i, func in enumerate(TS_LASSO_FUNCS):
        print(
            f"Function {func} was selected "
            f"{np.sum(selections == i) / selections.shape[0]} "
            f"percent of the trials."
        )

    print("\n")

    norms_df = []
    for (snr_p, (_, lambdas, _, beta_norms)), sel in zip(results.items(), selections):
        for lamb, lamb_norms in zip(lambdas, beta_norms):
            for i, norm in enumerate(lamb_norms):
                norms_df.append(
                    {
                        "snr": snr_p,
                        "lambda": lamb,
                        "func": TS_LASSO_FUNCS[i],
                        "Bk_norm": norm,
                        "selected": (i in sel),
                    }
                )
    norms_df = pd.DataFrame.from_records(norms_df)

    plot_tslasso_norms(norms_df, selections, name)
