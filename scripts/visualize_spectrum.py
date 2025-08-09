import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from geometry_analysis.io import Database

igg_dataset = Database(database_name="igg_data_final")
hem_dataset = Database(database_name="hemagglutinin_data_final")


def make_eigvals_df(evals, name):
    evals_df = pd.DataFrame(
        {"Eigenvalue Order": np.arange(1, len(evals) + 1), "Eigenvalue": evals}
    )
    evals_df["Dataset"] = name
    return evals_df


sim_eigvals_df = make_eigvals_df(hem_dataset["lap_eigvals"]["sim"], "hem_sim")
real_eigvals_df = make_eigvals_df(hem_dataset["lap_eigvals"]["exp"], "hem_exp")
eigvals_df = pd.concat([sim_eigvals_df, real_eigvals_df])

sns.set_style("darkgrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    eigvals_df,
    x="Eigenvalue Order",
    y="Eigenvalue",
    hue="Dataset",
    palette={"hem_exp": "red", "hem_sim": "blue", "igg_sim": "green"},
)
plt.xlabel("Eigenvalue Order")
plt.ylabel("Eigenvalues")
plt.legend()
plt.show()

eigvals_df = make_eigvals_df(igg_dataset["lap_eigvals"]["sim"], "igg_sim")

plt.figure(figsize=(12, 6))
sns.barplot(
    eigvals_df,
    x="Eigenvalue Order",
    y="Eigenvalue",
    hue="Dataset",
    palette={"hem_exp": "red", "hem_sim": "blue", "igg_sim": "green"},
)
plt.xlabel("Eigenvalue Order")
plt.ylabel("Eigenvalues")
plt.legend()
plt.show()
