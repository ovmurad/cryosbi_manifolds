from itertools import product
from typing import Any, Dict, Optional, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from geometry_analysis.arr import AxIdx
from geometry_analysis.io import Database
from geometry_analysis.visualization import COLORS, normalize_color


DATASET = Database(database_name="igg_data_final")
RR_SCALE = 900.0


class Results:

    def __init__(self, name: str, ies_coords: Sequence[int]):

        self.name = name

        self.embedding = DATASET["lap_eigvecs"][f"{name}"]
        # self.relax_embedding = DATASET["lap_embedding"][f"{name}_relax"] / RR_SCALE

        # self.pred_embedding = DATASET["lap_embedding"][f"{name}_pred"]

        # self.sample_embedding = DATASET["lap_embedding"][f"{name}sample"]
        # self.sample_relax_embedding = DATASET["lap_embedding"][f"{name}sample_relax"] / RR_SCALE

        self.ies_coords = list(ies_coords)
        # self.estimated_d = DATASET[f"estimated_d|{name}|wc"]
        #
        self.conf = DATASET[f"params|{name}_conf"]
        self.snr = DATASET[f"params|{name}_snr"]
        #
        # self.sigma = DATASETf["params|{name}_sigma"]|wc
        # self.defoc = DATASETf["params|{name}_defoc"]|wc
        self.postm = DATASET[f"params|{name}_postm"]
        self.postw = DATASET[f"params|{name}_postw"]

        # self.shift = DATASET["params"][f"{name}_shift"]
        # self.shift1 = self.shift[:, 0]
        # self.shift2 = self.shift[:, 1]final_dataset["masks"][f"{data_name}_wc"] = mask
        #
        # self.rot = DATASET["params"][f"{name}_rot"]
        # self.rot1 = self.rot[:, 0]
        # self.rot2 = self.rot[:, 1]
        # self.rot3 = self.rot[:, 2]
        # self.rot4 = self.rot[:, 3]

        # self.ang = DATASET["params"][f"{name}_ang"]

        # self.umap = DATASET["params"][f"{name}_umap"]
        # self.umap1 = self.umap[:, 0]
        # self.umap2 = self.umap[:, 1]

        # self.color = COLORS[name]


sim_results = Results(
    name="sim",
    ies_coords=[0, 1, 2],
)
# sim_results = Results(name="exp", ies_coords=[0, 1, 3])


def _prepare_color_kwargs(
    results: Results,
    kwargs: Dict[str, Any],
    color: str | None,
    color_by: str | None,
    color_map: str | None,
    color_lims: Tuple[int, int] | Tuple[float, float] | None,
    npts: int | None,
) -> Dict[str, Any]:

    if color_by is not None:

        if color is None:
            color = getattr(results, color_by)

        if npts is not None:
            color = color[:npts]
        if color_lims is not None:
            color = normalize_color(color, *color_lims)

        color_map = "viridis" if color_map is None else color_map

        kwargs["c"] = color
        kwargs["cmap"] = color_map

    else:

        kwargs["color"] = getattr(results, "color") if color is None else color

    return kwargs


def plot_pairs(
    results: Results,
    data_key: str = "embedding",
    data_cols: Optional[AxIdx | str] = None,
    color: Optional[str] = None,
    color_by: Optional[str] = None,
    color_map: Optional[str] = None,
    color_lims: Optional[Tuple[int, int] | Tuple[float, float]] = None,
    x_lims: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Tuple[float, float]] = None,
    npts: Optional[int] = None,
    alpha: float = 0.5,
    size: float = 2.5,
):

    title = f"Pair plot of {results.name} {data_key} colored by {color_by}"

    data = getattr(results, data_key)
    if npts is not None:
        data = data[:npts]

    if data_cols is None:
        data_cols = slice(None)
    elif isinstance(data_cols, str):
        if data_cols == "ies":
            data_cols = getattr(results, "ies_coords")
            title += "(only IES coords)"
        else:
            raise ValueError("Only 'ies' can be a 'data_cols' key!")
    data_cols = np.r_[data_cols] if isinstance(data_cols, slice) else data_cols
    col_names = [f"x{d + 1}" for d in data_cols]

    df = pd.DataFrame(data=data[:, data_cols], columns=col_names)

    sub_plot_kws = {"alpha": alpha, "s": size}
    sub_plot_kws = _prepare_color_kwargs(
        results, sub_plot_kws, color, color_by, color_map, color_lims, npts
    )
    pair_plot_kws = {"height": 3.5, "aspect": 1.0, "diag_kind": "kde", "corner": True}

    pair_plot = sns.pairplot(df, plot_kws=sub_plot_kws, **pair_plot_kws)
    pair_plot.fig.suptitle(title)

    for i, j in product(range(len(pair_plot.axes)), repeat=2):
        if pair_plot.axes[i, j] is not None:
            pair_plot.axes[i, j].set_xlim(x_lims)
            pair_plot.axes[i, j].set_ylim(y_lims)

    plt.tight_layout()
    plt.show()


def plot_3d(
    ax: Axes3D,
    results: Results,
    data_key: str = "embedding",
    data_cols: Optional[AxIdx | str] = None,
    color: Optional[str] = None,
    color_by: Optional[str] = None,
    color_map: Optional[str] = None,
    color_lims: Optional[Tuple[int, int] | Tuple[float, float]] = None,
    x_lims: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Tuple[float, float]] = None,
    z_lims: Optional[Tuple[float, float]] = None,
    npts: Optional[int] = None,
    alpha: float = 0.5,
    size: float = 2.5,
):

    title = f"Plot of {results.name} {data_key} colored by {color_by}"

    data = getattr(results, data_key)
    if npts is not None:
        data = data[:npts]

    if data_cols is None:
        data_cols = slice(0, 3)
    elif isinstance(data_cols, str):
        if data_cols == "ies":
            data_cols = getattr(results, "ies_coords")
            title += "(only IES coords)"
        else:
            raise ValueError("Only 'ies' can be a 'data_cols' key!")

    data_cols = np.r_[data_cols] if isinstance(data_cols, slice) else data_cols
    data = data[:, data_cols]
    col_names = [f"x{d + 1}" for d in data_cols]

    plot_kws = {"alpha": alpha, "s": size}

    if results.name == "real" and color_by is not None:

        color = getattr(results, color_by)
        color_vals, color_counts = np.unique(color, return_counts=True)
        mean_val = color_vals[np.argmax(color_counts)]
        mean_mask = color == mean_val

        mean_data = data[mean_mask]
        ax.scatter(
            mean_data[:, 0],
            mean_data[:, 1],
            mean_data[:, 2],
            color="dimgray",
            **plot_kws,
        )
        data, color = data[~mean_mask], color[~mean_mask]

    plot_kws = _prepare_color_kwargs(
        results, plot_kws, color, color_by, color_map, color_lims, npts
    )

    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], **plot_kws)

    if color_by is not None:
        cbar = plt.colorbar(
            scatter, ax=ax, cax=plt.gcf().add_axes([0.675, 0.275, 0.01, 0.44])
        )
        # cbar = plt.colorbar(scatter, ax=ax, cax=plt.gcf().add_axes([0.65, 0.35, 0.01, 0.44]))
        # cbar = plt.colorbar(scatter, ax=ax, cax=plt.gcf().add_axes([0.65, 0.35, 0.01, 0.44]))
        cbar.set_label(color_by, rotation=270, labelpad=20, fontsize=15)

    ax.grid(False)

    if results.name == "real":
        ax.view_init(-90, 90, 90)
        # ax.view_init(-70, 60, 90)
    if results.name == "sim":
        # ax.view_init(90, -90, 0)
        ax.view_init(-150, -35, 0)

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_zlim(z_lims)

    # Disable grid
    ax.grid(False)

    # Disable axis
    ax._axis3don = False

    # Removing the background color
    ax.set_facecolor("white")

    # ax.set_xlabel(col_names[0])
    # ax.set_ylabel(col_names[1])
    # ax.set_zlabel(col_names[2])

    ax.set_title(title)


# Not used in the paper.
# for res in (sim_results, real_results):
#     for param in ("conf", "snr", "ang"):
#         plot_pairs(
#             res,
#             data_key="embedding",
#             color_by=param,
#             color_lims=(5, 95),
#             data_cols=range(0, 8),
#             x_lims=(-0.02, 0.02),
#             y_lims=(-0.02, 0.02),
#             npts=7500,
#         )


def color_by_figures():

    for res in (sim_results,):
        for param in ("conf", "snr", "postm", "postw"):

            fig = plt.figure(figsize=(20, 14))
            ax = fig.add_subplot(111, projection="3d")

            plot_3d(
                data_key="embedding",
                ax=ax,
                results=res,
                color_by=param,
                color_lims=(2, 98),
                data_cols="ies",
                x_lims=(-0.020, 0.020),
                y_lims=(-0.020, 0.020),
                z_lims=(-0.020, 0.020),
                alpha=0.6,
                size=2.5,
                color_map="viridis",
            )

            plt.tight_layout()
            plt.show()


def density_figures():

    for results in sim_results:

        fig = plt.figure(figsize=(20, 14))
        ax = fig.add_subplot(111, projection="3d")

        plot_3d(
            ax=ax,
            results=results,
            data_key="embedding",
            color="orange",
            data_cols="ies",
            x_lims=(-0.02, 0.02),
            y_lims=(-0.02, 0.02),
            z_lims=(-0.02, 0.02),
            alpha=0.25,
            size=1.5,
        )

        plot_3d(
            ax=ax,
            results=results,
            data_key="sample_embedding",
            color="royalblue",
            data_cols="ies",
            x_lims=(-0.02, 0.02),
            y_lims=(-0.02, 0.02),
            z_lims=(-0.02, 0.02),
            alpha=0.25,
            size=1.5,
        )

        plt.tight_layout()
        plt.show()


color_by_figures()
# density_figures()
