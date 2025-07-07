from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes, Axes3D
from numpy.typing import NDArray

from ..arr.arr import RealDeArr
from ..arr.index import AxIdx


def make_radius_dict_df(
    radius_to_data: Dict[float, np.ndarray],
    data_name: str,
    dataset_key: Optional[Dict[str, str]] = None,
):

    radii = list(radius_to_data.keys())
    num_pts = len(radius_to_data[next(iter(radius_to_data.keys()))])

    radius_to_data = pd.DataFrame(
        {
            data_name: np.concatenate(list(radius_to_data.values())),
            "Radius": np.repeat(radii, num_pts),
        }
    )

    if dataset_key is not None:
        for k, v in dataset_key.items():
            radius_to_data[k] = v

    return radius_to_data


def _validate_columns(
    data: RealDeArr, columns: AxIdx | None, column_names: Sequence[str] | None
) -> Tuple[AxIdx, Sequence[str]]:

    columns = slice(0, data.shape[1]) if columns is None else None

    if column_names is None:
        columns_ = np.r_[columns] if isinstance(columns, slice) else columns
        column_names = [f"x{d + 1}" for d in columns_]

    return columns, column_names


def _parse_color_args(
    data: RealDeArr,
    color: int | str | RealDeArr | None,
    color_name: str | None,
    hue: Union[None, NDArray[np.str_], NDArray[np.int_]],
    hue_name: Union[None, str],
    cmap: str,
    palette: Union[None, Dict[Union[str, int], Union[str, NDArray[np.float_]]]],
) -> Dict[str, Any]:

    return_dict = {}

    if hue is None:

        if isinstance(color, int):
            return_dict["c"] = data[:, color]
            color_name = f"x{color + 1}" if color_name is None else color_name
        elif isinstance(color, np.ndarray):
            return_dict["c"] = color
        elif isinstance(color, (str, tuple)):
            return_dict["color"] = color
        elif color is not None:
            raise ValueError(
                "Unknown color type! Can be int(a column of the data), "
                "np.array(custom color) or str/tuple(plotlib color)."
            )

        if "c" in return_dict:
            return_dict["cmap"] = cmap
            return_dict["color_name"] = color_name
            return_dict["has_colorbar"] = True

    else:

        hue_vals = np.unique(hue)

        return_dict["hue"] = hue
        return_dict["hue_name"] = hue_name
        return_dict["hue_vals"] = hue_vals

        if palette is None:
            return_dict["palette"] = {hue_val: None for hue_val in hue_vals}
        else:
            if len(palette) != len(hue_vals):
                raise ValueError(
                    f"The given palette has keys: {tuple(palette.keys())}, but needs "
                    f"{tuple(hue_vals)}!"
                )
            return_dict["palette"] = palette

        return_dict["has_legend"] = True

    return return_dict


# def _make_pairplot(
#     df: pd.DataFrame,
#     cols: Optional[Sequence[str]] = None,
#     x_cols: Optional[Sequence[str]] = None,
#     y_cols: Optional[Sequence[str]] = None,
#     hue: Optional[str] = None,
#     hue_order: Optional[Tuple[Any, ...]] = None,
#     palette: Optional[Dict[str, Tuple[float, ...]] | str] = None,
#     kind: str = "scatter",
#     diag_kind: str = "kde",
#     height: float = 2.5,
#     aspect: float = 1.0,
#     corner: bool = True,
#     plot_kws: Optional[Dict[str, Any]] = None,
#     diag_kws: Optional[Dict[str, Any]] = None,
#     grid_kws: Optional[Dict[str, Any]] = None,
# ):
#
#     pair_plot = sns.pairplot(df, hue=hue, plot_kws=sub_plot_kws, **pair_plot_kws)


def make_pairplot(
    data: NDArray[np.float_],
    columns: Optional[Union[Iterable[int], slice]] = None,
    column_names: Optional[Iterable[str]] = None,
    color: Optional[Union[int, NDArray[np.float_], str]] = None,
    color_name: Optional[str] = None,
    hue: Optional[Union[int, NDArray[np.int_]]] = None,
    hue_name: Optional[str] = None,
    num_points: Optional[int] = None,
    axes_lims: Optional[Dict[str, Tuple[float, float]]] = None,
    sub_plot_kws: Optional[Dict[str, Any]] = None,
    pair_plot_kws: Optional[Dict[str, Any]] = None,
):

    if sub_plot_kws is None:
        sub_plot_kws = {}
    if pair_plot_kws is None:
        pair_plot_kws = {}
    if axes_lims is None:
        axes_lims = {}

    columns, column_names = _validate_columns(data, columns, column_names)
    data = data[:, columns]

    cmap = sub_plot_kws.get("cmap", None)
    palette = pair_plot_kws.get("palette", None)
    color_kwargs = _parse_color_args(
        data, color, color_name, hue, hue_name, cmap, palette
    )

    df = pd.DataFrame(data=data, columns=column_names)

    if hue is None:

        sub_plot_kws["color"] = color_kwargs.get("color", None)

        c = color_kwargs.get("c", None)
        sub_plot_kws["c"] = None if c is None else c[:num_points]
        sub_plot_kws["cmap"] = color_kwargs.get("cmap", None)

        df = df.head(num_points)

    else:

        df[hue_name] = hue

        if num_points is not None:
            num_points /= len(color_kwargs["hue_vals"])
            df = df.groupby(hue_name).apply(lambda x: x.head(num_points))
            df.reset_index(drop=True, inplace=True)

        pair_plot_kws["palette"] = color_kwargs["palette"]

    pair_plot = sns.pairplot(df, hue=hue_name, plot_kws=sub_plot_kws, **pair_plot_kws)

    for i, j in product(range(len(pair_plot.axes)), repeat=2):
        if pair_plot.axes[i, j] is not None:
            pair_plot.axes[i, j].set_xlim(x_lims)
            pair_plot.axes[i, j].set_ylim(axes_lims.get("y", None))

    return pair_plot


def make_3d_plot(
    ax: Axes3D,
    data: NDArray[np.float_],
    columns: Optional[Union[Sequence[int], slice]] = None,
    column_names: Optional[Sequence[str]] = None,
    color: Optional[Union[int, NDArray[np.float_], str]] = None,
    color_name: Optional[str] = None,
    hue: Optional[Union[int, NDArray[np.int_]]] = None,
    hue_name: Optional[str] = None,
    num_points: Optional[int] = None,
    axes_lims: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs,
):

    columns, column_names = _validate_columns(columns, column_names, )
    data = data[:num_points, columns]

    cmap = kwargs.get("cmap", None)
    palette = kwargs.pop("palette", None)
    color_kwargs = _parse_color_args(
        data, color, color_name, hue, hue_name, cmap, palette
    )

    if hue is None:

        kwargs["color"] = color_kwargs.get("color", None)

        c = color_kwargs.get("c", None)
        kwargs["c"] = None if c is None else c[:num_points]
        kwargs["cmap"] = color_kwargs.get("cmap", None)

        data = data[:num_points]
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], **kwargs)

        if color_kwargs.get("has_colorbar"):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_name, rotation=270, labelpad=20)

    else:

        df = pd.DataFrame(data=data, columns=column_names)
        df[hue_name] = hue

        if num_points is not None:
            num_points /= len(color_kwargs["hue_vals"])
            df = df.groupby(hue_name).apply(lambda x: x.head(num_points))
            df.reset_index(drop=True, inplace=True)

        data = df[column_names].to_numpy()
        hue = df[hue_name].to_numpy()

        for hue_val in color_kwargs["hue_vals"]:

            hue_data = data[hue == hue_val]

            kwargs["color"] = color_kwargs["palette"].get(hue_val, None)
            kwargs["label"] = hue_val

            ax.scatter(hue_data[:, 0], hue_data[:, 1], hue_data[:, 2], **kwargs)

        ax.legend(title=hue_name)

    ax.set_xlim(axes_lims.get("x", None))
    ax.set_ylim(axes_lims.get("y", None))
    ax.set_zlim(axes_lims.get("z", None))

    ax.set_xlabel(column_names[0])
    ax.set_ylabel(column_names[1])
    ax.set_zlabel(column_names[2])
