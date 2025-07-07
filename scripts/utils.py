from typing import Union, Sequence, Dict, Optional, Tuple, Callable, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from src.geometry_analysis.arr import IntDeArr, DeArr, RealArr, NT, BoolDeArr
from src.geometry_analysis.arr import reduce_arr_with_func
from src.geometry_analysis.geometry import (
    count_x,
    dist,
    doubling_dimension,
    correlation_dimension,
    levina_bickel,
    score_from_mult_n_count,
    score_from_mult_knn_dist,
    density_from_mult_n_count,
)
from src.geometry_analysis.io import Database
from src.geometry_analysis.sampling import RANDOM_STATE
from src.geometry_analysis.utils import create_grid_1d


def _format_single_key(dn: str, mn: str) -> str:
    return dn if mn == "all" else f"{dn}_{mn}"


def _format_pair_key(
    dn: Union[str, Tuple[str, str]],
    mn: Union[str, Tuple[str, str]],
    reverse: bool = False,
) -> str:

    if isinstance(dn, tuple) and isinstance(mn, tuple):
        if reverse:
            return (
                f"{_format_single_key(dn[1], mn[1])}-{_format_single_key(dn[0], mn[0])}"
            )
        else:
            return (
                f"{_format_single_key(dn[0], mn[0])}-{_format_single_key(dn[1], mn[1])}"
            )
    key = _format_single_key(dn, mn)
    return f"{key}-{key}"


# ------------------------------ UTILS ------------------------------


def index_to_mask(npts: int, index: IntDeArr) -> BoolDeArr:
    mask = np.zeros(npts, dtype=bool)
    mask[index] = True
    return mask


def compute_split_masks(
    dataset: Database,
    data_key: str,
    split_npts: Dict[str, int],
) -> None:

    npts = dataset["points"][data_key].shape[0]

    split_names = list(split_npts.keys())
    split_npts = tuple(split_npts.values())

    assert sum(split_npts) <= npts, "Sum of splits cannot exceed number of points."

    permuted = RANDOM_STATE.permutation(npts)
    split_points = np.cumsum(split_npts)
    split_indices = np.split(permuted, split_points)[:-1]

    for split_name, split_index in zip(split_names, split_indices):
        dataset["masks"][f"{data_key}_{split_name}"] = index_to_mask(npts, split_index)


def compute_distance_statistics(
    dataset: Database,
    data_name: Union[str, Tuple[str, str]],
    mask_name: Union[str, Tuple[str, str]],
    rs: Optional[Tuple[float, ...]] = None,
    ks: Optional[Tuple[int, ...]] = None,
) -> None:

    pair_key = _format_pair_key(data_name, mask_name)
    if isinstance(data_name, str) and isinstance(mask_name, str):
        x_pts = dataset[f"points|{data_name}|{mask_name}"]
        y_pts = None
    else:
        x_pts = dataset[f"points|{data_name[0]}|{mask_name[0]}"]
        y_pts = dataset[f"points|{data_name[1]}|{mask_name[1]}"]

    if rs is not None:
        dataset["n_counts"][pair_key] = count_x(x_pts=x_pts, y_pts=y_pts, threshold=rs)
    if ks is not None:
        dataset["knn_dists"][pair_key] = dist(x_pts=x_pts, y_pts=y_pts, threshold=ks)


def detect_outliers(
    dataset: Database,
    data_name: Union[str, Tuple[str, str]],
    mask_name: Union[str, Tuple[str, str]],
    threshold: float,
    algo: str = "n_counts",
) -> BoolDeArr:

    if isinstance(data_name, str) and isinstance(mask_name, str):
        pair_key_self = pair_key_other = _format_pair_key(data_name, mask_name)
        single_key = _format_single_key(data_name, mask_name)
        cur_mask_key = f"{single_key}"
    else:
        pair_key_self = _format_pair_key(data_name[0], mask_name[0])
        pair_key_other = _format_pair_key(data_name, mask_name, reverse=True)
        cur_mask_key = _format_single_key(data_name[1], mask_name[1])

    if algo == "n_counts":
        n_count_self = dataset["n_counts"][pair_key_self]
        n_count_other = dataset["n_counts"][pair_key_other]
        pts_scores = score_from_mult_n_count(n_count_self, n_count_other)
    elif algo == "knn_dists":
        knn_dist_self = dataset["knn_dists"][pair_key_self]
        knn_dist_other = dataset["knn_dists"][pair_key_other]
        pts_scores = score_from_mult_knn_dist(knn_dist_self, knn_dist_other)
    else:
        raise ValueError(f"Unknown scoring algorithm: {algo}!")

    mask = dataset["masks"][cur_mask_key]
    mask[mask] = pts_scores >= threshold
    return mask


def compute_uniform_sample(
    dataset: Database, data_name: str, mask_name: str, sample_size: int
) -> None:

    # first estimate the local intrinsic dimension at each point using the doubling dimension
    # doubling dimension is the most robust to non-uniform density and hence a good choice before resampling the data
    single_key = _format_single_key(data_name, mask_name)
    pair_key = _format_pair_key(data_name, mask_name)

    n_count = dataset["n_counts"][pair_key]
    pw_estim_d = doubling_dimension(n_count)

    # where ever we have nans, we replace with the average value
    for pw_r_estim_d in pw_estim_d.values():
        nan_mask = np.isnan(pw_r_estim_d)
        pw_r_estim_d[nan_mask] = np.mean(pw_r_estim_d[~nan_mask])

    # estimate the density
    n_count = {r: n_count[r] for r in pw_estim_d.keys()}
    estim_density = density_from_mult_n_count(n_count, pw_estim_d)

    # invert the density and normalize
    inv_density = 1.0 / (estim_density + 1e-16)
    inv_density = inv_density / np.sum(inv_density)

    # sample data uniformly from the remaining clean points
    cur_mask = dataset["masks"][single_key]
    unif_index = RANDOM_STATE.choice(
        np.flatnonzero(cur_mask), size=sample_size, replace=False, p=inv_density
    )

    # create and save the mask corresponding to the new uniform dataset
    unif_mask = np.zeros_like(cur_mask)
    unif_mask[unif_index] = True
    dataset["masks"][f"{data_name}_unif"] = unif_mask


def perform_classification(
    dataset: Database,
    train_name: str,
    test_name: str,
    thresholds: Sequence[float],
    algo: str = "n_counts",
) -> Dict[float, float]:

    npts = np.sum(dataset["masks"][f"{test_name}_test"])
    results = {}
    for threshold in thresholds:
        non_outlier_mask = detect_outliers(
            dataset,
            data_name=(train_name, test_name),
            mask_name=("train", "test"),
            threshold=threshold,
            algo=algo,
        )
        results[threshold] = np.sum(non_outlier_mask) / npts
    return results


def subsample_dataset(
    in_dataset: Database,
    out_dataset: Database,
    mask_name: str,
    key_pairs: Tuple[Tuple[str, str], ...],
) -> None:
    for dir_name, data_name in key_pairs:
        masked_input_key = f"{dir_name}|{data_name}|{mask_name}"
        out_dataset[dir_name][data_name] = in_dataset[masked_input_key]


# ------------------------------ VISUALIZATION ------------------------------


def plot_hist_with_percentile_lines(
    data: DeArr,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    q_args: Tuple[int, int, int] = (10, 90, 10),
    nbins: int = 100,
) -> None:

    qs = create_grid_1d(*q_args, scale="int")
    ps = np.percentile(data, q=qs)

    show_plot = ax is None
    ax = sns.histplot(data, bins=nbins, stat="density", kde=False, ax=ax)

    for p, q in zip(ps, qs / 100.0):
        ax.axvline(p, color="red", linestyle="--", linewidth=1)
        ax.text(
            p,
            ax.get_ylim()[1] * 0.95,
            f"{int(q * 100)}%",
            ha="center",
            va="center",
            fontsize=6,
            rotation=90,
            color="red",
        )

    ax.set_title(title, size=9)
    ax.set_xlim((np.min(data), np.percentile(data, q=98)))

    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_mult_hist_with_percentile_lines(
    data: Dict[NT, DeArr],
    title: Callable[[Any], str] = lambda x: f"Key = {x}",
    q_args: Tuple[int, int, int] = (10, 90, 10),
    nbins: int = 100,
) -> None:

    num_plots = len(data)

    ncols = min(num_plots, 3)
    nrows = int(np.ceil(num_plots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
    )

    for idx, (label, d) in enumerate(data.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        plot_hist_with_percentile_lines(d, ax, title(label), q_args, nbins)

    for i in range(num_plots, nrows * ncols):
        fig.delaxes(axes[i // ncols][i % ncols])

    plt.tight_layout()
    plt.show()


def plot_n_count_or_knn_dist(
    n_count: Optional[Dict[float, IntDeArr]] = None,
    knn_dist: Optional[Dict[int, RealArr]] = None,
):

    if n_count is not None:
        plot_mult_hist_with_percentile_lines(
            data=n_count, title=lambda x: f"Radius = {x}"
        )
    if knn_dist is not None:
        max_knn_dist = {
            k: reduce_arr_with_func(kd, axis=1, reduce_func=np.max)
            for k, kd in knn_dist.items()
        }
        plot_mult_hist_with_percentile_lines(
            data=max_knn_dist, title=lambda x: f"K = {x}"
        )


# ------------------------------ ROUTINES ------------------------------
def run_dimensionality_estimation(
    n_count: Optional[Dict[float, IntDeArr]] = None,
    knn_dist: Optional[Dict[int, RealArr]] = None,
    algos: Tuple[str, ...] = ("dd", "cd", "lb"),
):

    if "dd" in algos:

        dim_estimates_dict = doubling_dimension(n_count)
        dim_estimates = np.stack(list(dim_estimates_dict.values()))

        for r, de in dim_estimates_dict.items():
            r_de = np.mean(de[~np.isnan(de)])
            print(f"Doubling Dimension Estimate for Radius = {r}: {r_de}")

        de = np.mean(dim_estimates[~np.isnan(dim_estimates)])
        print(f"Doubling Dimension Estimate: {de}")

    if "cd" in algos:
        cd_dim = correlation_dimension(n_count)[0]
        print(f"Correlation Dimension Estimate: {cd_dim}")

    if "lb" in algos:
        lb_dim_estimates = {
            r: de[~np.isnan(de)] for r, de in levina_bickel(knn_dist).items()
        }

        lb_dim = 0.0
        for r, de in lb_dim_estimates.items():
            r_lb_dim = np.mean(de)
            lb_dim += 1.0 / r_lb_dim
            print(f"Levina-Bickel Dimension Estimate for K = {r}: {r_lb_dim}")

        lb_dim = 1.0 / (
            lb_dim / len(lb_dim_estimates)
        )  # MacKay - Ghahramani correction
        print(f"Levina-Bickel Dimension Estimate: {lb_dim}")
