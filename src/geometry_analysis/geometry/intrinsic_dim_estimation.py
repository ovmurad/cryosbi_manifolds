from typing import Dict, Literal, Sequence, TypeAlias, Tuple

import numpy as np
from scipy.stats import linregress

from ..arr import IntDeArr, RealDeArr, Arr, reduce_arr_with_func

EigenGapMode: TypeAlias = Literal["softmax", "max"]


def correlation_dimension(
    n_counts: Dict[float, IntDeArr],
) -> Tuple[float, RealDeArr, RealDeArr]:

    radii = np.asarray(list(n_counts.keys()))
    n_counts = np.asarray(list(n_counts.values())) - 1

    log_radii = np.log(radii)
    log_mean_counts = np.log(n_counts.mean(axis=1))

    # Perform linear regression
    return linregress(x=log_radii, y=log_mean_counts)[0], log_radii, log_mean_counts


def doubling_dimension(n_counts: Dict[float, IntDeArr]) -> Dict[float, RealDeArr]:

    radii = np.array([r for r in n_counts.keys() if 2 * r in n_counts])

    estimates = {}

    for r in radii:

        counts_r = n_counts[r] - 1
        counts_2r = n_counts[2 * r] - 1

        mask = (counts_r > 0) & (counts_2r > 0)
        d_estim = np.full_like(counts_r, np.nan, dtype=np.float64)
        d_estim[mask] = np.log2(counts_2r[mask]) - np.log2(counts_r[mask])

        estimates[r] = d_estim

    return estimates


def levina_bickel(knn_dist: Dict[int, Arr]) -> Dict[int, RealDeArr]:

    estimates = {}

    for k, kd in knn_dist.items():

        nrows = kd.shape[0]
        diffs = np.diff(kd.indptr)
        row_mask = diffs == (k + 1)

        data_mask = np.repeat(row_mask, diffs)
        data_mask &= ~(np.repeat(np.arange(nrows), diffs) == kd.indices)

        maxr = reduce_arr_with_func(kd, axis=1, reduce_func=np.max)
        maxr = np.expand_dims(np.log(maxr[row_mask]), axis=1)

        rs = np.reshape(np.log(kd.data[data_mask]), (-1, k))
        r_diffs = maxr - rs

        d_estim = np.full(nrows, np.nan, dtype=np.float64)
        d_estim[row_mask] = (k - 2) / np.sum(r_diffs, axis=1)

        estimates[k] = d_estim

    return estimates


def slope_estimation(
    radii: Sequence[float], counts: Sequence[IntDeArr]
) -> Dict[str, float | RealDeArr]:
    """
    Estimates the intrinsic dimension using the slope in log2 n(R) = log2 R * slope
    + c. Here n(R) = avg_{x in dataset} n_x(R) is the average number of neighbors
    within radius R of points in the dataset. n_x(R) represents the number of points
    within radius R of some point x. Returns various statistics.

    :param radii: The radii corresponding to each of the counts.
    :param counts: A np array of shape nradii x npoints with the neighbor counts of
        each point at each radius.

    :return: A dictionary with analysis results.
    """

    radii = np.asarray(radii)
    counts = np.asarray(counts) - 1

    log_radii = np.log(radii)
    log_mean_counts = np.log(counts.mean(axis=1))

    # Perform linear regression
    slope, _, r_value, _, _ = linregress(x=log_radii, y=log_mean_counts)

    return {
        "log_radii": log_radii,
        "log_mean_counts": log_mean_counts,
        "slope": slope,
        "r_value": r_value,
    }


def log_ratio_estimation(
    radii: Sequence[float], counts: Sequence[IntDeArr]
) -> Dict[str, RealDeArr]:
    """
    Estimates the intrinsic dimension by using the method of computing the expected
    value of log2[n_x(2R)/n_x(R)]. n_x(R) represents the number of points within
    radius R of some point x. Returns various statistics.

    :param radii: The radii corresponding to each of the counts.
    :param counts: A np array of shape nradii x npoints with the neighbor counts of
        each point at each radius.

    :return: A dictionary with analysis results.
    """

    radii_with_ratios = np.array([r for r in radii if 2 * r in radii])
    log_ratios = np.array(
        [
            np.log2(counts[radii.index(2 * r)] / counts[radii.index(r)])
            for r in radii_with_ratios
        ]
    )

    return {
        "radii": radii_with_ratios,
        "log_ratios": log_ratios,
        "radii_exp_log_ratio": np.mean(log_ratios, axis=1),
        "exp_log_ratio": np.mean(log_ratios),
    }


def eigen_gap_estimation(eigvals: RealDeArr, mode: EigenGapMode = "max") -> RealDeArr:
    """
    Estimates the intrinsic dimension as d = argmax_d or softmax
    (lambda_d - lambda_{d+1}) where lambda_i represent the eigenvalues of the eigen
    decomp of the local weighted covariance matrix.

    :param eigvals: A dense n x ncomp array containing the eigen values at
        each point ordered in decreasing magnitude.
    :param mode: Type of estimation to use between max and softmax.

    :return: estimated_d: The estimate of d, depending on the mode in the config,
        for each point in the data set.
    """

    eiggaps = -np.diff(eigvals)

    if mode == "max":
        estimated_d = np.argmax(eiggaps, axis=1)
        estimated_d += 1  # add one because of 0 based index.
    elif mode == "softmax":
        eiggaps_exp = np.exp(eiggaps)
        eiggaps_weights = eiggaps_exp / np.sum(eiggaps_exp, axis=1, keepdims=True)
        ds = np.arange(1, eigvals.shape[1], dtype=float)
        estimated_d = np.sum(eiggaps_weights * ds, axis=1)
    else:
        raise ValueError(f"Unknown eigen gap mode {mode}!")

    return estimated_d
