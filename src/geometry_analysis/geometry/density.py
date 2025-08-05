from typing import Union, Dict, Optional

import numpy as np
from scipy.special import gamma

from geometry_analysis.arr import RealArr, RealDeArr, IntDeArr
from geometry_analysis.arr import reduce_arr_with_func


def score_from_knn_dist(knn_dist_train: RealArr, knn_dist_test: RealArr) -> RealDeArr:

    npts = knn_dist_train.shape[0]

    km_train = reduce_arr_with_func(knn_dist_train, axis=1, reduce_func=np.max)
    km_train = np.sort(km_train)

    km_test = reduce_arr_with_func(knn_dist_test, axis=1, reduce_func=np.max)

    return (npts - np.searchsorted(km_train, km_test)) / npts


def score_from_n_count(n_count_train: IntDeArr, n_count_test: IntDeArr) -> RealDeArr:

    npts = n_count_train.shape[0]

    n_count_train = np.sort(n_count_train)

    return np.searchsorted(n_count_train, n_count_test) / npts


def density_from_knn_dist(
    knn_dist: RealArr, k: int, d: Optional[Union[int, RealDeArr]] = 1
) -> RealDeArr:

    npts = knn_dist.shape[0]

    km = reduce_arr_with_func(knn_dist, axis=1, reduce_func=np.max)

    d = d.astype(int) if isinstance(d, np.ndarray) else d
    ball_vol = (np.pi ** (d / 2)) / gamma(1 + d / 2)
    return k / (npts * (ball_vol * (km**d)))


def density_from_n_count(
    n_count: IntDeArr, r: float, d: Optional[Union[int, RealDeArr]] = 1
) -> RealDeArr:

    npts = n_count.shape[0]

    d = d.astype(int) if isinstance(d, np.ndarray) else d
    ball_vol = (np.pi ** (d / 2)) / gamma(1 + d / 2)
    return n_count / (npts * (ball_vol * (r**d)))


def score_from_mult_knn_dist(
    knn_dist_train: Dict[int, RealArr], knn_dist_test: Dict[int, RealArr]
) -> RealArr:

    scores = None
    nks = 0

    for kd_train, kd_test in zip(knn_dist_train.values(), knn_dist_test.values()):

        k_scores = score_from_knn_dist(kd_train, kd_test)
        if scores is None:
            scores = k_scores
        else:
            scores += k_scores
        nks += 1

    scores /= nks
    return scores


def score_from_mult_n_count(
    n_count_train: Dict[float, IntDeArr], n_count_test: Dict[float, IntDeArr]
) -> RealDeArr:

    scores = None
    nks = 0

    for nc_train, nc_test in zip(n_count_train.values(), n_count_test.values()):

        r_scores = score_from_n_count(nc_train, nc_test)
        if scores is None:
            scores = r_scores
        else:
            scores += r_scores
        nks += 1

    scores /= nks
    return scores


def density_from_mult_knn_dist(
    knn_dists: Dict[int, RealArr], ds: Union[int, Dict[int, RealDeArr]] = 1
) -> RealDeArr:

    densities = None
    nks = 0

    for k, kd in knn_dists.items():

        d = ds if isinstance(ds, int) else ds[k]

        k_densities = density_from_knn_dist(kd, k, d)
        if densities is None:
            densities = k_densities
        else:
            densities += k_densities
        nks += 1

    densities /= nks
    return densities


def density_from_mult_n_count(
    n_count: Dict[float, IntDeArr], ds: Union[int, Dict[float, RealDeArr]] = 1
) -> RealDeArr:

    densities = None
    nks = 0

    for r, nc in n_count.items():

        d = ds if isinstance(ds, int) else ds[r]

        k_densities = density_from_n_count(nc, r, d)
        if densities is None:
            densities = k_densities
        else:
            densities += k_densities
        nks += 1

    densities /= nks
    return densities
