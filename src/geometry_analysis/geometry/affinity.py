import numpy as np

from ..arr.arr import RealArr, RealDeArr
from ..arr.create import create_sp_from_data_and_csr_idx


def _de_affinity_in_place(dists: RealDeArr, eps: float, dist_is_sq: bool) -> RealDeArr:

    if not dist_is_sq:
        np.power(dists, 2, out=dists)

    np.divide(dists, (eps**2), out=dists)
    np.multiply(dists, -1, out=dists)
    np.exp(dists, out=dists)

    return dists


def _de_affinity_out_of_place(
    dists: RealDeArr, eps: float, dist_is_sq: bool
) -> RealDeArr:
    if dist_is_sq:
        return np.exp(-(dists / (eps**2)))
    return np.exp(-((dists / eps) ** 2))


def affinity(
    dists: RealArr, eps: float, dist_is_sq: bool = False, in_place: bool = False
) -> RealArr:
    """
    Computes gaussian affinity given a sparse or dense distance matrix. The operation
    is computed either in place or out of place depending on the 'in_place' flag.
    Depending on the 'dist_is_sq' flag, the dist matrix is assumed to contain
    squared distances or not.

    :param dists:
    :param eps:
    :param in_place:
    :param dist_is_sq:

    :return:
    """

    data = dists if isinstance(dists, np.ndarray) else dists.data

    if in_place:
        _de_affinity_in_place(data, eps, dist_is_sq)
        return dists

    data = _de_affinity_out_of_place(data, eps, dist_is_sq)

    if isinstance(dists, np.ndarray):
        return data

    csr_idx = (dists.indices.copy(), dists.indptr.copy())
    return create_sp_from_data_and_csr_idx(data, csr_idx, dists.shape)
