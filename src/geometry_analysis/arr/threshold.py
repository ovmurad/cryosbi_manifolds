import numpy as np

from .arr import NT, Arr, DeArr, SpArr
from .compress.de import compress_de_to_sp_with_coo_mask
from .compress.sp import compress_sp_axis
from .create import create_de_from_de_and_idx


def threshold_sp(arr: SpArr[NT], thresh: NT, in_place: bool = False) -> SpArr[NT]:
    """
    Applies a threshold to a sparse matrix. Elements greater than the threshold are
    removed from the matrix. 'arr' is modified in place when 'in_place' = True. 0s that
    are in the matrix are preserved.

    :param arr: Input sparse matrix to be processed.
    :param thresh: Threshold value above which the elements are removed.
    :param in_place: Determines if the threshold operation is applied in place.

    :return: Thresholded sparse matrix.
    """
    if isinstance(thresh, (int, np.integer)):

        rows = arr.shape[0]
        diffs = np.diff(arr.indptr)
        max_nnz = np.max(diffs)
        k = thresh + 1

        cols = np.expand_dims(np.arange(max_nnz), axis=0)
        diffs = np.expand_dims(diffs, axis=1)
        buffer_mask = cols < diffs

        buffer = np.full(shape=(rows, max_nnz), fill_value=np.inf)
        buffer[buffer_mask] = arr.data

        top_k_indices = np.argpartition(buffer, k, axis=1)[:, :k]

        idx_mask = np.zeros(buffer.shape, dtype=bool)
        idx_mask[np.repeat(np.arange(rows), k), top_k_indices.flatten()] = True

        mask = idx_mask[buffer_mask]

    else:
        mask = arr.data <= thresh

    return compress_sp_axis(arr, mask, in_place=in_place)


def threshold_de(
    arr: DeArr[NT], thresh: NT, return_sp: bool = True, in_place: bool = False
) -> Arr[NT]:
    """
    Thresholds a dense matrix with non-negative entries by setting the elements greater
    than a specified threshold to np.inf or greatest int value, if
    'return_sp' = False, or to implicit 0 if 'return_sp' = True. The sparse
    matrix will preserve true 0s. 'arr' is modified in place when 'in_place' = True.

    :param arr: The dense matrix to be thresholded.
    :param thresh: Values greater than this in the matrix will be set to zero for
        sparse matrices and np.inf or greatest int value for dense matrices.
    :param in_place: Determines if the threshold operation is applied in place in order
        to save memory.
    :param return_sp: If True, the function returns the thresholded array in sparse
        format.

    :return: Thresholded dense or sparse matrix depending on 'return_sp'.
    """
    if isinstance(thresh, (int, np.integer)):
        rows = arr.shape[0]
        k = thresh + 1

        top_k_indices = np.argpartition(arr, k, axis=1)[:, :k]

        mask = np.zeros(arr.shape, dtype=bool)
        mask[np.repeat(np.arange(rows), k), top_k_indices.flatten()] = True
    else:
        mask = arr <= thresh

    if return_sp and arr.ndim == 2:
        return compress_de_to_sp_with_coo_mask(arr, mask, in_place)
    return create_de_from_de_and_idx(arr, ~mask, in_place=in_place)


def threshold_arr(
    arr: Arr[NT], thresh: NT, return_sp: bool = True, in_place: bool = False
) -> Arr[NT]:
    if isinstance(arr, np.ndarray):
        return threshold_de(arr, thresh, return_sp, in_place)
    return threshold_sp(arr, thresh, in_place)
