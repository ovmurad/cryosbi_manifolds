from typing import Iterator

import numpy as np

from .arr import Arr, DeArr, SpArr
from .index.index import AxLoc
from .create import create_sp_from_data_and_csr_idx


def _nbatches(arr_len: int, batch_size: int) -> int:
    return int(np.ceil(arr_len / batch_size))


def iter_de_batches(arr: DeArr | AxLoc, bsize: int) -> Iterator[DeArr]:
    """
    Splits dense data arrays into smaller arrays (batches) with maximum size being
    batch_size and returns an iterator over the batch.

    :param arr: A numpy array that needs to be split into batches.
    :param bsize: The maximum size for each batch(last batch could be smaller).

    :yield: Batches of the original data as numpy arrays.
    """

    # If the data array length is less than or equal to the batch size,
    # yield the entire array as a single batch.
    if len(arr) <= bsize:
        yield arr
    else:
        for b in range(_nbatches(len(arr), bsize)):
            # Yield a slice from data array from the b x batch_size to the start index
            # of the next batch
            yield arr[b * bsize : (b + 1) * bsize]


def iter_sp_batches(arr: SpArr, bsize: int) -> Iterator[SpArr]:
    """
    Splits a sparse data array into smaller batches with maximum size being batch_size
    and returns a generator of csr_matrices. The data and indices in these matrices will
    be views of the data in the original matrix, while the indptr will be copies
    starting from 0(unless the 'batch_size' is greater than the nrows of the array).

    :param arr: A sparse csr_matrix that needs to be split into batches.
    :param bsize: The maximum size for each batch(last batch could be smaller).

    :yield: Batches of the original data as csr_matrices with indptr starting from 0.
    """

    if arr.shape[0] <= bsize:
        yield arr
    else:
        for b in range(_nbatches(arr.shape[0], bsize)):

            indptr = arr.indptr[b * bsize : (b + 1) * bsize + 1]

            indptr_start = indptr[0]
            indptr_stop = indptr[-1]

            yield create_sp_from_data_and_csr_idx(
                data=arr.data[indptr_start:indptr_stop],
                csr_idx=(arr.indices[indptr_start:indptr_stop], indptr - indptr_start),
                shape=(indptr.shape[0] - 1, arr.shape[1]),
            )


def iter_arr_batches(arr: Arr, bsize: int) -> Iterator[Arr]:
    if isinstance(arr, np.ndarray):
        return iter_de_batches(arr, bsize)
    return iter_sp_batches(arr, bsize)
