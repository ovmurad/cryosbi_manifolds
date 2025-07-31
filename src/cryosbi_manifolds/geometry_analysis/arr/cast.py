import numpy as np

from .arr import BoolSpArr, SpArr, DT, DeArr
from .create import create_de_from_data_and_idx, create_sp_from_data_and_csr_idx


def cast_sp_to_sp_bool(arr: SpArr, in_place: bool = False) -> BoolSpArr:

    indices = arr.indices if in_place else arr.indices.copy()
    indptr = arr.indptr if in_place else arr.indptr.copy()
    csr_idx = (indices, indptr)

    return create_sp_from_data_and_csr_idx(data=True, csr_idx=csr_idx, shape=arr.shape)


def cast_sp_to_de(arr: SpArr[DT], fill_value=None) -> DeArr[DT]:

    rows = np.repeat(np.arange(arr.shape[0]), repeats=np.diff(arr.indptr))
    coo_idx = (rows, arr.indices)

    return create_de_from_data_and_idx(arr.data, arr.shape, coo_idx, fill_value)
