from typing import Callable, Protocol, Tuple, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

# DataType
DT = TypeVar("DT", np.bool_, np.int_, np.float_, int, float)

# NumericType
NT = TypeVar("NT", np.int_, np.float_, int, float)


class SpArr(Protocol[DT]):

    data: NDArray[DT]
    indices: NDArray[np.int32]
    indptr: NDArray[np.int32]
    shape: Tuple[int, ...]
    dtype: np.dtype[DT]

    astype: Callable[..., csr_matrix]

    def __matmul__(self, other: NDArray[DT] | csr_matrix) -> NDArray[DT] | csr_matrix:
        ...


DeArr: TypeAlias = NDArray[DT]
Arr: TypeAlias = DeArr[DT] | SpArr[DT]

BoolDeArr: TypeAlias = DeArr[np.bool_]
BoolSpArr: TypeAlias = SpArr[np.bool_]
BoolArr = TypeVar("BoolArr", BoolSpArr, BoolSpArr)

IntDeArr: TypeAlias = DeArr[np.int_]
IntSpArr: TypeAlias = SpArr[np.int_]
IntArr = TypeVar("IntArr", IntDeArr, IntSpArr)

RealDeArr: TypeAlias = DeArr[np.float_]
RealSpArr: TypeAlias = SpArr[np.float_]
RealArr = TypeVar("RealArr", RealDeArr, RealSpArr)

NumDeArr: TypeAlias = DeArr[NT]
NumSpArr: TypeAlias = SpArr[NT]
NumArr: TypeAlias = Arr[NT]
