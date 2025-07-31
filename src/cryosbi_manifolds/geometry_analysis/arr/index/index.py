from typing import List, Tuple, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

# Axis Index Types
AxMask = NDArray[np.bool_] | List[bool]
AxLoc = NDArray[np.integer] | List[int]
AxSlice = slice
AxIdx = TypeVar("AxIdx", AxMask, AxLoc, AxSlice)

# Box Index Types
BoxMask: TypeAlias = Tuple[AxMask, AxMask]
BoxLoc: TypeAlias = Tuple[AxLoc, AxLoc]
BoxSlice: TypeAlias = Tuple[AxSlice, AxSlice]
BoxIdx = TypeVar("BoxIdx", BoxMask, BoxLoc, BoxSlice)

# Coo Index Types
CooMask: TypeAlias = NDArray[np.bool_] | List[AxMask]
CooLoc: TypeAlias = Tuple[AxLoc, AxLoc]
CooIdx = TypeVar("CooIdx", CooMask, CooLoc)

# Csr Index
CsrIndptr: TypeAlias = NDArray[np.int32]
CsrIndices: TypeAlias = NDArray[np.int32]
CsrDiffs: TypeAlias = NDArray[np.int32]
CsrIdx: TypeAlias = Tuple[CsrIndices, CsrIndptr]

# All Index Types
Idx = TypeVar("Idx", AxMask, AxLoc, AxSlice, BoxMask, BoxLoc, BoxSlice, CooMask, CooLoc)

# Indices that can be transformed to csr indices
SpIdx = TypeVar("SpIdx", AxMask, AxLoc, CooMask, CooLoc)

