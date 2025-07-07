from typing import Literal, Optional, TypeAlias, Final, Set

import numpy as np

from ..arr.arr import RealArr
from ..arr.modify import add_to_arr_diag
from ..arr.normalize import normalize_arr

LaplacianType: TypeAlias = Literal["random_walk", "geometric", "symmetric"]
SYM_LAPLACIAN_TYPES: Final[Set[LaplacianType]] = {"symmetric"}
NON_SYM_LAPLACIAN_TYPES: Final[Set[LaplacianType]] = {"geometric", "random_walk"}


def eps_adjustment(eps: float) -> float:
    return 4.0 / (eps**2)


def laplacian(
    affs: RealArr,
    eps: Optional[float] = None,
    lap_type: LaplacianType = "geometric",
    diag_add: float = 1.0,
    aff_minus_id: bool = True,
    in_place: bool = False,
) -> RealArr:

    match lap_type:
        case "geometric":
            affs = normalize_arr(affs, sym_norm=True, in_place=in_place)
            lap = normalize_arr(affs, sym_norm=False, in_place=True)
        case "random_walk":
            lap = normalize_arr(affs, sym_norm=False, in_place=in_place)
        case "symmetric":
            lap = normalize_arr(affs, sym_norm=True, in_place=in_place, degree_exp=0.5)
        case _:
            raise ValueError(f"Unknown laplacian type: {lap_type}!")

    # data has been copied at this point even if in_place was False, so we can
    # operate in place.
    if not aff_minus_id:
        lap *= -1.0
    else:
        diag_add *= -1.0

    lap = add_to_arr_diag(lap, diag_add)

    if eps is not None:
        lap_data = lap if isinstance(lap, np.ndarray) else lap.data
        lap_data *= eps_adjustment(eps)

    return lap
