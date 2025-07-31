from typing import Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from ..arr.arr import RealArr, RealDeArr
from ..arr.normalize import normalize_arr
from ..arr.reduce import reduce_arr_to_degrees
from ..linalg.eigen_decomp import SYM_EIGEN_SOLVERS, EigenSolver, eigen_decomp
from .laplacian import (
    NON_SYM_LAPLACIAN_TYPES,
    SYM_LAPLACIAN_TYPES,
    LaplacianType,
    eps_adjustment,
    laplacian,
)

_DIAG_ADD = 2.0


def spectral_embedding(
    affs: RealArr,
    ncomp: int,
    eps: Optional[float] = None,
    lap_type: LaplacianType = "geometric",
    eigen_solver: EigenSolver = "amg",
    drop_first: bool = True,
    check_connected: bool = True,
    in_place: bool = False,
    **kwargs,
) -> Tuple[RealDeArr, RealDeArr]:

    if check_connected and connected_components(affs)[0] > 1:
        print("Graph is not fully connected: spectral embedding may not work properly!")

    # If the eigen solver requires PD matrices and if the laplacian is not a
    # symmetric one, we create a PD matrix that has the same eigvals as the
    # non-symmetric laplacian. We also ensure that the matrix is in fact PD by adding
    # IDENTITY_EPS. In general adding this should improve stability and at the end we
    # subtract the IDENTITY_EPS from the eigvals

    degrees = None

    if eigen_solver in SYM_EIGEN_SOLVERS and lap_type in NON_SYM_LAPLACIAN_TYPES:

        if lap_type == "geometric":
            affs = normalize_arr(affs, sym_norm=True, in_place=in_place)
            in_place = True

        degrees = reduce_arr_to_degrees(affs, axis=1, keepdims=True)

        lap_type: LaplacianType = "symmetric"

    lap = laplacian(
        affs=affs,
        eps=eps,
        lap_type=lap_type,
        diag_add=_DIAG_ADD + 1.0,
        aff_minus_id=False,
        in_place=in_place,
    )

    eigvals, eigvecs = eigen_decomp(
        arr=lap,
        ncomp=ncomp + int(drop_first),
        eigen_solver=eigen_solver,
        is_symmetric=lap_type in SYM_LAPLACIAN_TYPES,
        largest=False,
        **kwargs,
    )

    eigvals -= _DIAG_ADD if eps is None else (_DIAG_ADD * eps_adjustment(eps))

    if degrees is not None:
        eigvecs /= np.sqrt(degrees)
        eigvecs /= np.linalg.norm(eigvecs, axis=0)

    if drop_first:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]

    return eigvals, eigvecs
