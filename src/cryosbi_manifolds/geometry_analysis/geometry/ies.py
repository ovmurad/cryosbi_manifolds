from typing import Optional, Sequence, Tuple

import numpy as np
from megaman.ies import projected_volume, zeta_search

from ..arr.arr import RealDeArr, RealArr
from ..linalg import local_covariance, eigen_decomp
from ..sampling import sample_array


def ies(
    emb_pts: RealDeArr,
    emb_eigvals: RealDeArr,
    ds: int | Sequence[int],
    sample: Optional[int] = None,
    tan_spaces: Optional[RealDeArr] = None,
    lap: Optional[RealArr] = None,
    s: Optional[int] = None,
    bsize: Optional[int] = None,
) -> Tuple[Tuple[int, ...], ...]:

    ds = (ds,) if isinstance(ds, int) else ds

    if tan_spaces is None:

        if sample is None:
            mean_pts = emb_pts
        else:
            mean_pts = sample_array(emb_pts.shape[0], num_or_pct=sample)

        covs = local_covariance(
            emb_pts,
            mean_pts=mean_pts,
            weights=lap,
            needs_means=False,
            needs_norm=False,
            bsize=bsize,
        )
        _, tan_spaces = eigen_decomp(covs, ncomp=ds[-1])

    ies_axes = []

    for d in ds:

        zeta, _ = zeta_search(tan_spaces, emb_eigvals, intrinsic_dim=d, embedding_dim=s)

        proj_volume, all_comb = projected_volume(
            tan_spaces,
            intrinsic_dim=d,
            embedding_dim=s,
            eigen_values=emb_eigvals,
            zeta=zeta,
        )
        max_vol_comb = np.argmax(np.mean(proj_volume, axis=1))
        ies_axes.append(tuple(all_comb[max_vol_comb].tolist()))

    return tuple(ies_axes)
