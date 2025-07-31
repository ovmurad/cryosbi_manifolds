from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .laplacian import eps_adjustment
from .rmetric import local_rmetric
from ..arr.arr import RealArr, RealDeArr


def _loss_and_grad(
    emb_pts_flat: RealDeArr,
    lap: RealArr,
    lap_t: RealArr,
    weights: RealDeArr,
    eps_adjust: RealDeArr,
    npts: int,
    d: int,
    s: int,
) -> RealArr:

    emb_pts = np.reshape(emb_pts_flat, (npts, s))

    eigvals, eigvecs = local_rmetric(emb_pts, lap)

    eigvals[:, :d] -= 1.0
    eigvals *= eps_adjust
    eigvals_sq = eigvals**2
    argmax_eigvals = np.argmax(eigvals_sq, axis=1)

    eps_adjust = eps_adjust[argmax_eigvals]
    argmax_eigvals = np.expand_dims(argmax_eigvals, axis=1)

    max_eigvals = np.take_along_axis(eigvals, argmax_eigvals, axis=1).reshape(-1)
    spectral_norms = np.take_along_axis(eigvals_sq, argmax_eigvals, axis=1).reshape(-1)

    loss = np.average(spectral_norms, weights=weights)

    argmax_eigvals = np.expand_dims(argmax_eigvals, axis=2)
    max_eigvecs = np.take_along_axis(eigvecs, argmax_eigvals, axis=2)
    max_eigvecs = np.squeeze(max_eigvecs, axis=2)

    scalars = 2.0 * weights * max_eigvals * eps_adjust

    outer_eigvecs = np.einsum("np,n,nq->npq", max_eigvecs, scalars, max_eigvecs)

    # second minus term
    grads = lap_t @ np.einsum("np,npq->nq", emb_pts, outer_eigvecs)
    # first minus term
    grads += np.einsum("np,npq->nq", lap @ emb_pts, outer_eigvecs)

    outer_eigvecs = np.reshape(outer_eigvecs, (npts, s * s))
    outer_eigvecs = np.reshape(lap_t @ outer_eigvecs, (npts, s, s))

    # trace term
    grads -= np.einsum("np,npq->nq", emb_pts, outer_eigvecs)

    grads -= np.average(grads, axis=0, keepdims=True)
    grads = grads.reshape(-1)

    return loss, -grads


def riemannian_relaxation(
    emb_pts: RealDeArr,
    lap: RealArr,
    weights: Optional[RealDeArr] = None,
    lap_eps: Optional[float] = None,
    orth_eps: float = 0.0,
    d: Optional[int] = None,
    maxiter: int = 2000,
) -> RealDeArr:

    npts, s = emb_pts.shape
    d = s if d is None else d

    if weights is None:
        weights = (lap.diagonal() / eps_adjustment(lap_eps)) + 1.0
        weights /= weights.sum()

    lap_t = lap.transpose()

    eps_adjust = np.full(fill_value=1.0 / (1.0 + orth_eps**2), shape=s)
    if d < s:
        eps_adjust[d:] = 1.0 / (orth_eps**2)

    results = minimize(
        fun=_loss_and_grad,
        x0=emb_pts.reshape(-1),
        args=(lap, lap_t, weights, eps_adjust, npts, d, s),
        jac=True,
        options={"maxiter": maxiter, "disp": True},
        method="CG",
    )

    return np.reshape(results.x, (npts, s))
