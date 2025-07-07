from typing import Optional, Tuple

import numpy as np
from einops import rearrange
from sortedcontainers import SortedDict

from ..arr.arr import RealArr, RealDeArr
from ..arr.index import AxLoc, is_ax_loc
from ..linalg import local_weighted_pca
from ..sampling import sample_array


class GradientGroupLasso:

    def __init__(
        self, dg_M, df_M, reg_l1s, reg_l2, max_iter, learning_rate, tol, beta0_npm=None
    ):

        n = dg_M.shape[0]
        d = dg_M.shape[1]
        m = df_M.shape[2]
        p = dg_M.shape[2]
        dummy_beta = np.ones((n, p, m))

        self.dg_M = dg_M
        self.df_M = df_M
        self.reg_l1s = reg_l1s
        self.reg_l2 = reg_l2
        self.beta0_npm = beta0_npm
        self.n = n
        self.p = p
        self.m = m
        self.d = d
        self.dummy_beta = dummy_beta

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.Tau = None
        self.alpha = 1.0
        self.lossresults = {}
        self.dls = {}
        self.l2loss = {}
        self.penalty = {}

    def _prox(self, beta_npm, thresh):
        """Proximal operator."""

        result = np.zeros(beta_npm.shape)
        result = np.asarray(result, dtype=float)
        for j in range(self.p):
            if np.linalg.norm(beta_npm[:, j, :]) > 0.0:
                potentialoutput = (
                    beta_npm[:, j, :]
                    - (thresh / np.linalg.norm(beta_npm[:, j, :])) * beta_npm[:, j, :]
                )
                posind = np.asarray(np.where(beta_npm[:, j, :] > 0.0))
                negind = np.asarray(np.where(beta_npm[:, j, :] < 0.0))
                po = beta_npm[:, j, :].copy()
                po[posind[0], posind[1]] = np.asarray(
                    np.clip(
                        potentialoutput[posind[0], posind[1]], a_min=0.0, a_max=1e15
                    ),
                    dtype=float,
                )
                po[negind[0], negind[1]] = np.asarray(
                    np.clip(
                        potentialoutput[negind[0], negind[1]], a_min=-1e15, a_max=0.0
                    ),
                    dtype=float,
                )
                result[:, j, :] = po
        return result

    def _grad_L2loss(self, beta_npm):

        df_M_hat = np.einsum("ndp,npm->ndm", self.dg_M, beta_npm)
        error = df_M_hat - self.df_M
        return np.einsum("ndm,ndp->npm", error, self.dg_M)

    def _L1penalty(self, beta_npm):

        beta_mn_p = rearrange(beta_npm, "n p m -> (m n) p")
        return np.linalg.norm(beta_mn_p, axis=0).sum()

    def _loss(self, beta_npm, reg_lambda):
        """Define the objective function for elastic net."""
        L = self._logL(beta_npm)
        P = self._L1penalty(beta_npm)
        return -L + reg_lambda * P

    def _logL(self, beta_npm):
        df_M_hat = np.einsum("ndp,npm -> ndm", self.dg_M, beta_npm)
        return -0.5 * np.linalg.norm((self.df_M - df_M_hat)) ** 2

    def _L2loss(self, beta_npm):
        return -self._logL(beta_npm)

    def fhatlambda(self, learning_rate, beta_npm_new, beta_npm_old):

        return (
            self._L2loss(beta_npm_old)
            + np.einsum(
                "npm,npm",
                self._grad_L2loss(beta_npm_old),
                (beta_npm_new - beta_npm_old),
            )
            + (1 / (2 * learning_rate))
            * np.linalg.norm(beta_npm_new - beta_npm_old) ** 2
        )

    def _btalgorithm(self, beta_npm, learning_rate, b, maxiter_bt, rl):

        grad_beta = self._grad_L2loss(beta_npm=beta_npm)
        for i in range(maxiter_bt):
            beta_npm_postgrad = beta_npm - learning_rate * grad_beta
            beta_npm_postgrad_postprox = self._prox(
                beta_npm_postgrad, learning_rate * rl
            )
            fz = self._L2loss(beta_npm_postgrad_postprox)
            fhatz = self.fhatlambda(learning_rate, beta_npm_postgrad_postprox, beta_npm)
            if fz <= fhatz:
                break
            learning_rate = b * learning_rate

        return (beta_npm_postgrad_postprox, learning_rate)

    def fit(self, beta0_npm=None):

        reg_l1s = self.reg_l1s
        n = self.n
        m = self.m
        p = self.p

        tol = self.tol
        np.random.RandomState(0)

        if beta0_npm is None:
            beta_npm_hat = 1 / (n * m * p) * np.random.normal(0.0, 1.0, [n, p, m])
        else:
            beta_npm_hat = beta0_npm

        fit_params = list()
        for l, rl in enumerate(reg_l1s):
            fit_params.append({"beta": beta_npm_hat})
            if l == 0:
                fit_params[-1]["beta"] = beta_npm_hat
            else:
                fit_params[-1]["beta"] = fit_params[-2]["beta"]

            beta_npm_hat = fit_params[-1]["beta"]
            L, DL, L2, PEN = list(), list(), list(), list()
            learning_rate = self.learning_rate
            beta_npm_hat_1 = beta_npm_hat.copy()
            beta_npm_hat_2 = beta_npm_hat.copy()
            for t in range(0, self.max_iter):
                L.append(self._loss(beta_npm_hat, rl))
                L2.append(self._L2loss(beta_npm_hat))
                PEN.append(self._L1penalty(beta_npm_hat))
                w = t / (t + 3)
                beta_npm_hat_momentumguess = beta_npm_hat + w * (
                    beta_npm_hat_1 - beta_npm_hat_2
                )

                beta_npm_hat, learning_rate = self._btalgorithm(
                    beta_npm_hat_momentumguess, learning_rate, 0.5, 1000, rl
                )
                beta_npm_hat_2 = beta_npm_hat_1.copy()
                beta_npm_hat_1 = beta_npm_hat.copy()

                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < tol:
                        print("converged", rl)
                        break

            fit_params[-1]["beta"] = beta_npm_hat
            self.lossresults[rl] = L
            self.l2loss[rl] = L2
            self.penalty[rl] = PEN
            self.dls[rl] = DL

        self.fit_ = fit_params
        return self


def group_lasso(
    Y: RealDeArr,
    X: RealDeArr,
    ncomp: int,
    lr: float = 100.0,
    l2_reg: float = 0.0,
    max_nlamb: int = 20,
    max_niter: int = 500,
    tol: float = 1e-12,
) -> Tuple[int, RealDeArr, RealDeArr, RealDeArr]:

    print("initializing lambda search")
    highprobes = np.asarray([])
    lowprobes = np.asarray([])

    ul = np.linalg.norm(
        np.einsum("n d m, n d p -> n p m ", Y, X), axis=tuple([0, 2])
    ).max()
    probe_init_low = 0.0
    probe_init_high = ul

    coeffs = SortedDict()
    combined_norms = SortedDict()

    GGL = GradientGroupLasso(
        X,
        Y,
        np.asarray([probe_init_low]),
        l2_reg,
        max_niter,
        lr,
        tol,
        beta0_npm=None,
    )
    GGL.fit()
    beta0_npm = GGL.fit_[-1]["beta"]
    coeffs[probe_init_low] = GGL.fit_[-1]["beta"]
    combined_norms[probe_init_low] = np.sqrt(
        (np.linalg.norm(coeffs[probe_init_low], axis=0) ** 2).sum(axis=1)
    )

    GGL = GradientGroupLasso(
        X,
        Y,
        np.asarray([probe_init_high]),
        l2_reg,
        max_niter,
        lr,
        tol,
        beta0_npm=beta0_npm,
    )
    GGL.fit(beta0_npm=beta0_npm)
    coeffs[probe_init_high] = GGL.fit_[-1]["beta"]
    combined_norms[probe_init_high] = np.sqrt(
        (np.linalg.norm(coeffs[probe_init_high], axis=0) ** 2).sum(axis=1)
    )

    cur_n_comp = len(
        np.where(~np.isclose(combined_norms[probe_init_high], 0.0, 1e-12))[0]
    )
    lowprobes = np.append(lowprobes, probe_init_low)

    if cur_n_comp == ncomp:
        print(
            "Selected functions",
            np.where(~np.isclose(combined_norms[probe_init_high], 0.0, 1e-12))[0],
        )
        probe = probe_init_high

    else:

        if cur_n_comp < ncomp:
            highprobes = np.append(highprobes, probe_init_high)
            probe = (lowprobes.max() + highprobes.min()) / 2
        if cur_n_comp > ncomp:
            lowprobes = np.append(lowprobes, probe_init_high)
            probe = lowprobes.max() * 2

        for i in range(max_nlamb):
            print(i, probe, "probe")
            beta0_npm = coeffs[lowprobes.max()]
            if not np.isin(probe, list(combined_norms.keys())):

                GGL = GradientGroupLasso(
                    X,
                    Y,
                    np.asarray([probe]),
                    l2_reg,
                    max_niter,
                    lr,
                    tol,
                    beta0_npm=beta0_npm,
                )
                GGL.fit(beta0_npm=beta0_npm)
                coeffs[probe] = GGL.fit_[-1]["beta"]
                combined_norms[probe] = np.sqrt(
                    (np.linalg.norm(coeffs[probe], axis=0) ** 2).sum(axis=1)
                )

            cur_n_comp = len(
                np.where(~np.isclose(combined_norms[probe], 0.0, 1e-12))[0]
            )
            if cur_n_comp == ncomp:
                print(
                    "Selected functions",
                    np.where(~np.isclose(combined_norms[probe], 0.0, 1e-12))[0],
                )
                break
            else:
                if cur_n_comp < ncomp:
                    highprobes = np.append(highprobes, probe)
                if cur_n_comp > ncomp:
                    lowprobes = np.append(lowprobes, probe)
                if len(highprobes) > 0:
                    probe = (lowprobes.max() + highprobes.min()) / 2
                else:
                    probe = lowprobes.max() * 2

                if i == max_nlamb - 1:
                    print("Failed to select d functions")
                    break

    solution_idx = coeffs.index(probe)
    lambdas = np.array(coeffs.keys())
    beta = np.array(coeffs.values())
    beta_norms = np.array(combined_norms.values())

    return solution_idx, lambdas, beta, beta_norms


def tslasso(
    x_pts: RealDeArr,
    grads: RealDeArr,
    ncomp: int,
    sample: AxLoc | int | float,
    needs_norm: bool = True,
    affs: Optional[RealArr] = None,
    tan_planes: Optional[RealDeArr] = None,
    in_place_norm: bool = False,
    bsize: Optional[int] = None,
    **kwargs,
) -> Tuple[int, RealDeArr, RealDeArr, RealDeArr]:

    if not is_ax_loc(sample):
        sample = sample_array(x_pts.shape[0], num_or_pct=sample)
    sample = np.sort(sample)

    npts = len(sample)

    if tan_planes is None:
        _, tan_planes = local_weighted_pca(
            x_pts, ncomp, sample, affs, needs_norm, in_place_norm, bsize,
        )

    grads = grads[sample]
    gamma = np.sqrt(np.sum(grads**2, axis=(0, 1), keepdims=True) / npts)
    grads = grads / (gamma + np.finfo(float).eps)

    proj_grads = np.einsum("mDf,mDd->mdf", grads, tan_planes)

    Y = np.expand_dims(np.identity(ncomp), axis=0)
    Y = np.repeat(Y, repeats=npts, axis=0)

    return group_lasso(Y, proj_grads, ncomp, **kwargs)
