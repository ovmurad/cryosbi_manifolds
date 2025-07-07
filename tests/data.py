import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import ortho_group
from sklearn.metrics import pairwise_distances
from src.geometry_analysis.arr.index.transforms import ax_idx_to_ax_mask
from src.geometry_analysis.sampling.constants import RANDOM_STATE

from utils import cast_to_de, mask_de

NROWS = 100
NCOLS = 20
SIZE = NROWS * NCOLS
SHAPE = (NROWS, NCOLS)

EPS = 1.2
NON_TRIVIAL_THRESHOLDS = (1.9, 2.1)
THRESHOLDS = (-1.0,) + NON_TRIVIAL_THRESHOLDS + (np.inf,)

LAPLACIAN_TYPES = ("geometric", "random_walk", "symmetric")
SUBSPACE_DIM = NCOLS // 4


def _sample_mask(size):
    mask = RANDOM_STATE.randint(0, 3, size=size).astype(np.bool_)
    return mask


def _sample_slice(ax_len):
    start = RANDOM_STATE.randint(0, 5)
    end = RANDOM_STATE.randint(ax_len - 5, ax_len)
    step = RANDOM_STATE.randint(1, round(ax_len**0.25))
    return slice(start, end, step)


def _sample_tan_space(nspaces=None):
    if nspaces is None:
        return ortho_group.rvs(dim=NCOLS)[:, :SUBSPACE_DIM]
    return np.stack([_sample_tan_space() for _ in range(nspaces)], axis=0)


def _make_lists(indices):
    def _to_list(idx):
        if isinstance(idx, tuple):
            return tuple(i.tolist() for i in idx)
        return idx.tolist()

    return tuple(_to_list(idx) if RANDOM_STATE.rand() < 0.5 else idx for idx in indices)


def _make_thresh_dists_dict(x_pts, y_pts, thresholds):

    de_dists = {None: pairwise_distances(x_pts, y_pts)}

    for t in thresholds:
        mask = de_dists[None] > t
        de_dists[t] = mask_de(de_dists[None], mask=mask)

    sp_dists = {
        t: (mask_de(ddx, rm_value=np.inf, return_sp=True) if t is not None else ddx)
        for t, ddx in de_dists.items()
    }

    return de_dists, sp_dists


def _gaussian_affinity(dists, eps):
    if isinstance(dists, np.ndarray):
        return np.exp(-(dists**2) / (eps**2))
    else:
        affs = _gaussian_affinity(dists.data, eps)
        return csr_matrix(
            (affs, dists.indices.copy(), dists.indptr.copy()), shape=dists.shape
        )


def _laplacian(affs, lap_type):

    if isinstance(affs, csr_matrix):
        affs = np.asarray(affs.todense())
        inp_is_sp = True
    else:
        inp_is_sp = False

    if lap_type == "geometric":
        deg = np.sum(affs, axis=1, keepdims=True)
        affs = affs / deg / deg.flatten()
        affs = affs / np.sum(affs, axis=1, keepdims=True)
    elif lap_type == "random_walk":
        affs = affs / np.sum(affs, axis=1, keepdims=True)
    elif lap_type == "symmetric":
        deg = np.sqrt(np.sum(affs, axis=1, keepdims=True))
        affs = affs / deg / deg.flatten()

    lap = affs - np.identity(affs.shape[0])

    if inp_is_sp:
        return mask_de(lap, rm_value=0.0, return_sp=True)
    return lap


class Arr:

    de_mask = _sample_mask(SHAPE)
    sp_mask = csr_matrix(de_mask)

    remove_rc_mask = sp_mask.copy()
    remove_rc_mask[[0, NROWS // 2, NROWS - 1]] = False
    remove_rc_mask[:, [0, NCOLS // 2, NCOLS - 1]] = False

    # Mask with some empty rows and columns
    remove_rc_mask = csr_matrix(remove_rc_mask)

    # Data indices we will make 0, but not remove from the sparse array
    zero_data_idx = np.random.choice(sp_mask.nnz, sp_mask.nnz // 10, replace=False)

    # bool arrays

    bool_1d = _sample_mask(NROWS)
    de_bool_2d = _sample_mask(SHAPE)
    sp_bool = sp_mask.copy()
    sp_bool.data[zero_data_idx] = False
    sp_bool_empty = csr_matrix(SHAPE, dtype=np.bool_)
    sp_bool_empty_rc = remove_rc_mask.multiply(sp_bool).tocsr()

    # int arrays

    int_1d = RANDOM_STATE.randint(0, 100, NROWS)
    de_int_2d = RANDOM_STATE.randint(0, 100, SHAPE)
    sp_int = sp_bool.multiply(de_int_2d).tocsr()
    sp_int.data[zero_data_idx] = 0
    sp_int_empty = csr_matrix(SHAPE, dtype=np.int_)
    sp_int_empty_rc = remove_rc_mask.multiply(sp_int).tocsr()

    # real arrays

    real_1d = RANDOM_STATE.rand(NROWS)
    de_real_2d = RANDOM_STATE.rand(*SHAPE)
    sp_real = sp_bool.multiply(de_real_2d).tocsr()
    sp_real.data[zero_data_idx] = 0.0
    sp_real_empty = csr_matrix(SHAPE, dtype=np.float_)
    sp_real_empty_rc = remove_rc_mask.multiply(sp_real).tocsr()

    # combination arrays

    num_arrs_1d = (int_1d, real_1d)
    arrs_1d = (bool_1d,) + num_arrs_1d

    num_de_arrs_2d = (de_int_2d, de_real_2d)
    de_arrs_2d = (de_bool_2d,) + num_de_arrs_2d

    de_arrs = arrs_1d + de_arrs_2d

    bool_sp_arrs = (sp_bool, sp_bool_empty, sp_bool_empty_rc)
    int_sp_arrs = (sp_int, sp_int_empty, sp_int_empty_rc)
    real_sp_arrs = (sp_real, sp_real_empty, sp_real_empty_rc)
    num_sp_arrs = int_sp_arrs + real_sp_arrs
    sp_arrs = bool_sp_arrs + num_sp_arrs

    num_sp_non_empty_arrs = (sp_int, sp_real)
    sp_non_empty_arrs = (sp_bool,) + num_sp_non_empty_arrs

    # points arrays

    x_pts = de_real_2d.copy()
    y_pts = RANDOM_STATE.rand(NROWS // 2, NCOLS)

    # distance arrays

    x_de_dists, x_sp_dists = _make_thresh_dists_dict(x_pts, None, THRESHOLDS)
    xy_de_dists, xy_sp_dists = _make_thresh_dists_dict(x_pts, y_pts, THRESHOLDS)

    de_lap_dists = {r: d for r, d in x_de_dists.items() if r in NON_TRIVIAL_THRESHOLDS}
    sp_lap_dists = {r: d for r, d in x_sp_dists.items() if r in NON_TRIVIAL_THRESHOLDS}

    # affinities

    x_de_affs = {r: _gaussian_affinity(dists, EPS) for r, dists in x_de_dists.items()}
    xy_de_affs = {r: _gaussian_affinity(dists, EPS) for r, dists in xy_de_dists.items()}

    x_sp_affs = {r: _gaussian_affinity(dists, EPS) for r, dists in x_sp_dists.items()}
    xy_sp_affs = {r: _gaussian_affinity(dists, EPS) for r, dists in xy_sp_dists.items()}

    de_lap_affs = {r: _gaussian_affinity(d, EPS) for r, d in de_lap_dists.items()}
    sp_lap_affs = {r: _gaussian_affinity(d, EPS) for r, d in sp_lap_dists.items()}

    # laplacians

    de_laps = {
        r: {lt: _laplacian(affs, lt) for lt in LAPLACIAN_TYPES}
        for r, affs in de_lap_affs.items()
    }
    sp_laps = {
        r: {lt: _laplacian(affs, lt) for lt in LAPLACIAN_TYPES}
        for r, affs in sp_lap_affs.items()
    }

    geometric_de_laps = tuple(lap["geometric"] for lap in de_laps.values())
    geometric_sp_laps = tuple(lap["geometric"] for lap in sp_laps.values())
    geometric_laps = geometric_de_laps + geometric_sp_laps

    de_weights = tuple(de_lap_affs.values())
    sp_weights = tuple(sp_lap_affs.values())
    weights = de_weights + sp_weights

    tan_space = _sample_tan_space()

    x_pts_subspace = np.einsum("nD,Dd,Ed->nE", x_pts, tan_space, tan_space)

    f1_vals = np.einsum("nD,Dd->nd", x_pts_subspace, tan_space)
    f1_vals = np.sum(f1_vals, axis=1)
    f2_vals = 2.0 * f1_vals - 3.0
    f_vals = np.stack([f1_vals, f2_vals], axis=1)

    x_pts_subspace += np.random.normal(scale=0.01, size=x_pts_subspace.shape)

    _uniform_weights = np.full(fill_value=1.0 / NROWS, shape=(NROWS,), dtype=np.float_)
    _first_pt_weights = de_weights[0][0]

    _zero_mean = np.zeros(NCOLS, dtype=np.float_)
    _first_pt_mean = x_pts[0]
    _x_pts_mean = np.mean(x_pts, axis=0)
    _x_pts_weighted_mean = np.sum(x_pts * _first_pt_weights[:, None], axis=0)

    # mean_pt, weight, exp_mean_pt, exp_weights
    mean_pt_and_weights = (
        (None, None, _zero_mean, _uniform_weights),
        (None, _first_pt_weights, _zero_mean, _first_pt_weights),
        (0, None, _first_pt_mean, _uniform_weights),
        (0, _first_pt_weights, _first_pt_mean, _first_pt_weights),
        (True, None, _x_pts_mean, _uniform_weights),
        (True, _first_pt_weights, _x_pts_weighted_mean, _first_pt_weights),
        (_first_pt_mean, None, _first_pt_mean, _uniform_weights),
        (_first_pt_mean, _first_pt_weights, _first_pt_mean, _first_pt_weights),
    )

    _first_subsp_pt_mean = x_pts_subspace[0]
    _x_subsp_pts_mean = np.mean(x_pts_subspace, axis=0)
    _x_subsp_pts_weighted_mean = np.sum(x_pts_subspace * _first_pt_weights[:, None], axis=0)

    subsp_mean_pt_and_weights = (
        (None, None, _zero_mean, _uniform_weights),
        (None, _first_pt_weights, _zero_mean, _first_pt_weights),
        (0, None, _first_subsp_pt_mean, _uniform_weights),
        (0, _first_pt_weights, _first_subsp_pt_mean, _first_pt_weights),
        (True, None, _x_subsp_pts_mean, _uniform_weights),
        (True, _first_pt_weights, _x_subsp_pts_weighted_mean, _first_pt_weights),
        (_first_subsp_pt_mean, None, _first_subsp_pt_mean, _uniform_weights),
        (_first_subsp_pt_mean, _first_pt_weights, _first_subsp_pt_mean, _first_pt_weights),
    )

    _mean_pts_locs = [0, 2, 4, 6, 8, 10]
    _mean_pts_sl = slice(10, 60, 3)
    _half_pts_sl = slice(0, NROWS, 2)

    _de_weights = de_weights[0]
    _sp_weights = sp_weights[0]
    _sp_weights_to_de = cast_to_de(_sp_weights, fill_value=0)

    _x_pts_uniform_weights = np.expand_dims(_uniform_weights, axis=0)
    _x_pts_uniform_weights = np.repeat(_x_pts_uniform_weights, axis=0, repeats=NROWS)

    _locs_uniform_weights = _x_pts_uniform_weights[_mean_pts_locs]
    _locs_de_weights = _de_weights[_mean_pts_locs]
    _locs_sp_weights = _sp_weights[_mean_pts_locs]
    _locs_sp_weights_to_de = _sp_weights_to_de[_mean_pts_locs]

    _pts_uniform_weights = _x_pts_uniform_weights[_mean_pts_sl]
    _pts_de_weights = _de_weights[_mean_pts_sl]
    _pts_sp_weights = _sp_weights[_mean_pts_sl]
    _pts_sp_weights_to_de = _sp_weights_to_de[_mean_pts_sl]

    _half_de_weights = _de_weights[_half_pts_sl]
    _half_sp_weights = _sp_weights[_half_pts_sl]
    _half_sp_weights_to_de = _sp_weights_to_de[_half_pts_sl]

    _locs_mean = x_pts[_mean_pts_locs]
    _pts_mean = x_pts[_mean_pts_sl]
    _half_de_mean = _half_de_weights @ x_pts
    _half_sp_mean = _half_sp_weights @ x_pts

    _locs_subsp_mean = x_pts_subspace[_mean_pts_locs]
    _pts_subsp_mean = x_pts_subspace[_mean_pts_sl]
    _half_subsp_de_mean = _half_de_weights @ x_pts_subspace
    _half_subsp_sp_mean = _half_sp_weights @ x_pts_subspace

    _de_lap = geometric_de_laps[0]
    _sp_lap = geometric_sp_laps[0]
    _sp_lap_to_de = cast_to_de(_sp_lap, fill_value=0)

    _locs_de_lap = _de_lap[_mean_pts_locs]
    _locs_sp_lap = _sp_lap[_mean_pts_locs]
    _locs_sp_lap_to_de = _sp_lap_to_de[_mean_pts_locs]

    _pts_de_lap = _de_lap[_mean_pts_sl]
    _pts_sp_lap = _sp_lap[_mean_pts_sl]
    _pts_sp_lap_to_de = _sp_lap_to_de[_mean_pts_sl]

    _half_de_lap = _de_lap[_half_pts_sl]
    _half_sp_lap = _sp_lap[_half_pts_sl]
    _half_sp_lap_to_de = _sp_lap_to_de[_half_pts_sl]

    _laps = (
        (_de_lap, _de_lap),
        (_sp_lap, _locs_sp_lap_to_de),
        (_de_lap, _locs_de_lap),
        (_sp_lap, _locs_sp_lap_to_de),
        (_pts_de_lap, _pts_de_lap),
        (_pts_de_lap, _pts_de_lap),
        (_pts_sp_lap, _pts_sp_lap_to_de),
        (_half_de_lap, _half_de_lap),
        (_half_sp_lap, _half_sp_lap_to_de),
    )

    _x_pts_tan_spaces = _sample_tan_space(NROWS)
    _locs_tan_spaces = _x_pts_tan_spaces[_mean_pts_locs]
    _pts_tan_spaces = _x_pts_tan_spaces[_mean_pts_sl]
    _half_tan_spaces = _x_pts_tan_spaces[_half_pts_sl]

    _tan_spaces = (
        (_x_pts_tan_spaces, _x_pts_tan_spaces),
        (_x_pts_tan_spaces, _locs_tan_spaces),
        (_x_pts_tan_spaces, _locs_tan_spaces),
        (_x_pts_tan_spaces, _locs_tan_spaces),
        (_pts_tan_spaces, _pts_tan_spaces),
        (_pts_tan_spaces, _pts_tan_spaces),
        (_pts_tan_spaces, _pts_tan_spaces),
        (_half_tan_spaces, _half_tan_spaces),
        (_half_tan_spaces, _half_tan_spaces),
    )

    mean_pts_and_weights = (
        (None, None, x_pts, _x_pts_uniform_weights),
        (_mean_pts_locs, None, _locs_mean, _locs_uniform_weights),
        (_mean_pts_locs, _de_weights, _locs_mean, _locs_de_weights),
        (_mean_pts_locs, _sp_weights, _locs_mean, _locs_sp_weights_to_de),
        (_pts_mean, None, _pts_mean, _pts_uniform_weights),
        (_pts_mean, _pts_de_weights, _pts_mean, _pts_de_weights),
        (_pts_mean, _pts_sp_weights, _pts_mean, _pts_sp_weights_to_de),
        (None, _half_de_weights, _half_de_mean, _half_de_weights),
        (None, _half_sp_weights, _half_sp_mean, _half_sp_weights_to_de),
    )
    mean_pts_weights_and_laps = (
        mpt_and_w[0:2] + laps[0:1] + mpt_and_w[2:4] + laps[1:2]
        for mpt_and_w, laps in zip(mean_pts_and_weights, _laps)
    )
    mean_pts_weights_and_tan_spaces = (
        mpt_and_w[0:2] + ts[0:1] + mpt_and_w[2:4] + ts[1:2]
        for mpt_and_w, ts in zip(mean_pts_and_weights, _tan_spaces)
    )
    subsp_mean_pts_weights_and_f_vals = (
        (None, None, f1_vals, f1_vals, x_pts_subspace, _x_pts_uniform_weights, f1_vals),
        (_mean_pts_locs, None, f_vals, f_vals, _locs_subsp_mean, _locs_uniform_weights, f_vals[_mean_pts_locs]),
        (_mean_pts_locs, _de_weights, f1_vals, f1_vals, _locs_subsp_mean, _locs_de_weights, f1_vals[_mean_pts_locs]),
        (_mean_pts_locs, _sp_weights, f_vals, f_vals, _locs_subsp_mean, _locs_sp_weights_to_de, f_vals[_mean_pts_locs]),
        (_pts_subsp_mean, None, f1_vals[_mean_pts_sl], f1_vals, _pts_subsp_mean, _pts_uniform_weights, f1_vals[_mean_pts_sl]),
        (_pts_subsp_mean, _pts_de_weights, f_vals[_mean_pts_sl], f_vals, _pts_subsp_mean, _pts_de_weights, f_vals[_mean_pts_sl]),
        (_pts_subsp_mean, _pts_sp_weights, f1_vals[_mean_pts_sl], f1_vals, _pts_subsp_mean, _pts_sp_weights_to_de, f1_vals[_mean_pts_sl]),
        (None, _half_de_weights, f_vals[_half_pts_sl], f_vals, _half_subsp_de_mean, _half_de_weights, f_vals[_half_pts_sl]),
        (None, _half_sp_weights, f1_vals[_half_pts_sl], f1_vals, _half_subsp_sp_mean, _half_sp_weights_to_de, f1_vals[_half_pts_sl]),
    )


class Index:

    # Slices

    a0_slices = (_sample_slice(NROWS), slice(2, NROWS, 3), slice(None), slice(0, 0))
    a1_slices = (_sample_slice(NCOLS), slice(1, NCOLS, 4), slice(None), slice(0, 0))
    an_slices = (_sample_slice(SIZE), slice(None, SIZE, 3), slice(None), slice(0, 0))
    box_slices = tuple(zip(a0_slices, a1_slices))

    # Masks

    a0_masks = _make_lists((ax_idx_to_ax_mask(sl, NROWS) for sl in a0_slices))
    a1_masks = _make_lists((ax_idx_to_ax_mask(sl, NCOLS) for sl in a1_slices))
    an_masks = _make_lists((ax_idx_to_ax_mask(sl, SIZE) for sl in an_slices))
    box_masks = tuple(zip(a0_masks, a1_masks))
    coo_masks = _make_lists((np.asarray(m).reshape(SHAPE) for m in an_masks))

    # Locs

    a0_locs = _make_lists(np.flatnonzero(m) for m in a0_masks)
    a1_locs = _make_lists(np.flatnonzero(m) for m in a1_masks)
    an_locs = _make_lists(np.flatnonzero(m) for m in an_masks)
    box_locs = tuple(zip(a0_locs, a1_locs))
    coo_locs = _make_lists(tuple(np.where(m)) for m in coo_masks)

    # Csr indices which match an_mask, an_locs, coo_mask, coo_locs

    csr_indices = tuple(csr_matrix((cl[0], cl), SHAPE) for cl in coo_locs)
    csr_indices = tuple((csr_mat.indices, csr_mat.indptr) for csr_mat in csr_indices)
    csr_indices = csr_indices + csr_indices

    # Combinations corresponding to types of selections.

    a0_indices = a0_masks + a0_locs + a0_slices
    a1_indices = a1_masks + a1_locs + a1_slices
    an_indices = an_masks + an_locs + an_slices

    box_indices = box_masks + box_locs + box_slices
    coo_indices = coo_masks + coo_locs

    an_sp_indices = an_masks + an_locs
    coo_sp_indices = coo_masks + coo_locs
