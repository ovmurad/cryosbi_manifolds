import numpy as np
from scipy.sparse import csr_matrix

from geometry_analysis.arr.cast import cast_sp_to_de
from geometry_analysis.arr.utils import get_fill_value


def _get_keep_mask_for_de(arr, mask, thresh, rm_value):

    if mask is not None:
        return ~mask
    elif thresh is not None:
        return arr <= thresh
    rm_value = get_fill_value(arr) if rm_value is None else rm_value
    return arr != rm_value


def _get_remove_mask_for_de(arr, mask, thresh, fill_value):
    keep_mask = _get_keep_mask_for_de(arr, mask, thresh, fill_value)
    return ~keep_mask


def mask_sp(
    arr, mask=None, thresh=None, rm_value=None, fill_value=None, return_sp=True
):

    sp_zero_fill_value = (np.min(arr.data) - 1) if arr.nnz > 0 else -1
    arr = cast_sp_to_de(arr, sp_zero_fill_value)

    mask = _get_keep_mask_for_de(arr, mask, thresh, rm_value)
    mask &= _get_keep_mask_for_de(arr, None, None, sp_zero_fill_value)

    if return_sp:
        return csr_matrix(mask).multiply(arr).astype(arr.dtype).tocsr()
    return mask_de(arr, ~mask, fill_value=fill_value)


def mask_de(
    arr, mask=None, thresh=None, rm_value=None, fill_value=None, return_sp=False
):

    if return_sp:
        mask = _get_keep_mask_for_de(arr, mask, thresh, rm_value)
        return csr_matrix(mask).multiply(arr).astype(arr.dtype).tocsr()

    arr = arr.copy()
    mask = _get_remove_mask_for_de(arr, mask, thresh, rm_value)
    arr[mask] = get_fill_value(arr) if fill_value is None else fill_value
    return arr


def cast_to_de(arr, fill_value=None):
    if not isinstance(arr, np.ndarray):
        return cast_sp_to_de(arr, fill_value)
    return arr


def assert_de_in_place(original, inp, results, expected, in_place):

    in_place = in_place or (expected.base is original or expected is original)
    expected_same = original.shape == expected.shape and np.all(original == expected)

    if in_place:
        assert results is inp or results.base is inp
    elif not expected_same:
        assert results is not inp and results.base is not inp
    else:
        assert np.all(original == inp)


def assert_arr_changed(original, inp, should_change, fill_value=None):
    if should_change:
        original = cast_to_de(original, fill_value)
        inp = cast_to_de(inp, fill_value)
        assert original.shape != inp.shape or not np.all(original == inp)
    else:
        assert_arr_equal(original, inp)


def assert_arr_in_place(original, inp, result, expected, in_place):

    if isinstance(original, csr_matrix):
        for attr in ("data", "indices", "indptr"):
            assert_de_in_place(
                getattr(original, attr),
                getattr(inp, attr),
                getattr(result, attr),
                getattr(expected, attr),
                in_place,
            )
    elif isinstance(original, np.ndarray) and isinstance(expected, csr_matrix):
        assert_de_in_place(original, inp, result.data, expected.data, in_place)
    else:
        assert_de_in_place(original, inp, result, expected, in_place)


def assert_arr_equal(
    result, expected, close=False, atol=None, rtol=None, fill_value=None
):

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape

    if isinstance(expected, csr_matrix):
        assert isinstance(result, csr_matrix)
        result = cast_sp_to_de(result, fill_value)
        expected = cast_sp_to_de(expected, fill_value)

    if close:
        np.testing.assert_array_almost_equal(result, expected)
    elif atol is not None or rtol is not None:
        rtol = 1e-5 if rtol is None else rtol
        atol = 1e-8 if atol is None else atol
        assert np.allclose(result, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(result, expected)


def assert_eig_equal(
    res_eigvals,
    res_eigvecs,
    exp_eigvals,
    exp_eigvecs,
    check_spectrum="none",
    is_symmetric=False,
):

    if check_spectrum == "largest":
        assert np.all(np.diff(res_eigvals) <= 0.0)
    elif check_spectrum == "smallest":
        assert np.all(np.diff(res_eigvals) >= 0.0)

    if is_symmetric:
        assert np.all(res_eigvals >= 0.0)

    assert_arr_equal(res_eigvals, exp_eigvals, close=True)

    if exp_eigvecs.ndim == 2:
        exp_eigvecs_corr = np.abs(exp_eigvecs.T @ res_eigvecs)
        self_eigvecs_corr = np.abs(res_eigvecs.T @ res_eigvecs)
        ids = np.identity(exp_eigvals.shape[0])
    else:
        exp_eigvecs_corr = np.abs(exp_eigvecs.transpose(0, 2, 1) @ res_eigvecs)
        self_eigvecs_corr = np.abs(res_eigvecs.transpose(0, 2, 1) @ res_eigvecs)
        ids = np.expand_dims(np.identity(exp_eigvals.shape[1]), axis=0)
        ids = np.repeat(ids, repeats=exp_eigvals.shape[0], axis=0)

    atol = None if is_symmetric else 0.1
    rtol = None if is_symmetric else 0.1
    close = True if is_symmetric else False

    assert_arr_equal(exp_eigvecs_corr, ids, close, rtol, atol)
    assert_arr_equal(self_eigvecs_corr, ids, close, rtol, atol)
