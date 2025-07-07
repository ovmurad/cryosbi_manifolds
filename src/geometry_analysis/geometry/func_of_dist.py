from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import attr
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import vstack as sp_vstack
from sklearn.metrics import pairwise_distances_chunked

from ..arr.arr import (
    BoolArr,
    BoolDeArr,
    BoolSpArr,
    IntDeArr,
    RealArr,
    RealDeArr,
    RealSpArr,
)
from ..arr.cast import cast_sp_to_sp_bool
from ..arr.reduce import reduce_arr_to_nnz
from ..arr.threshold import threshold_arr

_FuncOut = TypeVar("_FuncOut", BoolDeArr, BoolSpArr, IntDeArr, RealDeArr, RealSpArr)


# Below are a classes implementing batched functions that can be applied to a batch of
# distances, store the per batch results, and aggregate the batch results at the end.
# Since the distance computation can return a sparse or dense array, we need to
# implement both sparse and dense batch and aggregate functions.
@attr.define(slots=True)
class _FuncOfDist(Generic[_FuncOut]):
    """
    Container object for the batched functions. The dispatcher initializes the
    object with the appropriate func and agg_func.
    """

    func: Callable[[RealArr], _FuncOut] = attr.field()
    agg_func: Callable[[List[_FuncOut]], _FuncOut] = attr.field()
    memory: List[_FuncOut] = attr.field(factory=list)

    def add_batch(self, dists: RealArr) -> None:
        self.memory.append(self.func(dists))

    def agg_batches(self) -> _FuncOut:
        if len(self.memory) == 1:
            return self.memory[0]
        return self.agg_func(self.memory)


_copy_func = lambda d: d.copy()
_count_x_func = partial(reduce_arr_to_nnz, axis=1)
_count_y_func = partial(reduce_arr_to_nnz, axis=0)


def _func_of_dist_factory(func_name: str, return_sp: bool) -> _FuncOfDist:
    match (func_name, return_sp):
        case ("dist", False):
            return _FuncOfDist[RealDeArr](_copy_func, np.vstack)
        case ("dist", True):
            return _FuncOfDist[RealSpArr](_copy_func, sp_vstack)
        case ("neigh", False):
            return _FuncOfDist[BoolDeArr](lambda d: d != np.inf, np.vstack)
        case ("neigh", True):
            return _FuncOfDist[BoolSpArr](cast_sp_to_sp_bool, sp_vstack)
        case ("count_x", False):
            return _FuncOfDist[IntDeArr](_count_x_func, np.concatenate)
        case ("count_x", True):
            return _FuncOfDist[IntDeArr](_count_x_func, np.concatenate)
        case ("count_y", False):
            return _FuncOfDist[IntDeArr](_count_y_func, np.add.reduce)
        case ("count_y", True):
            return _FuncOfDist[IntDeArr](_count_y_func, np.add.reduce)
        case _:
            raise NotImplementedError(f"Unknown func of distance {func_name}!")


def func_of_dist(
    x_pts: Optional[RealDeArr] = None,
    y_pts: Optional[RealDeArr] = None,
    dists: Optional[RealArr | Iterator[RealArr]] = None,
    funcs: str | Sequence[str] = "dist",
    threshold: Optional[float | Sequence[float]] = None,
    return_sp: bool = True,
    **kwargs: Any,
) -> Dict[float, Dict[str, _FuncOut]]:
    """
    This function computes the self distances between points in a dense array of shape
    num_points x num_features, or between two arrays with the same num_features,
    or uses a precomputed distance matrix or batches of it which can be dense or sparse.
    This function splits the distance computation (when we don't have a precomputed
    distance matrix) into batches, using kwargs["working_memory"] = batch size in MB,
    optionally thresholds the distances setting elements with values greater than the
    specified radii to implicit 0 if the output is sparse or to np.inf if the output is
    dense, and applies functions to the thresholded distances computed for each batch,
    storing only their results, in order to reduce memory usage. Finally, an aggregation
    function combines the outputs of these results into one final result. If more than
    one threshold is given, for each batch we start by thresholding the distances using
    the loosest threshold and continue to censor the distance matrix in place for that
    batch for the remaining radii, in decreasing order. The output will be a dict of
    dicts with the results for all threshold and functions.

    :param x_pts: Array of shape (num_points_x, num_features) representing the first set
        of points. Optional if 'dists' is given.
    :param y_pts: Array of shape (num_points_y, num_features) representing the second
        set of points. Optional.
    :param dists: Precomputed distance matrix, either dense or sparse. Optional if
        'x_pts' is given.
    :param funcs: Function(s) to apply to the distances. Current options are:
        - "dist": distance between points.
        - "neigh": boolean mask indicating whether two points are r-neighbors.
        - "count_x": count the number of neighbors 'x_pts' has either w.r.t to
        itself if 'y_pts' = None or to 'y_pts' otherwise.
        - "count_y": same, but for 'y_pts'.
    :param threshold: Radius or radii for thresholding distances. Default is np.inf
        meaning no thresholding is being applied.
    :param return_sp: Whether the distance matrix is sparse.

    :return: a dict with keys being the thresholds in increasing order and values being
        another dict with function names as keys and the function's results for that
        threshold as values.
    """

    # Initialize arguments and set defaults.
    funcs = (funcs,) if isinstance(funcs, str) else funcs

    if threshold is None:
        return_sp = False
    if isinstance(threshold, (int, np.integer, float, np.floating)) or threshold is None:
        threshold = (threshold, )
    threshold = sorted(threshold)

    # Determine whether distance matrix is precomputed or needs to be computed
    # Else use the sklearn.pairwise_distance_chunked to chunk the data into batches.
    if dists is not None:
        if isinstance(dists, (np.ndarray, csr_matrix)):
            if isinstance(dists, csr_matrix):
                return_sp = True
            dists = (dists,)
    elif x_pts is not None:
        dists = pairwise_distances_chunked(x_pts, y_pts, **kwargs)
    else:
        raise ValueError("If 'dists' is not given, 'x_pts' has to be set")

    thresh_to_funcs: Dict[float, Dict[str, _FuncOfDist[_FuncOut]]] = {
        thresh: {name: _func_of_dist_factory(name, return_sp) for name in funcs}
        for thresh in threshold
    }

    # Process each distance batch
    for b_dists in dists:

        # Start from the largest threshold and keep thresholding.
        for thresh in threshold[::-1]:
            # Nothing to do when not theshold, but otherwise threshold in place.
            if thresh is not None:
                b_dists = threshold_arr(b_dists, thresh, return_sp, in_place=True)
            # Apply each function on the batch distances and save the results.
            for func in thresh_to_funcs[thresh].values():
                func.add_batch(b_dists)

    # aggregate batches and return
    return {
        thresh: {name: func.agg_batches() for name, func in name_to_func.items()}
        for thresh, name_to_func in thresh_to_funcs.items()
    }


class FuncOfDist(Generic[_FuncOut]):

    def __init__(self, func: str) -> None:
        self.func = func

    def __call__(
        self,
        x_pts: Optional[RealDeArr] = None,
        y_pts: Optional[RealDeArr] = None,
        dists: Optional[RealArr | Iterator[RealArr]] = None,
        threshold: Optional[float | Sequence[float]] = None,
        return_sp: bool = True,
        **kwargs: Any,
    ) -> _FuncOut | Dict[float, _FuncOut]:

        thresh_to_funcs = func_of_dist(
            x_pts, y_pts, dists, self.func, threshold, return_sp, **kwargs
        )

        if isinstance(threshold, float) or threshold is None:
            return thresh_to_funcs.popitem()[1][self.func]

        return {
            thresh: name_to_func[self.func]
            for thresh, name_to_func in thresh_to_funcs.items()
        }


class FuncsOfDist(Generic[_FuncOut]):
    def __init__(self, funcs: Tuple[str, ...]) -> None:
        self.funcs = funcs

    def __call__(
        self,
        x_pts: Optional[RealDeArr] = None,
        y_pts: Optional[RealDeArr] = None,
        dists: Optional[RealArr | Iterator[RealArr]] = None,
        threshold: Optional[float | Sequence[float]] = None,
        return_sp: bool = True,
        **kwargs: Any,
    ) -> Tuple[_FuncOut, ...] | Dict[str, Dict[float, _FuncOut]]:

        thresh_to_funcs = func_of_dist(
            x_pts, y_pts, dists, self.funcs, threshold, return_sp, **kwargs
        )

        if isinstance(threshold, float) or threshold is None:
            return tuple(thresh_to_funcs.popitem()[1].values())

        return {
            name: {
                thresh: name_to_func[name]
                for thresh, name_to_func in thresh_to_funcs.items()
            }
            for name in self.funcs
        }


dist = FuncOfDist[RealArr]("dist")
neigh = FuncOfDist[BoolArr]("neigh")
count_x = FuncOfDist[IntDeArr]("count_x")
count_y = FuncOfDist[IntDeArr]("count_y")
count_xy = FuncsOfDist[IntDeArr](("count_x", "count_y"))


# def knn_dist(dists: RealArr, ks: Sequence[int]) -> Dict[int, RealDeArr]:
#
#     if issparse(dists):
#
#         data = dists.data[dists.data > 0.0]
#         diffs = np.diff(dists.indptr) - 1
#         max_nnz = np.max(diffs)
#
#         cols = np.expand_dims(np.arange(0, max_nnz), axis=0)
#         diffs = np.expand_dims(diffs, axis=1)
#
#         dists = np.full(shape=(dists.shape[0], max_nnz), fill_value=np.inf)
#         dists[cols < diffs] = data
#
#     else:
#
#         dists = np.reshape(dists[dists > 0], (dists.shape[0], dists.shape[1] - 1))
#
#     ks = np.sort(ks)
#     knn_dists = {}
#
#     for k in ks[::-1]:
#
#         dists = np.partition(dists, k, axis=1)[:, :k]
#         knn_dists[k] = dists
#
#     return knn_dists
#
#
