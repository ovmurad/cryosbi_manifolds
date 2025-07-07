import numpy as np

from ..arr.arr import BoolArr, BoolDeArr


def graph_dilation(neigh: BoolArr, vec: BoolDeArr, n_dilations: int = 1) -> BoolDeArr:
    """
    Performs the graph theoretical dilation operation. That is, given boolean vector
    'vec', we extend the value True of a node to its neighbors. The output will be
    the updated boolean vector after the dilation.

    :param neigh: The n x n bool neighbor relationship matrix, either dense or sparse
    :param vec: The size n dense boolean vector we dilate.
    :param n_dilations: The number of dilations to perform

    :returns: The result of the dilation operation as a dense boolean vector.
    """

    if (dtype := neigh.dtype) != np.bool_:
        raise ValueError(f"'neigh' matrix must be boolean, found {dtype} instead!")
    if (dtype := vec.dtype) != np.bool_:
        raise ValueError(f"'vec' must be boolean, found {dtype} instead!")

    for n in range(n_dilations):
        # Perform the dilation operation
        vec = vec @ neigh

    return vec
