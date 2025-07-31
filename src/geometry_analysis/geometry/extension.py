from ..arr.arr import RealArr, RealDeArr
from ..arr.reduce import reduce_arr_to_degrees


def nadaraya_watson_kernel_extension(
    x_pts: RealDeArr,
    affs: RealArr,
) -> RealDeArr:
    norm = reduce_arr_to_degrees(affs, axis=1, keepdims=x_pts.ndim == 2)
    return (affs @ x_pts) / norm
