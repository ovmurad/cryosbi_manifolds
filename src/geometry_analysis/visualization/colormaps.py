from typing import Optional

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize

from ..arr.arr import RealDeArr


def normalize_color(
    color: RealDeArr, vmin: Optional[int | float], vmax: Optional[int | float]
) -> RealDeArr:

    def _cast_color_lim(lim: int | float | None) -> float | None:
        return np.percentile(color, q=lim) if isinstance(lim, int) else lim

    vmin, vmax = _cast_color_lim(vmin), _cast_color_lim(vmax)
    color_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    return color_norm(color) * (vmax - vmin) + vmin


def truncate_colormap(
    cmap: Colormap, minval: Optional[float] = 0.0, maxval: Optional[float] = 1.0
) -> LinearSegmentedColormap:
    """Truncate a colormap to limit its range.

    :param cmap: Original colormap.
    :param minval: Minimum value to start the colormap.
    :param maxval: Maximum value to end the colormap.

    :return: Truncated colormap.
    """

    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name}, {minval:.2f}, {maxval:.2f})",
        cmap(np.linspace(minval, maxval, cmap.N)),
    )
