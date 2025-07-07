from typing import Any, Callable, Generic, Literal, Optional, Type, TypeAlias

import numpy as np

from ..arr.arr import NT, IntDeArr, NumDeArr, RealDeArr

GridScale: TypeAlias = Literal["int", "linear", "exp"]


class GridFunc1D(Generic[NT]):

    def __init__(self, func: Callable[..., NT], num_type: Type[NT] = None):
        self.func = func
        self.num_type = num_type

    def __call__(
        self,
        start: NT,
        stop: NT,
        step_size: Optional[NT] = None,
        num_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> NumDeArr[NT]:

        start, stop = self.num_type(start), self.num_type(stop)
        if start >= stop:
            raise ValueError("Must have 'start' < 'stop'!")

        if num_steps is not None and step_size is not None:
            raise ValueError("Only one of 'num_steps' and 'step_size' must be set!")
        if num_steps is not None and num_steps <= 0:
            raise ValueError("'num_steps' must be positive!")
        if step_size is not None and step_size <= 0:
            raise ValueError("'step_size' must be positive!")

        if num_steps is None and step_size is None:
            step_size = 1

        if num_steps is None:

            step_size = self.num_type(step_size)
            num_steps_float = (stop - start) / step_size
            num_steps = round(num_steps_float)

            if np.abs(num_steps - num_steps_float) > 1e-9:
                raise ValueError("'step_size' must divide 'stop' - 'start'!")

        else:
            num_steps = int(num_steps)
            step_size = self.num_type((stop - start) / num_steps)

        return self.func(start, stop, step_size, num_steps, **kwargs)


def _create_int_grid_1d(
    start: int, stop: int, step_size: int, num_steps: int
) -> IntDeArr:
    if (start - stop) % num_steps != 0:
        raise ValueError("'num_steps' must divide 'stop - 'start'!")
    step_size = round(step_size)
    return np.arange(start, stop + step_size, step_size)


def _create_linear_grid_1d(
    start: float, stop: float, _: float, num_steps: int
) -> RealDeArr:
    return np.linspace(start, stop, num_steps + 1)


def _create_exp_grid_1d(
    start: float, stop: float, _: float, num_steps: int, base: float = 10.0
) -> RealDeArr:
    return np.logspace(start, stop, num_steps + 1, base=base)


create_int_grid_1d = GridFunc1D[int](_create_int_grid_1d, int)
create_linear_grid_1d = GridFunc1D[float](_create_linear_grid_1d, float)
create_exp_grid_1d = GridFunc1D[float](_create_exp_grid_1d, float)


def create_grid_1d(
    start: NT,
    stop: NT,
    step_size: Optional[NT] = None,
    num_steps: Optional[int] = None,
    scale: GridScale = "linear",
    **kwargs,
) -> NumDeArr[NT]:
    """
    Create a grid of values between the given start and end points. 'num_steps' refers
    to the number of gaps on the grid, not to the num argument used by np.linspace.
    i.e. num_steps = num - 1.

    :param start: The starting value of the grid.
    :param stop: The ending value of the grid and the last point on the grid.
        Must be greater than start.
    :param step_size: The size of each step in the grid. Optional if `num_steps` is
        given.
    :param num_steps: The number of steps/gaps in the grid. Optional if step_size is
        given.
    :param scale: The type of scaling to use for the grid. Defaults to "linear".
        Options are "int", "linear", and "exp".

    :return: A numpy array of values representing the grid.
    """
    match scale:
        case "int":
            grid = create_int_grid_1d(start, stop, step_size, num_steps)
        case "linear":
            grid = create_linear_grid_1d(start, stop, step_size, num_steps)
        case "exp":
            grid = create_exp_grid_1d(start, stop, step_size, num_steps, **kwargs)
        case _:
            raise ValueError(f"Unsupported scale type: {scale}!")

    return grid
