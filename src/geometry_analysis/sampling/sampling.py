from typing import Literal, Optional, Tuple, TypeAlias

import numpy as np
from sklearn.model_selection import train_test_split

from .constants import RANDOM_STATE
from ..arr.arr import Arr, DeArr

SampleMode: TypeAlias = Literal["first", "every", "random"]


def sample_array(
    data: int | Arr,
    num_or_pct: Optional[int | float] = None,
    mode: SampleMode = "random",
) -> Arr:
    """
    Samples a numpy array along the 0th axis based on the specified `mode`.

    :param data: np.ndarray representing the input array to be sampled. Alternatively,
        if an int is given, it will be interpreted as an array of integers from 0 to
        that int.
    :param num_or_pct:
        If an int, specifies the number of samples.
        If a float, specifies the percentage of the total size to sample.
        If None, uses the entire array (default is None).
    :param mode : The sampling mode. Can be one of the following:
        - "first": Selects the first `num_or_pct` samples.
        - "every": Selects every `num_or_pct`-th sample.
        - "random": Selects `num_or_pct` samples randomly without replacement.
        Default is "first".

    :return: The sampled array.
    """

    if isinstance(data, int):
        data = np.arange(0, data, dtype=np.int_)

    if isinstance(num_or_pct, float):
        num = round(num_or_pct * data.shape[0])
    elif num_or_pct is None or isinstance(num_or_pct, int):
        num = num_or_pct
    else:
        raise ValueError("'num_or_pct' must be an int, float, or None!")

    match mode:
        case "first":
            data_sample = data[:num]
        case "every":
            data_sample = data[::num]
        case "random":
            random_idx = RANDOM_STATE.choice(data.shape[0], num, replace=False)
            data_sample = data[random_idx]
        case _:
            raise ValueError(f"Unknown mode: {mode}!")

    return data_sample


def train_val_test_split(
    data: DeArr, test_p: float, val_p: Optional[float] = None,
) -> Tuple[DeArr, DeArr, DeArr] | Tuple[DeArr, DeArr]:

    """
    Splits the input data into training, validation, and test sets.

    :param data: The dataset to be split.
    :param test_p: The proportion of the dataset to include in the test set.
    :param val_p: The proportion of the dataset to include in the validation set. If
        None, no validation set is sampled.

    :return: A tuple containing either (training data, validation data, and test data)
        or (training data, test data) in the case where 'val_p' = None.
    """

    # Split the data into (train + validation sets) and test set
    train_val_data, test_data = train_test_split(
        data, test_size=test_p, random_state=RANDOM_STATE
    )

    if val_p is None:
        print(
            f"Splitting {data.shape[0]} data points into: {train_val_data.shape[0]} "
            f"training points, and {test_data.shape[0]} test points."
        )
        return np.sort(train_val_data), np.sort(test_data)

    # Split the train + validation sets into separate training and validation sets
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_p / (1 - test_p), random_state=RANDOM_STATE
    )

    print(
        f"Splitting {data.shape[0]} data points into: {train_data.shape[0]} "
        f"training points, {val_data.shape[0]} validation points, and "
        f"{test_data.shape[0]} test points."
    )

    return np.sort(train_data), np.sort(val_data), np.sort(test_data)
