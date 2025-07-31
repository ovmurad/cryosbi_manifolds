import numpy as np
from scipy.spatial.transform import Rotation
from ..arr.arr import RealDeArr


def quaternion_to_viewing_angle(quaternions: RealDeArr) -> RealDeArr:
    """
    Converts a list of quaternions to the viewing angle with respect to the z-axis.

    :param quaternions: A dense array of shape (N, 4), where each row is a
        quaternion [w, x, y, z].

    :returns: An array of shape (N, ) where each entry represents the viewing
        angle.
    """

    # Define the z-axis as a reference direction
    z_axis = np.array([0.0, 0.0, 1.0])

    # Convert quaternion to rotation matrix
    rotations = Rotation.from_quat(np.roll(quaternions, shift=-1, axis=1))
    rotation_mats = rotations.as_matrix()

    # The rotated z-axis is the 3rd column of the rotation matrix
    viewing_direction = rotation_mats[:, :, 2]

    cos_theta = viewing_direction @ z_axis
    # Clip to avoid numerical issues and return
    return np.arccos(np.clip(cos_theta, a_min=-1.0, a_max=1.0))
