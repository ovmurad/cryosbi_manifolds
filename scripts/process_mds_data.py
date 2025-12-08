from itertools import combinations
from typing import Any

import numpy as np
from scipy.io import loadmat

from geometry_analysis.io import Database
from geometry_analysis.sampling import RANDOM_STATE

MDS_D = {"ethanol": 27, "toluene": 45, "malonaldehyde": 27}
MDS_d = {"ethanol": 2, "toluene": 1, "malonaldehyde": 2}
MDS_quads = {"ethanol": [(0, 1, 3, 7), (0, 2, 3, 8)], "toluene": [(0, 1, 6, 7)], "malonaldehyde": [(0, 1, 3, 7), (1, 2, 4, 7)]}

RAW_DATA_FOLDER_NAME = "mds_raw_data"


def pos_to_angles(data: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    data : (B, N, 3)
        B conformations, N atoms, 3D coordinates.

    Returns
    -------
    angles : (B, T)
        Planar angles (in radians) at the middle atom of each triple (i < j < k),
        where T = C(N, 3).
    """
    B, natoms, _ = data.shape

    atoms3 = np.array(list(combinations(range(natoms), 3)), dtype=int)
    a0, a1, a2 = atoms3[:, 0], atoms3[:, 1], atoms3[:, 2]
    d0, d1, d2 = data[:, a0], data[:, a1], data[:, a2]

    # Vectors from middle atom (a1) to the other two
    v0 = d0 - d1
    v2 = d2 - d1

    # Dot products and norms
    dot = np.einsum("btk,btk->bt", v0, v2)
    eps = 1e-12

    # Cos of angle
    cos_theta = dot / (np.linalg.norm(v0, axis=2) * np.linalg.norm(v2, axis=2) + eps)

    # Numerical safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Actual angles in radians
    angles = np.arccos(cos_theta)

    return angles


def pos_to_torsion(data: np.ndarray, molecule_name: str, batch_size: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    data : (B, N, 3)
        B molecules, N atoms, 3 coords
    molecule_name: str
        Name of the molecule
    batch_size : int
        How many molecules to process at once

    Returns
    -------
    torsions : (B, T)
        T torsions per conformation, T = C(N,4)
    z: (B, MDS_d[molecule_name])
        Torsions that parametrize the molecule's configuration manifold
    """

    n_batch, natoms, _ = data.shape
    atoms4 = np.asarray(list(combinations(range(natoms), 4)), dtype=int)
    n_tors = atoms4.shape[0]

    # Prepare output
    torsions = np.empty((n_batch, n_tors), dtype=np.float64)

    # Pre-extract index arrays for the 4 atoms
    a0, a1, a2, a3 = atoms4[:, 0], atoms4[:, 1], atoms4[:, 2], atoms4[:, 3]

    # Loop over molecule batches
    for start in range(0, n_batch, batch_size):

        end = min(start + batch_size, n_batch)
        db = data[start:end]          # (b, N, 3)
        b = end - start               # Actual batch size

        # Extract four atom positions → shapes (b, T, 3)
        # Flatten batch + torsion dims → (b*T, 3)
        d1 = db[:, a0].reshape(-1, 3)
        c1 = db[:, a1].reshape(-1, 3)
        c2 = db[:, a2].reshape(-1, 3)
        d2 = db[:, a3].reshape(-1, 3)

        # Vector math Vlad doesn't understand
        cc = c2 - c1
        num_ip = np.einsum("ij,ij->i", d1 - c1, cc)
        denom_ip = np.einsum("ij,ij->i", cc, cc)
        ip = num_ip / denom_ip
        tilded1 = d1 - ip[:, None] * cc

        cc2 = c1 - c2
        num_iq = np.einsum("ij,ij->i", d2 - c2, cc2)
        denom_iq = np.einsum("ij,ij->i", cc2, cc2)
        iq = num_iq / denom_iq
        tilded2 = d2 - iq[:, None] * cc2
        tilded2star = tilded2 + cc2

        diff_ab = tilded2star - c1
        diff_bc = tilded2star - tilded1
        diff_ca = tilded1 - c1

        ab2 = np.einsum("ij,ij->i", diff_ab, diff_ab)
        bc2 = np.einsum("ij,ij->i", diff_bc, diff_bc)
        ca2 = np.einsum("ij,ij->i", diff_ca, diff_ca)

        ab = np.sqrt(ab2)
        ca = np.sqrt(ca2)

        cos_theta = (ab2 - bc2 + ca2) / (2 * ab * ca)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        b_torsions_flat = np.arccos(cos_theta)
        b_torsions = b_torsions_flat.reshape(b, n_tors)

        # Store results
        torsions[start:end] = b_torsions

    param_quads = MDS_quads[molecule_name]
    param_indices = []
    for quad in param_quads:
        quad_idx = np.flatnonzero(np.all(atoms4 == quad, axis=1)).item()
        param_indices.append(quad_idx)
    z = torsions[:, param_indices]

    return torsions, z


def mds_mat_to_database(molecule_name: str, npts: int) -> None:

    database = Database(database_name=f"{molecule_name}_data", mode="overwrite")
    raw_data_path = database.path.parent / RAW_DATA_FOLDER_NAME / f"{molecule_name}.mat"

    data = loadmat(str(raw_data_path))["R"]
    sample_idx = RANDOM_STATE.choice(data.shape[0], size=npts, replace=False)
    data = data[sample_idx]

    database["points"]["positions"] = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    database["points"]["angles"] = pos_to_angles(data)

    torsions, z = pos_to_torsion(data, molecule_name)
    database["points"]["torsions"] = torsions
    database["params"]["z"] = torsions


if __name__ == "__main__":

    molecules = ("ethanol", "malonaldehyde", "toluene")
    npts = 20000

    for mol in molecules:
        print(f"Creating {mol} data!")
        mds_mat_to_database(mol, npts)
