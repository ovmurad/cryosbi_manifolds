from itertools import combinations
from typing import Any

import numpy as np
from scipy.io import loadmat

from geometry_analysis.io import Database
from geometry_analysis.sampling import RANDOM_STATE

MDS_D = {"ethanol": 27, "toluene": 45, "malonaldehyde": 27}
MDS_d = {"ethanol": 2, "toluene": 1, "malonaldehyde": 2}

RAW_DATA_FOLDER_NAME = "mds_raw_data"


def pos_to_feature(data: np.ndarray) -> np.ndarray:

    npts = data.shape[0]
    natoms = data.shape[1]

    idx = np.array(list(combinations(range(natoms), 3)), dtype=int)
    idx = np.stack([np.roll(idx, axis=1, shift=i) for i in range(3)], axis=1)
    idx = np.reshape(idx, -1)

    data = data[:, idx]
    data = np.reshape(data, (npts, -1, 3, 3))

    v1 = data[:, :, 1] - data[:, :, 0]
    v2 = data[:, :, 2] - data[:, :, 0]

    return np.einsum("fgx,fgx->fg", v1, v2) / (np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2))


def pos_to_torsion(data: np.ndarray) -> np.ndarray:

    natoms = data.shape[1]
    atoms4 = np.asarray(list(combinations(range(natoms), 4)))

    output = []

    for i, d in enumerate(data):
        torsions = []
        for ats in atoms4:
            d1 = d[ats[0]]
            c1 = d[ats[1]]
            c2 = d[ats[2]]
            d2 = d[ats[3]]
            cc = c2 - c1
            ip = ((d1 - c1) * (c2 - c1)) / (np.sum((c2 - c1) ** 2))
            tilded1 = [d1[0] - ip * cc[0], d1[1] - ip * cc[1], d1[2] - ip * cc[2]]
            iq = ((d2 - c2) * (c1 - c2)) / (np.sum((c1 - c2) ** 2))
            cc2 = c1 - c2
            tilded2 = [d2[0] - iq * cc2[0], d2[1] - iq * cc2[1], d2[2] - iq * cc2[2]]
            tilded2star = [tilded2[0] + cc2[0], tilded2[1] + cc2[1], tilded2[2] + cc2[2]]
            ab = np.sqrt(
                (tilded2star[0] - c1[0]) ** 2
                + (tilded2star[1] - c1[1]) ** 2
                + (tilded2star[2] - c1[2]) ** 2
            )
            bc = np.sqrt(
                (tilded2star[0] - tilded1[0]) ** 2
                + (tilded2star[1] - tilded1[1]) ** 2
                + (tilded2star[2] - tilded1[2]) ** 2
            )
            ca = np.sqrt(
                (tilded1[0] - c1[0]) ** 2
                + (tilded1[1] - c1[1]) ** 2
                + (tilded1[2] - c1[2]) ** 2
            )
            torsions.append(np.arccos((ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca)))
        output.append(np.array(torsions))
    return np.array(output)


def mds_mat_to_database(molecule_name: str, **kwargs: Any) -> None:

    database = Database(database_name=f"{molecule_name}_data", mode="overwrite")
    raw_data_path = database.path.parent / RAW_DATA_FOLDER_NAME / f"{molecule_name}.mat"

    data = loadmat(str(raw_data_path))["R"]
    sample_idx = RANDOM_STATE.choice(data.shape[0], kwargs["npts"], replace=False)
    data = data[sample_idx]

    database["points"]["x"] = pos_to_feature(data)
    database["points"]["y"] = pos_to_torsion(data)


if __name__ == "__main__":
    molecules = ("ethanol", "malonaldehyde", "toluene")
    kwargs = {"npts": 10000}
    for mol in molecules:
        mds_mat_to_database(mol, **kwargs)