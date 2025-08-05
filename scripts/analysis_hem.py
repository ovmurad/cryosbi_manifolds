from itertools import product

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from geometry_analysis.utils import create_grid_1d
from geometry_analysis.io import Database
from geometry_analysis.sampling import RANDOM_STATE, sample_array
from geometry_analysis.arr import reduce_arr_to_degrees, reduce_arr_with_func
from geometry_analysis.linalg import local_weighted_pca_iter
from geometry_analysis.geometry import (
    dist,
    count_x,
    affinity,
    laplacian,
    radii_distortions,
    spectral_embedding,
    local_grad_estimation,
    tslasso,
    doubling_dimension,
    correlation_dimension,
    levina_bickel,
    score_from_mult_knn_dist,
    score_from_mult_n_count,
    density_from_mult_knn_dist,
    density_from_mult_n_count,
    ies,
)
from geometry_analysis.utils.script_utils import (
    compute_split_masks,
    compute_distance_statistics,
    detect_outliers,
    compute_uniform_sample,
    subsample_dataset,
    plot_hist_with_percentile_lines,
    plot_mult_hist_with_percentile_lines,
    run_dimensionality_estimation,
    plot_n_count_or_knn_dist,
    perform_classification,
)


# ------------------ CREATE DATASET OBJECTS ------------------

SIM_PARAMS = ("conf", "snr", "rot", "shift", "sigma", "defoc")
NET_PARAMS = ("postm", "postw")
ALL_PARAMS = SIM_PARAMS + NET_PARAMS

SIM_SUBSAMPLE_KEY_PAIRS = (("points", "sim"),) + tuple(
    ("params", f"sim_{p}") for p in ALL_PARAMS
)
EXP_SUBSAMPLE_KEY_PAIRS = (("points", "exp"),) + tuple(
    ("params", f"exp_{p}") for p in NET_PARAMS
)

DATASET_NAME = "hem_latent_vecs_256"
SIM_NAME = "sim"
EXP_NAME = "exp"
DATA_NAMES = (EXP_NAME, SIM_NAME)
DATA_SUBSAMPLE_KEY_PAIRS = {
    EXP_NAME: EXP_SUBSAMPLE_KEY_PAIRS,
    SIM_NAME: SIM_SUBSAMPLE_KEY_PAIRS,
}

TRAIN_NAME = "train"
TEST_NAME = "test"
VALID_NAME = "valid"

CLEAN_NAME = "clean"
UNIFORM_NAME = "unif"

all_dataset = Database(database_name=f"{DATASET_NAME}_all")
final_dataset = Database(database_name=f"{DATASET_NAME}_final")

# ------------------ ANALYSIS PARAMETERS ------------------

# Check LOCAL STATISTICS
# the ks for the k-nn distances
MIN_K, MAX_K = 10, 120
KS = create_grid_1d(start=MIN_K, stop=MAX_K, step_size=10, scale="int")

# the rs of balls for counting neighbors
# it is important to not set MAX_R too large as that will bias the dimension estimate down.
# check some of the plots below and make sure that the for large rs, the histogram of the number of neighbors does
# not shift overwhelmingly to the right and stays relatively uniform.
MIN_R, MAX_R = 8.0, 22.0
RS = np.concatenate(
    [
        create_grid_1d(start=MIN_R, stop=MAX_R // 2, step_size=0.5, scale="linear"),
        create_grid_1d(start=MAX_R // 2 + 1, stop=MAX_R, step_size=1.0, scale="linear"),
    ]
)

# Number of points
EMB_NPTS = 2000
SPLIT_NPTS = {TRAIN_NAME: 10000, TEST_NAME: 5000, VALID_NAME: 5000}

# Check OUTLIER REMOVAL
# percentile cutoff for outlier removal(higher means more data is cutoff)
ALPHA = {EXP_NAME: 0.30, SIM_NAME: 0.30}
OUTLIER_ALGO = "knn_dists"  # algorithm for MV classification

# Check MANIFOLD LEARNING
# cutoff radius used for embedding.
RADIUS = {EXP_NAME: 14.0, SIM_NAME: 13.0}
RADIUS_EPS_RATIO = 3.0  # Good default value
EPS = {k: rad / RADIUS_EPS_RATIO for k, rad in RADIUS.items()}

# CHECK INTRINSIC DIMENSIONALITY ESTIMATION
DS = {EXP_NAME: [4, 5, 6], SIM_NAME: [2, 3, 4]}  # approximate intrinsic dimensions

# # ------------------ DATA SPLIT ------------------
for data_name in DATA_NAMES:
    compute_split_masks(all_dataset, data_name, SPLIT_NPTS)

# ------------------ OUTLIER REMOVAL ------------------
# Plot histograms of the number of neighbors for various radii and the knn distance for various k.
# Use these to find a cutoff percentage in [0, 1] for outlier detection.
# Intuitively, the outliers will be either points with few neighbors(in the bottom alpha/beta percentile)
# or points with large knn distance(in the top alpha/beta percentile).
# For the number of neighbors, the histogram will have large bins close to 0. We need to remove those.
# For the knn distance of neighbors, the histogram will have a long tail of points with large k-nn distances.
# Pick percentiles alpha(sim)/beta(exp) that reliably eliminate the anomalous bins.
# Compute local statistics(closest distance to nearest k neighbors or neighbor counts for different radii)
if OUTLIER_ALGO == "knn_dists":
    local_stats_dict = dict(ks=KS)
elif OUTLIER_ALGO == "n_counts":
    local_stats_dict = dict(rs=RS)
else:
    raise ValueError

for data_name in (EXP_NAME, SIM_NAME):

    compute_distance_statistics(all_dataset, data_name, TRAIN_NAME, **local_stats_dict)

    train_key = f"{data_name}_{TRAIN_NAME}-{data_name}_{TRAIN_NAME}"
    clean_key = f"{data_name}_{CLEAN_NAME}"

    # plot_n_count_or_knn_dist(knn_dist=all_dataset["knn_dists"][train_key])
    clean_mask = detect_outliers(
        all_dataset, data_name, TRAIN_NAME, ALPHA[data_name], OUTLIER_ALGO
    )

    if data_name == SIM_NAME:

        # Furthermore, we remove simulated points that are not in the real
        pair_data_name = (SIM_NAME, EXP_NAME)
        pair_mask_name = (TRAIN_NAME, TRAIN_NAME)

        compute_distance_statistics(
            all_dataset, pair_data_name, pair_mask_name, **local_stats_dict
        )

        near_mask = detect_outliers(
            all_dataset,
            pair_data_name[::-1],
            pair_mask_name[::-1],
            ALPHA[EXP_NAME],
            OUTLIER_ALGO,
        )
        clean_mask &= near_mask

    all_dataset["masks"][clean_key] = clean_mask


# ------------------ UNIFORM RESAMPLING ------------------
# Resample the data to obtain uniform samples.
for data_name, key_pairs in DATA_SUBSAMPLE_KEY_PAIRS.items():
    compute_distance_statistics(all_dataset, data_name, CLEAN_NAME, rs=RS)
    compute_uniform_sample(all_dataset, data_name, CLEAN_NAME, EMB_NPTS)
    subsample_dataset(all_dataset, final_dataset, UNIFORM_NAME, key_pairs)

# ------------------ INTRINSIC DIMENSIONALITY ESTIMATION ------------------
# Compute distances to closest KS neighbors and for a given set of radii RS compute the number of neighbors.
for data_name in DATA_NAMES:

    compute_distance_statistics(final_dataset, data_name, "all", RS, KS)

    print(f"Running intrinsic dimensionality estimation for hemagglutinin {data_name} data.")
    pair_name = f"{data_name}-{data_name}"
    run_dimensionality_estimation(
        n_count=final_dataset["n_counts"][pair_name],
        knn_dist=final_dataset["knn_dists"][pair_name],
        algos=("dd", "lb", "cd"),
    )

# ------------------ MANIFOLD LEARNING ------------------
# Select the bandwidth for the affinity matrix which will then be used for diffusion maps.
# We don't need the distortions really, just look at the printout, the distortion will go down, then it will start
# going up again so just pick the minimum when it occurs. There is generally no harm in picking a slightly larger radius
# than the one that achieves the minimum distortion. In fact, I generally find that to be more stable and yield better
# embeddings. Particularly for this type of data that seems to have varying topology and probably varying intrinsic
# dimensionality(so not really a manifold), I find it useful to go above the minimum computed here which seems to give
# a lower bound under which DM fails.

for data_name in DATA_NAMES:

    print(
        f"Running bandwidth distortion estimation for hemagglutinin {data_name} data."
    )
    final_dataset["eval_radii"][data_name] = radii_distortions(
        x_pts=final_dataset["points"][data_name],
        ds=DS[data_name],
        radii=RS,
        sample=64,
        rad_eps_ratio=RADIUS_EPS_RATIO,
        bsize=None,
    )

    rad = RADIUS[data_name]
    eps = EPS[data_name]
    #
    # Compute the geometry matrices: distance, gaussian affinity and geometric laplacian using the
    # radius and bandwidth suggested by the previous analysis
    pair_key = f"{data_name}-{data_name}"
    final_dataset["dists"][pair_key] = dist(
        x_pts=final_dataset["points"][data_name],
        threshold=rad,
    )
    final_dataset["affs"][pair_key] = affinity(
        dists=final_dataset["dists"][pair_key], eps=eps
    )
    final_dataset["laps"][pair_key] = laplacian(
        affs=final_dataset["affs"][pair_key], eps=eps
    )

    # Embed the data
    # Important!!!: The warning that the affinity matrix is not connected is a good indication that the
    # radius needs to be larger and that the embedding will probably fail miserably.
    mask_key = f"{data_name}_wc"
    affs = final_dataset["affs"][pair_key]
    degrees = reduce_arr_to_degrees(affs, axis=1)
    final_dataset["masks"][mask_key] = degrees >= np.percentile(degrees, 5)

    eigvals, embedding = spectral_embedding(
        affs=final_dataset[f"affs|{pair_key}|wc-wc"],
        ncomp=20,
        eps=eps,
        eigen_solver="amg",
    )
    final_dataset["lap_eigvals"][mask_key] = eigvals
    final_dataset["lap_eigvecs"][mask_key] = embedding
