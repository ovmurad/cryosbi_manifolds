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
ALL_PARAMS = ("z")

SUBSAMPLE_KEY_PAIRS = (("points", "sim"),) + tuple(
    ("params", f"{p}") for p in ALL_PARAMS
)
# This needs to be the name of the folder in data/ with all latent data from cryosbi
DATASET_NAME = "ethanol_data"
DATA_NAME = "torsions"

TRAIN_NAME = "train"
CLEAN_NAME = "clean"
UNIFORM_NAME = "unif"

all_dataset = Database(database_name=f"{DATASET_NAME}", mode='overwrite')
final_dataset = Database(database_name=f"{DATASET_NAME}_final", mode='overwrite')
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
EMB_NPTS = 20000
SPLIT_NPTS = {TRAIN_NAME: 60000}

# Check OUTLIER REMOVAL
ALPHA = 0.20  # percentile cutoff for outlier removal(higher means more data is cutoff)
OUTLIER_ALGO = "knn_dists"  # algorithm for MV classification

# Check MANIFOLD LEARNING.
RADIUS = 15.0  # cutoff radius used for embedding.
RADIUS_EPS_RATIO = 3.0  # Good default value
EPS = RADIUS / RADIUS_EPS_RATIO

# Check INTRINSIC DIMENSIONALITY ESTIMATION.
DS = [6, 7, 8]  # approximate intrinsic dimensions.

IES_SUBSAMPLE_SIZE = 500
TSLASSO_SUBSAMPLE_SIZE = 500

GRAD_ESTIM_SUBSAMPLE_SIZE = 5000
EIGENGAP_ESTIM_SUBSAMPLE_SIZE = 5000

# ------------------ DATA SPLIT ------------------
compute_split_masks(all_dataset, DATA_NAME, SPLIT_NPTS)

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

print(f"Running outlier removal for igg {DATA_NAME} data.")
compute_distance_statistics(all_dataset, DATA_NAME, TRAIN_NAME, **local_stats_dict)
# plot_n_count_or_knn_dist(knn_dist=all_dataset["knn_dists"][f"{DATA_NAME}_train-{DATA_NAME}_train"])
all_dataset["masks"][f"{DATA_NAME}_clean"] = detect_outliers(
    all_dataset, DATA_NAME, TRAIN_NAME, ALPHA, OUTLIER_ALGO
)

# ------------------ UNIFORM RESAMPLING ------------------
# Resample the data to obtain uniform samples.
print(f"Uniformly subsampling igg {DATA_NAME} data.")
compute_distance_statistics(all_dataset, DATA_NAME, CLEAN_NAME, rs=RS)
compute_uniform_sample(all_dataset, DATA_NAME, CLEAN_NAME, EMB_NPTS)

# Saves subsample of "all_dataset" into "final_dataset"
subsample_dataset(all_dataset, final_dataset, UNIFORM_NAME, SIM_SUBSAMPLE_KEY_PAIRS)

# ------------------ INTRINSIC DIMENSIONALITY ESTIMATION ------------------
# Compute distances to closest KS neighbors and for a given set of radii RS compute the number of neighbors.
print(f"Computing distance statistics for igg {DATA_NAME} data.")
compute_distance_statistics(final_dataset, DATA_NAME, "all", RS, KS)

# Run all dimensionality estimation algorithms. Look at the outputs and pick a reasonable span of dimensions.
# We'll use the average for kernel regression/density estimation and all values for estimating the Laplacian bandwidth.
# We cannot perform eigen-gap at this point because we haven't estimated the bandwidth.
# That algorithm will be computed later.
print(f"Running intrinsic dimensionality estimation for igg {DATA_NAME} data.")
run_dimensionality_estimation(
    n_count=final_dataset["n_counts"][f"{DATA_NAME}-{DATA_NAME}"],
    knn_dist=final_dataset["knn_dists"][f"{DATA_NAME}-{DATA_NAME}"],
    algos=("dd", "lb", "cd"),
)
dim_estimates_dict = doubling_dimension(final_dataset["n_counts"][f"{DATA_NAME}-{DATA_NAME}"])
dim_estimates = np.stack(list(dim_estimates_dict.values()))
dim_estimates[np.isnan(dim_estimates)] = np.mean(
    dim_estimates[~np.isnan(dim_estimates)], axis=0
)
final_dataset["estimated_d"][f"{DATASET_NAME}"] = np.mean(dim_estimates, axis=0)

#------------------ MANIFOLD LEARNING ------------------

# Select the bandwidth for the affinity matrix which will then be used for diffusion maps.
# We pick the radii slightly above the radii for which the distortion reaches its minimum.
# There is generally no harm in picking a slightly larger radius
# than the one that achieves the minimum distortion. 
# Particularly for this type of data that seems to have varying topology and probably varying intrinsic
# dimensionality, I find it useful to go above the minimum computed here which seems to give
# a lower bound under which DM fails.

print(f"Running bandwidth distortion estimation for igg {DATA_NAME} data.")
final_dataset["eval_radii"][DATA_NAME] = radii_distortions(
    x_pts=final_dataset["points"][DATA_NAME],
    ds=DS,
    radii=RS,
    sample=64,
    rad_eps_ratio=RADIUS_EPS_RATIO,
    bsize=None,
)

# Compute the geometry matrices: distance, gaussian affinity and geometric laplacian using the
# radius and bandwidth suggested by the previous analysis
print(f"Computing geometry matrices for igg {DATA_NAME} data.")
final_dataset["dists"][f"{DATA_NAME}-{DATA_NAME}"] = dist(
    x_pts=final_dataset["points"][DATA_NAME],
    threshold=RADIUS,
)
final_dataset["affs"][f"{DATA_NAME}-{DATA_NAME}"] = affinity(
    dists=final_dataset["dists"][f"{DATA_NAME}-{DATA_NAME}"], eps=EPS
)
final_dataset["laps"][f"{DATA_NAME}-{DATA_NAME}"] = laplacian(
    affs=final_dataset["affs"][f"{DATA_NAME}-{DATA_NAME}"], eps=EPS
)

# Embed the data
# Important!!!: The warning that the affinity matrix is not connected is a good indication that the
# radius needs to be larger and that the embedding will probably fail miserably.
affs = final_dataset["affs"][f"{DATA_NAME}-{DATA_NAME}"]
degrees = reduce_arr_to_degrees(affs, axis=1)
final_dataset["masks"][f"{DATA_NAME}_wc"] = degrees >= np.percentile(degrees, 5)

# Subsamples final dataset to only have well-connected ("wc") points
print(f"Subsampling igg {DATA_NAME} data to only well-connected points.")
subsample_dataset(final_dataset, final_dataset, "wc", SIM_SUBSAMPLE_KEY_PAIRS)

print(f"Running spectral embedding for igg {DATA_NAME} data.")
eigvals, embedding = spectral_embedding(
    affs=final_dataset[f"affs|{DATA_NAME}-{DATA_NAME}|wc-wc"],
    ncomp=20,
    eps=EPS,
    eigen_solver="amg",
)
final_dataset["lap_eigvals"][DATA_NAME] = eigvals
final_dataset["lap_eigvecs"][DATA_NAME] = embedding

# ------------------ FEATURE SELECTION ------------------
#
# ------------------ IES ------------------
# I'm taking only one d for IES because the algorithm is very slow. Definitely need to improve it.
# Look at the json file and display the top 3 axes selected by IES as they will be the embedding coordinates
# have the smallest frequency and are as independent as possible w.r.t. to the objective defined in the paper.

print(f"Running IES for igg {DATA_NAME} data.")
final_dataset["ies"][DATA_NAME] = ies(
    emb_pts=final_dataset["lap_eigvecs"][DATA_NAME],
    emb_eigvals=final_dataset[f"lap_eigvals"][DATA_NAME],
    lap=final_dataset[f"laps|{DATA_NAME}-{DATA_NAME}|wc-wc"],
    ds=3,
    s=DS[1],
    sample=500,
)

# ------------------ Gradient Estimation and TSLasso ------------------

print(f"Estimating gradients for igg {DATA_NAME} data.")

func_vals = [final_dataset[f"params|{DATA_NAME}_{sp}|wc"] for sp in SIM_PARAMS]
funcs = np.concatenate(
    [np.expand_dims(fv, axis=1) if fv.ndim == 1 else fv for fv in func_vals], axis=1
)

final_dataset["grads"][f"{DATA_NAME}_wc"] = local_grad_estimation(
    x_pts=final_dataset[f"points|{DATA_NAME}|wc"],
    f0_vals=funcs,
    f_vals=funcs,
    weights=final_dataset[f"affs|{DATA_NAME}-{DATA_NAME}|wc-wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE],
    bsize=50,
    ncomp=10,
)

x_pts = final_dataset[f"points|{DATA_NAME}|wc"]
grads = final_dataset["grads"][f"{DATA_NAME}_wc"]
affs = final_dataset[f"affs|{DATA_NAME}-{DATA_NAME}|wc-wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE]
snr = final_dataset[f"params|{DATA_NAME}_snr|wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE]

for percentile in create_grid_1d(start=0, stop=90, step_size=5, scale="int"):

    cutoff = np.percentile(snr, q=percentile)
    print(f"Running TSLasso for percentile={percentile} and cutoff={cutoff}.")

    sample = np.flatnonzero(snr >= cutoff)
    sample = np.sort(sample_array(sample, num_or_pct=TSLASSO_SUBSAMPLE_SIZE))

    tslasso_results = tslasso(
        x_pts=x_pts,
        affs=affs,
        grads=grads,
        ncomp=5,
        sample=sample,
        bsize=128,
        lr=100.0,
        l2_reg=0.0,
        max_nlamb=20,
        max_niter=500,
        tol=1e-14,
    )
    final_dataset["tslasso"][f"{DATA_NAME}_{percentile}"] = tslasso_results

# ------------------ LOCAL PCA for the Eigengap Dim Estimation ------------------

print(f"Estimating local PCA for for igg {DATA_NAME} data.")

pca_iter = local_weighted_pca_iter(
    x_pts=final_dataset[f"points|{DATA_NAME}|wc"],
    weights=final_dataset[f"affs|{DATA_NAME}-{DATA_NAME}|wc-wc"][:EIGENGAP_ESTIM_SUBSAMPLE_SIZE],
    ncomp=16,
    bsize=256,
    in_place_norm=True,
    needs_norm=True,
)
final_dataset["wlpca_eigvals"][DATA_NAME] = np.concatenate(
    [eigen_pair[0] for eigen_pair in pca_iter], axis=0
)

