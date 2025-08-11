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
    riemannian_relaxation, nadaraya_watson_kernel_extension
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

# This needs to be the name of the folder in data/ with all latent data from cryosbi
DATASET_NAME = "hemagglutinin_data"

SIM_NAME = "sim"
EXP_NAME = "exp"
DATA_NAMES = (EXP_NAME, SIM_NAME)
DATA_SUBSAMPLE_KEY_PAIRS = {
    EXP_NAME: EXP_SUBSAMPLE_KEY_PAIRS,
    SIM_NAME: SIM_SUBSAMPLE_KEY_PAIRS,
}

TRAIN_NAME = "train"

CLEAN_NAME = "clean"
UNIFORM_NAME = "unif"

all_dataset = Database(database_name=f"{DATASET_NAME}", mode="append")
final_dataset = Database(database_name=f"{DATASET_NAME}_final", mode="append")

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

IES_SUBSAMPLE_SIZE = 500
TSLASSO_SUBSAMPLE_SIZE = 500

GRAD_ESTIM_SUBSAMPLE_SIZE = 5000
EIGENGAP_ESTIM_SUBSAMPLE_SIZE = 5000

RR_SCALE = 900.0

# ------------------ DATA SPLIT ------------------
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
    print(f"Running outlier removal for hem {data_name} data.")
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
    print(f"Uniformly subsampling hem {data_name} data.")

    compute_distance_statistics(all_dataset, data_name, CLEAN_NAME, rs=RS)
    compute_uniform_sample(all_dataset, data_name, CLEAN_NAME, EMB_NPTS)
    # Saves subsample of "all_dataset" into "final_dataset"
    subsample_dataset(all_dataset, final_dataset, UNIFORM_NAME, key_pairs)

# ------------------ INTRINSIC DIMENSIONALITY ESTIMATION ------------------
# Compute distances to closest KS neighbors and for a given set of radii RS compute the number of neighbors.
for data_name in DATA_NAMES:
    print(f"Computing distance statistics for hem {data_name} data.")
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
  
  # Compute the geometry matrices: distance, gaussian affinity and geometric laplacian using the
  # radius and bandwidth suggested by the previous analysis
  print(f"Computing geometry matrices for hem {data_name} data.")
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
  affs = final_dataset["affs"][pair_key]
  degrees = reduce_arr_to_degrees(affs, axis=1)
  final_dataset["masks"][f"{data_name}_wc"] = degrees >= np.percentile(degrees, 5)
  
  
  # if data_name == SIM_NAME:
  #
  #     # Subsamples final dataset to only have well-connected ("wc") points
  #     print(f"Subsampling hem {data_name} data to only well-connected points.")
  #     subsample_dataset(final_dataset, final_dataset, "wc", SIM_SUBSAMPLE_KEY_PAIRS)
  
  print(f"Running spectral embedding for hem {data_name} data.")
  eigvals, embedding = spectral_embedding(
      affs=final_dataset[f"affs|{pair_key}|wc-wc"],
      ncomp=20,
      eps=eps,
      eigen_solver="amg",
  )
  final_dataset["lap_eigvals"][data_name] = eigvals
  final_dataset["lap_eigvecs"][data_name] = embedding
  
  # ------------------ FEATURE SELECTION ------------------
  # I'm taking only one d for IES because the algorithm is very slow. Definitely need to improve it.
  # Look at the json file and display the top 3 axes selected by IES as they will be the embedding coordinates
  # have the smallest frequency and are as independent as possible w.r.t. to the objective defined in the paper.
  final_dataset["ies"][data_name] = ies(
      emb_pts=final_dataset["lap_eigvecs"][data_name],
      emb_eigvals=final_dataset["lap_eigvals"][data_name],
      lap=final_dataset[f"laps|{pair_key}|wc-wc"],
      ds=3,
      s=DS[data_name][1],
      sample=IES_SUBSAMPLE_SIZE,
  )

#------------------ EXTEND PARAMETERS ------------------
#We will use kernel regression to extend the parameters from the simulated data to the real one

sim_pts, exp_pts = final_dataset[f"points|sim|wc"], final_dataset[f"points|exp|wc"]

final_dataset["dists"]["exp_wc-sim_wc"] = dist(exp_pts, sim_pts, threshold=RADIUS[EXP_NAME])
final_dataset["affs"]["exp_wc-sim_wc"] = affinity(final_dataset["dists"]["exp_wc-sim_wc"], eps=EPS[EXP_NAME])

# There are some points in the real set which are not close to any sim data. We
# set those values to the average of the embedding dimension.
exp_sim_affs = final_dataset["affs"]["exp_wc-sim_wc"]
weights = reduce_arr_to_degrees(exp_sim_affs, axis=1)

for param in ("conf", "snr", "rot", "shift", "sigma", "defoc"):

    sim_param = final_dataset[f"params|sim_{param}|wc"]

    exp_predicted_param = nadaraya_watson_kernel_extension(
        sim_param, exp_sim_affs
    )
    exp_predicted_param[weights == 0.0] = np.mean(sim_param, axis=0)

    final_dataset["params"][f"exp_{param}_wc"] = exp_predicted_param

# ------------------ Gradient Estimation and TSLasso ------------------

for data_name in DATA_NAMES:
    pair_name = f"{data_name}-{data_name}"

    if data_name == "sim":
        func_vals = [final_dataset[f"params|sim_{sp}|wc"] for sp in SIM_PARAMS]
    else:
        func_vals = [final_dataset["params"][f"exp_{sp}_wc"] for sp in SIM_PARAMS]

    funcs = np.concatenate(
        [np.expand_dims(fv, axis=1) if fv.ndim == 1 else fv for fv in func_vals], axis=1
    )

    final_dataset["grads"][f"{data_name}_wc"] = local_grad_estimation(
        x_pts=final_dataset[f"points|{data_name}|wc"],
        f0_vals=funcs,
        f_vals=funcs,
        weights=final_dataset[f"affs|{data_name}-{data_name}|wc-wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE],
        bsize=50,
        ncomp=10,
    )

    x_pts = final_dataset[f"points|{data_name}|wc"]
    grads = final_dataset["grads"][f"{data_name}_wc"]
    affs = final_dataset[f"affs|{data_name}-{data_name}|wc-wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE]

    if data_name == "sim":
        snr = final_dataset[f"params|sim_snr|wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE]
    else:
        snr = final_dataset["params"]["exp_snr_wc"][:GRAD_ESTIM_SUBSAMPLE_SIZE]

    for percentile in create_grid_1d(start=0, stop=90, step_size=5, scale="int"):

        cutoff = np.percentile(snr, q=percentile)
        print(f"Running TSLasso for percentile={percentile} and cutoff={cutoff}.")

        sample = np.flatnonzero(snr >= cutoff)
        sample = np.sort(sample_array(sample, num_or_pct=TSLASSO_SUBSAMPLE_SIZE))

        tslasso_results = tslasso(
            x_pts=x_pts,
            affs=affs,
            grads=grads,
            ncomp=4,
            sample=sample,
            bsize=128,
            lr=100.0,
            l2_reg=0.0,
            max_nlamb=20,
            max_niter=500,
            tol=1e-14,
        )
        final_dataset["tslasso"][f"{data_name}_{percentile}"] = tslasso_results

# ------------------ LOCAL PCA for the Eigengap Dim Estimation ------------------

for data_name in DATA_NAMES:

    print(f"Estimating local PCA for for hem {data_name} data.")

    pca_iter = local_weighted_pca_iter(
        x_pts=final_dataset[f"points|{data_name}|wc"],
        weights=final_dataset[f"affs|{data_name}-{data_name}|wc-wc"][:EIGENGAP_ESTIM_SUBSAMPLE_SIZE],
        ncomp=16,
        bsize=256,
        in_place_norm=True,
        needs_norm=True,
    )
    final_dataset["wlpca_eigvals"][data_name] = np.concatenate(
        [eigen_pair[0] for eigen_pair in pca_iter], axis=0
    )


# # ------------------ Riemannian Relaxation ------------------
# NOTE: this step takes substantially longer than the previous, on the order of 2-3 hours for full iters below.
#for data_name in DATA_NAMES:
#
#    lap = final_dataset[f"laps|{data_name}-{data_name}|wc-wc"]
#    lap_eps = RADIUS[data_name] / RADIUS_EPS_RATIO
#
#    niter = 20
#    emb_pts = final_dataset["lap_eigvecs"][data_name] * RR_SCALE
#
#    for i in range(0, 250, niter):
#
#        print(f"Starting iters {i} to {niter + i}!")
#        emb_pts = riemannian_relaxation(
#            emb_pts=emb_pts,
#            lap=lap,
#            lap_eps=lap_eps,
#            d=3,
#            orth_eps=0.5,
#            maxiter=niter,
#        )
#
#        key = f"{data_name}_relax_{niter + i}"
#
#        final_dataset["lap_eigvecs"][key] = emb_pts
