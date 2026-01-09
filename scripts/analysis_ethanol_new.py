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
    doubling_dimension,
    score_from_mult_knn_dist,
    score_from_mult_n_count,
    density_from_mult_knn_dist,
    density_from_mult_n_count,
)
from geometry_analysis.utils.script_utils import (
    run_dimensionality_estimation,
    plot_n_count_or_knn_dist,
    index_to_mask
)

def main():
    # Number of points
    EMB_NPTS = 5000
    SPLIT_NPTS = {"train": 10000}
    EIGENGAP_ESTIM_SUBSAMPLE_SIZE = 5000

    # Check OUTLIER REMOVAL
    ALPHA = 0.20  # percentile cutoff for outlier removal(higher means more data is cutoff)
    OUTLIER_ALGO = "knn_dists"  # algorithm for MV classification
    
    # Check MANIFOLD LEARNING.
    # TODO: should there be a command here on what to set this RADIUS as?
    RADIUS = 15.0  # cutoff radius used for embedding.
    RADIUS_EPS_RATIO = 3.0  # Good default value
    EPS = RADIUS / RADIUS_EPS_RATIO
    
    # Check INTRINSIC DIMENSIONALITY ESTIMATION.
    DS = [6, 7, 8]  # approximate intrinsic dimensions.
       
    ## ------------------ Load Data ------------------
    data_dir = "../data/ethanol_data"
    all_dataset = np.load(f"{data_dir}/points/angles.npy")
    
    ## ------------------ DATA SPLIT ------------------
    # Compute training set mask
    npts = all_dataset.shape[0]
    split_points = np.cumsum(SPLIT_NPTS["train"])
    permuted = RANDOM_STATE.permutation(npts)
    split_indices = np.split(permuted, split_points)[:-1]
    train_mask = index_to_mask(npts, split_indices)
    
    # ------------------ ANALYSIS PARAMETERS ------------------
    # Check LOCAL STATISTICS
    # the ks for the k-nn distances
    MIN_K, MAX_K = 10, 120
    KS = create_grid_1d(start=MIN_K, stop=MAX_K, step_size=10, scale="int")
    
    # the rs of balls for counting neighbors
    # it is important to not set MAX_R too large as that will bias the dimension estimate down.
    # check some of the plots below and make sure that the for large rs, the histogram of the number of neighbors does
    # not shift overwhelmingly to the right and stays relatively uniform.
    MIN_R, MAX_R = 1.0, 2.0
    step_size = 0.1
    RS = create_grid_1d(start=MIN_R, stop=MAX_R, step_size=step_size, scale="linear")
    
    # ------------------ OUTLIER REMOVAL ------------------
    # Plot histograms of the number of neighbors for various radii and the knn distance for various k.
    # Use these to find a cutoff percentage in [0, 1] for outlier detection.
    # Intuitively, the outliers will be either points with few neighbors(in the bottom alpha/beta percentile)
    # or points with large knn distance(in the top alpha/beta percentile).
    # For the number of neighbors, the histogram will have large bins close to 0. We need to remove those.
    # For the knn distance of neighbors, the histogram will have a long tail of points with large k-nn distances.
    # Pick percentiles alpha(sim)/beta(exp) that reliably eliminate the anomalous bins.
    # Compute local statistics(closest distance to nearest k neighbors or neighbor counts for different radii)
    print(f"Running outlier removal.")
    if OUTLIER_ALGO == "knn_dists":
        knn_dists = dist(x_pts=all_dataset[train_mask], y_pts=None, threshold=KS)
        pts_scores = score_from_mult_knn_dist(knn_dists, knn_dists)
        #plot_n_count_or_knn_dist(n_count=None, knn_dist=knn_dists)
    elif OUTLIER_ALGO == "n_counts":
        n_counts = count_x(x_pts=all_dataset[train_mask], y_pts=None, threshold=RS)
        pts_scores = score_from_mult_n_count(n_counts, n_counts)
        #plot_n_count_or_knn_dist(n_count=n_counts, knn_dist=None)
    else:
        raise ValueError
    
    clean_mask = train_mask.copy()
    clean_mask[train_mask] = pts_scores >= ALPHA

    ## ------------------ UNIFORM RESAMPLING ------------------
    # Resample the data to obtain uniform samples.
    print(f"Uniformly subsampling data.")
    n_counts = count_x(x_pts=all_dataset[clean_mask], y_pts=None, threshold=RS)
    unif_mask = new_compute_uniform_sample(all_dataset, clean_mask, n_counts, EMB_NPTS)
    subsampled_dataset = all_dataset[unif_mask]

    # ------------------ INTRINSIC DIMENSIONALITY ESTIMATION ------------------
    # Compute distances to closest KS neighbors and for a given set of radii RS compute the number of neighbors.
    print(f"Computing distance statistics for sub-sampled dataset.")
    knn_dists = dist(x_pts=subsampled_dataset, y_pts=None, threshold=KS)
    n_counts = count_x(x_pts=subsampled_dataset, y_pts=None, threshold=RS)

    # Run all dimensionality estimation algorithms. Look at the outputs and pick a reasonable span of dimensions.
    # We'll use the average for kernel regression/density estimation and all values for estimating the Laplacian bandwidth.
    # We cannot perform eigen-gap at this point because we haven't estimated the bandwidth.
    # That algorithm will be computed later.
    print(f"Running intrinsic dimensionality estimation for sub-sampled data.")
    run_dimensionality_estimation(
        n_count=n_counts,
        knn_dist=knn_dists,
        algos=("dd", "lb", "cd"),
    )
    dim_estimates_dict = doubling_dimension(n_counts)
    dim_estimates = np.stack(list(dim_estimates_dict.values()))
    dim_estimates[np.isnan(dim_estimates)] = np.mean(
        dim_estimates[~np.isnan(dim_estimates)], axis=0
    )
    estimated_d = np.mean(dim_estimates, axis=0)

    #------------------ MANIFOLD LEARNING ------------------
    # Select the bandwidth for the affinity matrix which will then be used for diffusion maps.
    # We pick the radii slightly above the radii for which the distortion reaches its minimum.
    # There is generally no harm in picking a slightly larger radius
    # than the one that achieves the minimum distortion. 
    # Particularly for this type of data that seems to have varying topology and probably varying intrinsic
    # dimensionality, I find it useful to go above the minimum computed here which seems to give
    # a lower bound under which DM fails.
    print(f"Running bandwidth distortion estimation for the sub-sampled data.")
    eval_radii = radii_distortions(
        x_pts=all_dataset[unif_mask],
        ds=DS,
        radii=RS,
        sample=64,
        rad_eps_ratio=RADIUS_EPS_RATIO,
        bsize=None,
    )

    # Compute the geometry matrices: distance, gaussian affinity and geometric laplacian using the
    # radius and bandwidth suggested by the previous analysis
    print(f"Computing geometry matrices for the sub-sampled data.")
    dists = dist(
        x_pts=all_dataset[unif_mask],
        threshold=RADIUS,
    )
    affs = affinity(
        dists=dists, eps=EPS
    )
    # Embed the data
    # Important!!!: The warning that the affinity matrix is not connected is a good indication that the
    # radius needs to be larger and that the embedding will probably fail miserably.
    # Subsamples final dataset to only have well-connected ("wc") points
    print(f"Re-subsampling the data to only well-connected points.")
    degrees = reduce_arr_to_degrees(affs, axis=1)
    wc_subsampled_mask = degrees >= np.percentile(degrees, 5)
    wc_subsampled_dataset = subsampled_dataset[wc_subsampled_mask]
    wc_subsampled_dists = dists[wc_subsampled_mask, :][:, wc_subsampled_mask]
    wc_subsampled_affs = affs[wc_subsampled_mask, :][:, wc_subsampled_mask]
    wc_subsampled_laps = laplacian(
        affs=wc_subsampled_affs, eps=EPS
    )

    print(f"Running spectral embedding for the re-subsampled data.")
    eigvals, embedding = spectral_embedding(
        affs=wc_subsampled_affs,
        ncomp=20,
        eps=EPS,
        eigen_solver="amg",
    )
    lap_eigvals = eigvals 
    lap_eigvecs = embedding
    
    ## ------------------ LOCAL PCA for the Eigengap Dim Estimation ------------------
    print(f"Estimating local PCA.")
    pca_iter = local_weighted_pca_iter(
        x_pts=wc_subsampled_dataset,
        weights=wc_subsampled_affs[:EIGENGAP_ESTIM_SUBSAMPLE_SIZE],
        ncomp=16,
        bsize=256,
        in_place_norm=True,
        needs_norm=True,
    )
    wlpca_eigvals = np.concatenate(
        [eigen_pair[0] for eigen_pair in pca_iter], axis=0
    )

    results_dict = {}
    results_dict["knn_dists"] = knn_dists
    results_dict["n_counts"] = n_counts
    results_dict["wlpca_eigvals"] = wlpca_eigvals
    results_dict["lap_eigvals"] = lap_eigvals
    results_dict["lap_eigvecs"] = lap_eigvecs

    import pickle
    filename = "data_dict.pkl"
    filehandler = open(filename, 'wb')
    pickle.dump(results_dict, filehandler)


def new_compute_uniform_sample(
    dataset, cur_mask, n_count, sample_size: int
) -> None:
    
    pw_estim_d = doubling_dimension(n_count)
    
    # where ever we have nans, we replace with the average value
    for pw_r_estim_d in pw_estim_d.values():
        nan_mask = np.isnan(pw_r_estim_d)
        pw_r_estim_d[nan_mask] = np.mean(pw_r_estim_d[~nan_mask])
    
    # estimate the density
    n_count = {r: n_count[r] for r in pw_estim_d.keys()}
    estim_density = density_from_mult_n_count(n_count, pw_estim_d)
    
    # invert the density and normalize
    inv_density = 1.0 / (estim_density + 1e-16)
    inv_density = inv_density / np.sum(inv_density)
    
    # sample data uniformly from the remaining clean points
    unif_index = RANDOM_STATE.choice(
        np.flatnonzero(cur_mask), size=sample_size, replace=False, p=inv_density
    )
    
    # create and save the mask corresponding to the new uniform dataset
    unif_mask = np.zeros_like(cur_mask)
    unif_mask[unif_index] = True
    return unif_mask

if __name__ == '__main__':
    main()