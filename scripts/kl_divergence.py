import numpy as np
import scipy as sp
from geometry_analysis.geometry import dist
from geometry_analysis.sampling import train_val_test_split
from geometry_analysis.io import Database
from geometry_analysis.sampling import sample_array
from geometry_analysis.utils import create_grid_1d
from geometry_analysis.utils.script_utils import compute_split_masks


KDE_BANDWIDTHS = create_grid_1d(start=0.1, stop=0.6, step_size=0.01, scale="linear")

dataset = Database(database_name="hemagglutinin_data")


def log_probs(sample_pts, model_pts, bw, dists=None):

    if dists is None:

        dists = dist(sample_pts, model_pts)
        dists **= 2

    npts, nfeats = model_pts.shape

    dists = dists / (bw**2)
    dists *= -0.5

    log_p = sp.special.logsumexp(dists, axis=1)
    log_p -= np.log(npts) + nfeats * np.log(bw * np.sqrt(2.0 * np.pi))

    return log_p


sim_pts = dataset[f"points|sim|train"]
exp_pts = dataset[f"points|exp|train"]

sim_data_splits = {"train": sim_pts[:17000], "val": sim_pts[17000:20000], "test": sim_pts[20000:23000]}
exp_data_splits = {"train": exp_pts[:17000], "val": exp_pts[17000:20000], "test": exp_pts[20000:23000]}


# for data_name, data_splits in (("exp", exp_data_splits), ("sim", sim_data_splits)):
#
#     x_pts_train = data_splits["train"]
#     x_pts_val = data_splits["val"]
#
#     dists = dist(x_pts_val, x_pts_train)
#     dists **= 2
#
#     npts = x_pts_train.shape[0]
#     nfeats = x_pts_train.shape[1]
#
#     for bw in KDE_BANDWIDTHS:
#
#         log_p = log_probs(x_pts_val, x_pts_train, bw, dists)
#         log_likelihood = np.sum(log_p)
#
#         print("bw:", bw, "avg_score:", log_likelihood)

# exp_bw = 0.31
# sim_bw = 0.48
#
#
# pts = (
#     ("exp", exp_data_splits["train"], exp_data_splits["test"], exp_bw),
#     ("sim", sim_data_splits["train"], sim_data_splits["test"], sim_bw),
# )
#
# for p_name, p_train, p_test, p_bw in pts:
#     for q_name, q_train, _, q_bw in pts:
#
#         # Step 1: Compute KDE for Q evaluated at points from p_test
#         log_q = log_probs(p_test, q_train, q_bw)
#
#         # Step 2: Compute KDE for P evaluated at points from p_test
#         log_p = log_probs(p_test, p_train, p_bw)
#
#         # Step 3: Compute the KL divergence as the weighted average of the log-ratio
#         kl_divergence = np.mean(log_p - log_q)  # This approximates the KL divergence
#         cross_entropy = np.mean(-log_q)
#
#         print(f"KL(p={p_name} || q={q_name}) = {kl_divergence}")
#         print(f"H(p={p_name} || q={q_name}) = {cross_entropy}")