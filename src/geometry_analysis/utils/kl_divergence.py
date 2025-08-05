import numpy as np
import scipy as sp
from geometry_analysis.geometry import dist
from geometry_analysis.io import Database
from geometry_analysis.sampling import sample_array, train_val_test_split
from geometry_analysis.utils import create_grid_1d

KDE_BANDWIDTHS = create_grid_1d(start=0.1, stop=0.6, step_size=0.01, scale="linear")
medium_dataset = Database(database_name="hem_latent_vecs_256_all")


def log_probs(sample_pts, model_pts, bw, dists=None):

    if dists is None:
        dists = dist(sample_pts, model_pts)
        dists **= 2

    npts, nfeats = model_pts.shape

    dists = dists / (bw**2)
    dists *= -0.5

    log_p = sp.special.logsumexp(dists, axis=1)
    log_p -= np.log(npts) + nfeats * 0.5 * np.log(2.0 * np.pi) + nfeats * np.log(bw)

    return log_p


for pn in ("exp", "sim"):

    npts = medium_dataset[f"points|{pn}|clean"].shape[0]

    kde_idx = np.sort(sample_array(data=npts, num_or_pct=10000))
    train_idx, val_idx, test_idx = train_val_test_split(kde_idx, val_p=0.1, test_p=0.1)

    train_val_idx = np.sort(np.concatenate([train_idx, val_idx]))
    print(len(train_idx), len(val_idx), len(test_idx))
    for mask_name, idx in (
        ("all", kde_idx),
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
        ("trainval", train_val_idx),
    ):
        mask = np.zeros(npts, dtype=np.bool_)
        mask[idx] = True
        medium_dataset["masks"][f"{pn}_kde{mask_name}"] = mask

    x_pts_train = medium_dataset[f"points|{pn}|kdetrain"]
    x_pts_val = medium_dataset[f"points|{pn}|kdeval"]

    dists = dist(x_pts_val, x_pts_train)
    dists **= 2

    npts = x_pts_train.shape[0]
    nfeats = x_pts_train.shape[1]

    for bw in KDE_BANDWIDTHS:

        log_p = log_probs(x_pts_val, x_pts_train, bw, dists)
        log_likelihood = np.sum(log_p)

        print("bw:", bw, "avg_score:", log_likelihood)

exp_bw = 0.36
sim_bw = 0.51

exp_pts_train = medium_dataset[f"points|exp|kdetrainval"]
exp_pts_test = medium_dataset[f"points|exp|kdetest"]

sim_pts_train = medium_dataset[f"points|sim|kdetrainval"]
sim_pts_test = medium_dataset[f"points|sim|kdetest"]

pts = (
    ("exp", exp_pts_train, exp_pts_test, exp_bw),
    ("sim", sim_pts_train, sim_pts_test, sim_bw),
)

for p_name, p_train, p_test, p_bw in pts:
    for q_name, q_train, _, q_bw in pts:

        # Step 1: Compute KDE for Q evaluated at points from p_test
        log_q = log_probs(p_test, q_train, q_bw)

        # Step 2: Compute KDE for P evaluated at points from p_test
        log_p = log_probs(p_test, p_train, p_bw)

        # Step 3: Compute the KL divergence as the weighted average of the log-ratio
        kl_divergence = np.mean(log_p - log_q)  # This approximates the KL divergence
        cross_entropy = np.mean(-log_q)

        print(f"KL(p={p_name} || q={q_name}) = {kl_divergence}")
        print(f"H(p={p_name} || q={q_name}) = {cross_entropy}")
