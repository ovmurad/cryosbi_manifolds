import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from geometry_analysis.geometry.intrinsic_dim_estimation import (
    slope_estimation,
    log_ratio_estimation,
    eigen_gap_estimation,
    levina_bickel,
)
from geometry_analysis.io import Database
from geometry_analysis.visualization import make_radius_dict_df

sns.set_style("whitegrid")
palette = {"hem_exp": "red", "hem_sim": "blue", "igg_sim": "green"}

igg_dataset = Database(database_name="igg_latent_vecs_256_final")
igg_dists = {dk: igg_dataset["knn_dists"][f"{dk}-{dk}"] for dk in ("sim",)}
igg_counts = {dk: igg_dataset["n_counts"][f"{dk}-{dk}"] for dk in ("sim",)}
igg_eigvals = {dk: igg_dataset["wlpca_eigvals"][dk] for dk in ("sim",)}

hem_dataset = Database(database_name="hem_latent_vecs_256_final")
hem_dists = {dk: hem_dataset["knn_dists"][f"{dk}-{dk}"] for dk in ("sim", "exp")}
hem_counts = {dk: hem_dataset["n_counts"][f"{dk}-{dk}"] for dk in ("sim", "exp")}
hem_eigvals = {dk: hem_dataset["wlpca_eigvals"][dk] for dk in ("sim", "exp")}


def plot_slopes():

    def make_slope_df(data_key, log_radii, log_mean_counts):
        slope_df = pd.DataFrame(
            {
                "Radius": log_radii,
                "Log Avg. Count": log_mean_counts,
            }
        )
        slope_df["Dataset"] = data_key
        return slope_df

    data_key_to_counts = {
        "hem_exp": {r: c for r, c in hem_counts["exp"].items() if 8.0 <= r <= 18.0},
        "hem_sim": {r: c for r, c in hem_counts["sim"].items() if 8.0 <= r <= 18.0},
        "igg_sim": {r: c for r, c in igg_counts["sim"].items() if 8.0 <= r <= 18.0},
    }

    slope_estimation_results = {
        dk: slope_estimation(
            np.array(list(radius_to_counts.keys())),
            np.array(list(radius_to_counts.values())),
        )
        for dk, radius_to_counts in data_key_to_counts.items()
    }

    print(slope_estimation_results)

    results_df = pd.concat(
        [
            make_slope_df(dk, dk_results["log_radii"], dk_results["log_mean_counts"])
            for dk, dk_results in slope_estimation_results.items()
        ]
    )

    print(f"Estimate of intrinsic dimension(slope of the graph):")
    for dk, dk_results in slope_estimation_results.items():
        print(
            f'- {dk} data: {dk_results["slope"]: .4f} (with R-squared: {dk_results["r_value"]:.4f}).'
        )

    sns.lmplot(
        data=results_df,
        x="Radius",
        y="Log Avg. Count",
        hue="Dataset",
        ci=None,
        palette=palette,
    )
    plt.text(
        2.5,
        2.6,
        f'Hg. Sim Slope:  {slope_estimation_results["hem_sim"]["slope"]: .2f}',
        size=12,
    )
    plt.text(
        2.5,
        2.2,
        f'Hg.. Exp Slope: {slope_estimation_results["hem_exp"]["slope"]: .2f}',
        size=12,
    )
    plt.text(
        2.5,
        1.8,
        f'IgG. Sim Slope: {slope_estimation_results["igg_sim"]["slope"]: .2f}',
        size=12,
    )

    # Get current axes
    ax = plt.gca()

    # Get x-axis in log scale and then convert them to linear scale
    x_ticks = ax.get_xticks()[1:-1]
    ax.set_xticks(x_ticks)
    exp_x_ticks_labels = np.around(x_ticks, 2)
    ax.set_xticklabels(exp_x_ticks_labels)

    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Log Radius", fontsize=12)
    ax.set_ylabel("Log Avg. Count", fontsize=12)

    plt.show()


def plot_dd():
    def plot_log_ratio_df(df, ax, radius=None, has_xlabel=True, has_ylabel=True):
        ax = sns.kdeplot(
            df,
            x="Estimated d",
            hue="Dataset",
            ax=ax,
            linewidth=0,
            common_norm=False,
            fill=True,
            palette=palette,
        )
        ax.set_xlabel("Estimated d" if has_xlabel else "", fontsize=12)
        ax.set_ylabel("Density" if has_ylabel else "", fontsize=12)
        ax.set_xlim((1, 10))

        if radius is not None:
            ax.set_title(f"R = {radius:.3f}", size=12)

        ax.tick_params(axis="both", labelsize=12)

    ### --- Your existing data prep ---
    data_key_to_counts = {
        "hem_exp": {r: c for r, c in hem_counts["exp"].items() if 8.5 <= r},
        "hem_sim": {r: c for r, c in hem_counts["sim"].items() if 8.5 <= r},
        "igg_sim": {r: c for r, c in igg_counts["sim"].items() if 8.5 <= r},
    }

    log_ratio_estimation_results = {
        dk: log_ratio_estimation(
            list(radius_to_counts.keys()), np.array(list(radius_to_counts.values()))
        )
        for dk, radius_to_counts in data_key_to_counts.items()
    }

    results_df = pd.concat(
        [
            make_radius_dict_df(
                dict(zip(dk_results["radii"], dk_results["log_ratios"])),
                "Estimated d",
                {"Dataset": dk},
            )
            for dk, dk_results in log_ratio_estimation_results.items()
        ]
    )

    print("Expected Log Ratio Analysis Results:")
    for dk, dk_results in log_ratio_estimation_results.items():
        print(f"For {dk} data:")
        print(
            f'- Intrinsic dimension estimation(avg. over all radii): {dk_results["exp_log_ratio"]: .4f}.'
        )
        for rad, rad_log_ratios in zip(
            dk_results["radii"], dk_results["radii_exp_log_ratio"]
        ):
            print(
                f"- Intrinsic dimension estimation(avg. only over radius {rad}): {rad_log_ratios: .4f}."
            )

    plot_log_ratio_df(results_df, plt.gca())
    plt.show()

    radii = log_ratio_estimation_results["hem_exp"]["radii"]
    n_cols = 2
    n_rows = len(radii) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for i, r in enumerate(radii):
        plot_log_ratio_df(
            results_df[results_df["Radius"] == r],
            axes[i],
            radius=r,
            has_xlabel=((i // n_cols) == n_cols),
            has_ylabel=False,
        )

    plt.tight_layout()
    plt.show()

    # Compute mean per dataset and radius
    grouped_df = (
        results_df.groupby(["Dataset", "Radius"])["Estimated d"].mean().reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=grouped_df, x="Radius", y="Estimated d", hue="Dataset", palette=palette
    )
    plt.xlabel("Radius", fontsize=12)
    plt.ylabel("Mean Estimated d", fontsize=12)
    # plt.title("Mean Estimated Intrinsic Dimension per Dataset, grouped by Radius")
    plt.legend(title="Dataset")
    plt.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_eigengap():

    def make_eigengap_df(dataset, estim_d):
        eigengap_df = pd.DataFrame(
            {
                "Estimated d": estim_d,
            }
        )
        eigengap_df["Dataset"] = dataset
        return eigengap_df

    data_key_to_eigvals = {
        "hem_exp": hem_eigvals["exp"],
        "hem_sim": hem_eigvals["sim"],
        "igg_sim": igg_eigvals["sim"],
    }

    eigen_gap_max_estimation_results = {
        dk: eigen_gap_estimation(eigvals, mode="max")
        for dk, eigvals in data_key_to_eigvals.items()
    }

    eigen_gap_softmax_estimation_results = {
        dk: eigen_gap_estimation(eigvals, mode="softmax")
        for dk, eigvals in data_key_to_eigvals.items()
    }

    max_results_df = pd.concat(
        [
            make_eigengap_df(dk, estimated_d)
            for dk, estimated_d in eigen_gap_max_estimation_results.items()
        ]
    )
    softmax_results_df = pd.concat(
        [
            make_eigengap_df(dk, estimated_d)
            for dk, estimated_d in eigen_gap_softmax_estimation_results.items()
        ]
    )

    print("For max eigen gap estimation:")
    for dk, estimated_d in eigen_gap_max_estimation_results.items():
        print(
            f"For {dk} data, the avg. over all eigen gap indices is: {estimated_d.mean(): .4f}."
        )

    print("For softmax eigen gap estimation:")
    for dk, estimated_d in eigen_gap_softmax_estimation_results.items():
        print(
            f"For {dk} data, the avg. over all eigen gap indices is: {estimated_d.mean(): .4f}."
        )

    sns.histplot(
        max_results_df,
        x="Estimated d",
        hue="Dataset",
        palette=palette,
        stat="percent",
        discrete=True,
        multiple="dodge",
        shrink=0.8,
        common_norm=False,
    )
    plt.title("Estimated intrinsic dimension using the max eigen gap method")
    plt.show()

    ax = sns.kdeplot(
        softmax_results_df,
        x="Estimated d",
        hue="Dataset",
        palette=palette,
        common_norm=False,
        linewidth=0,
        fill=True,
    )
    # plt.title("Estimated d using the Softmax Eigen Gap Method")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Estimated d", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    plt.show()


def plot_lb():

    def plot_kde(df, ax, k=None, has_xlabel=True, has_ylabel=True):
        ax = sns.kdeplot(
            df,
            x="Estimated d",
            hue="Dataset",
            ax=ax,
            linewidth=0,
            common_norm=False,
            fill=True,
            palette=palette,
        )
        ax.set_xlabel("Estimated d" if has_xlabel else "", fontsize=12)
        ax.set_ylabel("Density" if has_ylabel else "", fontsize=12)
        ax.set_xlim((1, 15))

        if k is not None:
            ax.set_title(f"k = {k}", size=12)

        ax.tick_params(axis="both", labelsize=12)

    data_key_to_dists = {
        "hem_exp": {k: v for k, v in hem_dists["exp"].items() if 30 <= k <= 100},
        "hem_sim": {k: v for k, v in hem_dists["sim"].items() if 30 <= k <= 100},
        "igg_sim": {k: v for k, v in igg_dists["sim"].items() if 30 <= k <= 100},
    }

    levina_bickel_results = {
        dk: levina_bickel(k_to_counts) for dk, k_to_counts in data_key_to_dists.items()
    }

    all_dfs = []
    for dk, results in levina_bickel_results.items():
        for k, dim_estimates in results.items():
            temp_df = pd.DataFrame(
                {"Estimated d": dim_estimates, "Dataset": dk, "k": k}
            )
            all_dfs.append(temp_df)

    results_df = pd.concat(all_dfs)

    plt.figure(figsize=(8, 5))
    plot_kde(results_df, plt.gca())
    # plt.title("Levina-Bickel Intrinsic Dimension (all k)")
    plt.show()

    unique_ks = sorted(results_df["k"].unique())
    n_cols = 2
    n_rows = (len(unique_ks) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for i, k in enumerate(unique_ks):
        plot_kde(
            results_df[results_df["k"] == k],
            axes[i],
            k=k,
            has_xlabel=((i // n_cols) == (n_rows - 1)),
            has_ylabel=False,
        )

    plt.tight_layout()
    plt.show()

    grouped_df = (
        results_df.groupby(["Dataset", "k"])["Estimated d"].mean().reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_df, x="k", y="Estimated d", hue="Dataset", palette=palette)
    plt.xlabel("Number of neighbors (k)", fontsize=12)
    plt.ylabel("Mean Estimated d", fontsize=12)
    # plt.title("Mean Estimated Intrinsic Dimension (Levina-Bickel) grouped by k")
    plt.legend(title="Dataset")
    plt.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()


# plot_slopes()
plot_lb()
plot_dd()
plot_eigengap()
