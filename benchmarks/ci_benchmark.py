import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmarks.DGM import DGP_REGISTRY
from pgmpy.estimators import CITests


DGM_TO_CITESTS = {
    "linear_gaussian": ["pearsonr", "gcm", "pillai"],
    "nonlinear_gaussian": ["gcm", "pillai"],
    "non_gaussian_continuous": ["gcm", "pillai"],
    "discrete_categorical": [
        "chi_square",
        "g_sq",
        "log_likelihood",
        "pillai",
    ],
    "mixed_data": ["pillai"],
}

dgms = {name: DGP_REGISTRY[name] for name in DGM_TO_CITESTS.keys()}

ci_tests = {
    "pearsonr": CITests.pearsonr,
    "gcm": CITests.gcm,
    "chi_square": CITests.chi_square,
    "g_sq": CITests.g_sq,
    "log_likelihood": CITests.log_likelihood,
    "modified_log_likelihood": CITests.modified_log_likelihood,
    "pillai": CITests.pillai_trace,
}


def run_benchmark(
    dgms=dgms,
    dgm_to_citests=DGM_TO_CITESTS,
    ci_tests=ci_tests,
    sample_sizes=[20, 40, 80, 160, 320, 640],
    n_cond_vars=[1, 3, 5, 7],
    effect_sizes=np.linspace(0, 1, 6),
    n_repeats=10,
):

    results = []

    dgm_pbar = tqdm(dgms.items())
    for dgm_name, dgm in dgm_pbar:
        dgm_pbar.set_description(f"Running benchmark for {dgm_name}")

        compatible_tests = dgm_to_citests[dgm_name]
        for n_cond_var in tqdm(n_cond_vars, desc="No. of conditional variables", leave=False):
            for n in tqdm(sample_sizes, desc="Sample Size", leave=False):
                # Null case (conditionally independent, effect size = 0)
                for rep in range(n_repeats):
                    df = dgm(n_samples=n,
                             effect_size=0.0,
                             n_cond_vars=n_cond_var,
                             seed=rep,
                             )

                    z_cols = list(df.drop(['X', 'Y'], axis=1).columns)

                    for test_name in compatible_tests:
                        ci_func = ci_tests[test_name]
                        result = ci_func("X", "Y", z_cols, df, boolean=False)
                        # Robust extraction of p-value
                        if isinstance(result, tuple):
                            # Heuristic: p-value is usually last in tuple
                            if isinstance(result[-1], float):
                                p_val = result[-1]
                            else:
                                # fallback to first item
                                p_val = result[0]
                        else:
                            p_val = result

                        results.append(
                            {
                                "dgm": dgm_name,
                                "sample_size": n,
                                "n_cond_vars": n_cond_var,
                                "effect_size": 0.0,
                                "repeat": rep,
                                "ci_test": test_name,
                                "cond_independent": True,
                                "p_value": p_val,
                            }
                        )
                # Alternative case (conditionally dependent, effect size > 0)
                for eff in effect_sizes:
                    if eff == 0.0:
                        continue
                    for rep in range(n_repeats):
                        df = dgm(n_samples=n,
                                 effect_size=eff,
                                 n_cond_vars=n_cond_var,
                                 seed=rep,
                                 )
                        z_cols = list(df.drop(['X', 'Y'], axis=1).columns)

                        for test_name in compatible_tests:
                            ci_func = ci_tests[test_name]
                            result = ci_func("X", "Y", z_cols, df, boolean=False)
                            # Robust extraction of p-value
                            if isinstance(result, tuple):
                                if isinstance(result[-1], float):
                                    p_val = result[-1]
                                else:
                                    p_val = result[0]
                            else:
                                p_val = result

                            results.append(
                                {
                                    "dgm": dgm_name,
                                    "sample_size": n,
                                    "n_cond_vars": n_cond_var,
                                    "effect_size": eff,
                                    "repeat": rep,
                                    "ci_test": test_name,
                                    "cond_independent": False,
                                    "p_value": p_val,
                                }
                            )

    return pd.DataFrame(results)


def compute_summary(df_results, significance_levels=[0.001, 0.01, 0.05, 0.1]):
    """
    Computes Type I/II errors and power at multiple significance levels using collected p-values.
    """
    summary_rows = []
    group_cols = ["dgm", "sample_size", "n_cond_vars", "effect_size", "ci_test"]
    for keys, group in df_results.groupby(group_cols):
        null_group = group[group["cond_independent"]]
        alt_group = group[~group["cond_independent"]]
        for sl in significance_levels:
            type1 = (
                (null_group["p_value"] < sl).mean() if not null_group.empty else np.nan
            )
            type2 = (
                1 - (alt_group["p_value"] < sl).mean()
                if not alt_group.empty
                else np.nan
            )
            power = 1 - type2 if not np.isnan(type2) else np.nan
            summary_rows.append(
                dict(
                    zip(group_cols, keys),
                    significance_level=sl,
                    type1_error=type1,
                    type2_error=type2,
                    power=power,
                    N_null=len(null_group),
                    N_alt=len(alt_group),
                )
            )
    df_summary = pd.DataFrame(summary_rows)
    return df_summary


def plot_benchmarks(df_summary, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    methods = sorted(df_summary["ci_test"].unique())
    palette = sns.color_palette("Set1", n_colors=len(methods))

    for dgm in df_summary["dgm"].unique():
        df_dgm = df_summary[df_summary["dgm"] == dgm]
        sample_sizes = sorted(df_dgm["sample_size"].unique())
        n_cond_vars_list = sorted(df_dgm["n_cond_vars"].unique())
        significance_levels = sorted(df_dgm["significance_level"].unique())

        # Plot: Type II Error vs Significance Level for each effect size
        for eff in sorted(df_dgm["effect_size"].unique()):
            fig, axes = plt.subplots(
                len(sample_sizes),
                len(n_cond_vars_list),
                figsize=(4 * len(n_cond_vars_list), 2.5 * len(sample_sizes)),
                sharex=True,
                sharey=True,
            )
            if len(sample_sizes) == 1 and len(n_cond_vars_list) == 1:
                axes = np.array([[axes]])
            elif len(sample_sizes) == 1 or len(n_cond_vars_list) == 1:
                axes = axes.reshape(len(sample_sizes), len(n_cond_vars_list))

            for i, n in enumerate(sample_sizes):
                for j, ncv in enumerate(n_cond_vars_list):
                    ax = axes[i, j]
                    subset = df_dgm[
                        (df_dgm["sample_size"] == n)
                        & (df_dgm["n_cond_vars"] == ncv)
                        & (df_dgm["effect_size"] == eff)
                    ]
                    for method, color in zip(methods, palette):
                        s = subset[subset["ci_test"] == method]
                        if not s.empty:
                            x_vals = np.log10(s["significance_level"])
                            y_vals = np.log10(s["type2_error"])
                            sort_idx = np.argsort(x_vals)
                            ax.plot(
                                x_vals.iloc[sort_idx],
                                y_vals.iloc[sort_idx],
                                marker="o",
                                linestyle="-",
                                label=method,
                                color=color,
                            )
                    if i == 0:
                        ax.set_title(f"Cond.vars: {ncv}")
                    if j == 0:
                        ax.set_ylabel(f"n={n}\nlog10 Type II Error")
                    if i == len(sample_sizes) - 1:
                        ax.set_xlabel("log10 Significance Level")
                    ax.grid(True, alpha=0.4)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(methods),
                bbox_to_anchor=(0.5, 1.15),
            )
            fig.suptitle(
                f"Type II Error vs Significance Level for {dgm}, effect size={eff}",
                y=1.12,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fname1 = f"{plot_dir}/{dgm}_effect{eff}_typeII_vs_signif.png"
            plt.savefig(fname1, bbox_inches="tight")
            plt.close(fig)

        # ---Power vs Sample Size, all significance levels in plot ---
        for eff in sorted(df_dgm["effect_size"].unique()):
            fig, axes = plt.subplots(
                len(n_cond_vars_list),
                1,
                figsize=(7, 2.5 * len(n_cond_vars_list)),
                sharex=True,
                sharey=True,
            )
            if len(n_cond_vars_list) == 1:
                axes = [axes]
            for j, ncv in enumerate(n_cond_vars_list):
                ax = axes[j]
                for method, color in zip(methods, palette):
                    for idx, sl in enumerate(significance_levels):
                        subset = df_dgm[
                            (df_dgm["n_cond_vars"] == ncv)
                            & (df_dgm["effect_size"] == eff)
                            & (df_dgm["significance_level"] == sl)
                            & (df_dgm["ci_test"] == method)
                        ]
                        if not subset.empty:
                            sort_idx = np.argsort(subset["sample_size"])
                            linestyle = ["-", "--", "-.", ":"][idx % 4]
                            ax.plot(
                                subset["sample_size"].iloc[sort_idx],
                                subset["power"].iloc[sort_idx],
                                marker="o",
                                linestyle=linestyle,
                                color=color,
                                label=f"{method}, sl={sl}",
                                alpha=0.8,
                            )
                ax.set_title(f"Cond.vars: {ncv}")
                if j == 0:
                    ax.set_ylabel("Power")
                if j == len(n_cond_vars_list) - 1:
                    ax.set_xlabel("Sample Size")
                ax.grid(True, alpha=0.4)
            handles, labels = [], []
            for ax in axes:
                handles_ax, labels_ax = ax.get_legend_handles_labels()
                handles += handles_ax
                labels += labels_ax
            by_label = dict(zip(labels, handles))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper center",
                ncol=2 * len(methods),
                bbox_to_anchor=(0.5, 1.15),
            )
            fig.suptitle(
                f"Power vs Sample Size for {dgm}, effect size={eff} (all significance levels)",
                y=1.12,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fname2 = f"{plot_dir}/{dgm}_effect{eff}_power_vs_samplesize_allSL.png"
            plt.savefig(fname2, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    df_results = run_benchmark()
    df_results.to_csv("results/ci_benchmark_raw_result.csv", index=False)

    df_summary = compute_summary(df_results)
    df_summary.to_csv("results/ci_benchmark_summaries.csv", index=False)
    print(df_summary)
    print(
        "\nDetailed results and summary saved to ci_benchmark_raw_result.csv and ci_benchmark_summaries.csv"
    )
    raw_csv_path = "results/ci_benchmark_raw_result.csv"
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)

    plot_benchmarks(df_summary)
