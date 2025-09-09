import os
import warnings
import shutil

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

                        if isinstance(result, tuple):
                            if len(result) >= 2 and isinstance(result[-1], (float, np.floating)):
                                p_val = float(result[-1])
                            elif isinstance(result[0], (float, np.floating)):
                                p_val = float(result[0])
                            else:
                                float_vals = [x for x in result if isinstance(x, (float, np.floating))]
                                if float_vals:
                                    p_val = float(float_vals[0])
                                else:
                                    raise ValueError(f"No valid float p-value found in result: {result}")
                        else:
                            p_val = float(result)

                        if not (0 <= p_val <= 1):
                            raise ValueError(f"Invalid p-value {p_val} for {test_name} - must be between 0 and 1")

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

                            if isinstance(result, tuple):
                                if len(result) >= 2 and isinstance(result[-1], (float, np.floating)):
                                    p_val = float(result[-1])
                                elif isinstance(result[0], (float, np.floating)):
                                    p_val = float(result[0])
                                else:
                                    float_vals = [x for x in result if isinstance(x, (float, np.floating))]
                                    if float_vals:
                                        p_val = float(float_vals[0])
                                    else:
                                        raise ValueError(f"No valid float p-value found in result: {result}")
                            else:
                                p_val = float(result)
                            
                            if not (0 <= p_val <= 1):
                                raise ValueError(f"Invalid p-value {p_val} for {test_name} - must be between 0 and 1")
                            

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
    if df_results.empty:
        raise ValueError("No benchmark results to compute summary from!")
    
    if 'p_value' not in df_results.columns:
        raise ValueError("p_value column missing from results!")
    
    summary_rows = []
    group_cols = ["dgm", "sample_size", "n_cond_vars", "effect_size", "ci_test"]
    
    for keys, group in df_results.groupby(group_cols):
        null_group = group[group["cond_independent"]]
        alt_group = group[~group["cond_independent"]]
        
        for sl in significance_levels:
            null_valid = null_group.dropna(subset=['p_value'])
            alt_valid = alt_group.dropna(subset=['p_value'])

            null_loss_rate = 1 - (len(null_valid) / len(null_group)) if len(null_group) > 0 else 0
            alt_loss_rate = 1 - (len(alt_valid) / len(alt_group)) if len(alt_group) > 0 else 0
            
            if null_loss_rate > 0.5:
                print(f"WARNING: Lost {null_loss_rate:.1%} of null data for {keys} due to invalid p-values")
            if alt_loss_rate > 0.5:
                print(f"WARNING: Lost {alt_loss_rate:.1%} of alternative data for {keys} due to invalid p-values")
            
            type1 = (
                (null_valid["p_value"] < sl).mean() if len(null_valid) > 0 else np.nan
            )
            type2 = (
                1 - (alt_valid["p_value"] < sl).mean()
                if len(alt_valid) > 0 else np.nan
            )
            power = 1 - type2 if not np.isnan(type2) else np.nan
            
            summary_rows.append(
                dict(
                    zip(group_cols, keys),
                    significance_level=sl,
                    type1_error=type1,
                    type2_error=type2,
                    power=power,
                    N_null=len(null_valid),
                    N_alt=len(alt_valid),
                    N_null_invalid=len(null_group) - len(null_valid),
                    N_alt_invalid=len(alt_group) - len(alt_valid),
                )
            )
    
    df_summary = pd.DataFrame(summary_rows)
    total_invalid = df_summary['N_null_invalid'].sum() + df_summary['N_alt_invalid'].sum()
    total_tests = len(df_results)
    invalid_rate = total_invalid / total_tests if total_tests > 0 else 0
    
    print(f"Data Quality Report:")
    print(f"  Total tests run: {total_tests}")
    print(f"  Invalid p-values: {total_invalid} ({invalid_rate:.1%})")
    
    if invalid_rate > 0.3:  
        raise ValueError(f"Too many test failures: {invalid_rate:.1%} of tests produced invalid p-values")
    elif invalid_rate > 0.1:  
        print(f"  WARNING: High failure rate of {invalid_rate:.1%}")
    
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
                figsize=(5 * len(n_cond_vars_list), 3 * len(sample_sizes)),  
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
                        if not s.empty and not s["type2_error"].isna().all():
                            x_vals = np.log10(s["significance_level"])
                            y_vals = np.log10(s["type2_error"])
                            valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isinf(y_vals))
                            if not valid_mask.any():
                                print(f"WARNING: No valid data points for {method} in {dgm} plot")
                                continue
                            if valid_mask.sum() < len(x_vals) * 0.8:  
                                lost_pct = (1 - valid_mask.sum() / len(x_vals)) * 100
                                print(f"WARNING: Lost {lost_pct:.1f}% of data points for {method} due to invalid values")
                            
                            sort_idx = np.argsort(x_vals[valid_mask])
                            ax.plot(
                                x_vals[valid_mask].iloc[sort_idx],
                                y_vals[valid_mask].iloc[sort_idx],
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
                    ax.set_aspect('auto') 
                    
            handles, labels = [], []
            for ax_row in axes:
                for ax in ax_row:
                    h, l = ax.get_legend_handles_labels()
                    for handle, label in zip(h, l):
                        if label not in labels:
                            handles.append(handle)
                            labels.append(label)
            
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
            plt.savefig(fname1, bbox_inches="tight", dpi=150)  
            plt.close(fig)

        # ---Power vs Sample Size, all significance levels in plot ---
        for eff in sorted(df_dgm["effect_size"].unique()):
            if eff == 0.0:  
                continue
                
            fig, axes = plt.subplots(
                len(n_cond_vars_list),
                1,
                figsize=(8, 3 * len(n_cond_vars_list)), 
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
                        if not subset.empty and not subset["power"].isna().all():
                            
                            valid_subset = subset.dropna(subset=['power'])
                            if len(valid_subset) == 0:
                                print(f"WARNING: No valid power data for {method} at significance level {sl}")
                                continue
                            if len(valid_subset) < len(subset) * 0.8:  
                                lost_pct = (1 - len(valid_subset) / len(subset)) * 100
                                print(f"WARNING: Lost {lost_pct:.1f}% of power data for {method} at sl={sl}")
                            
                            sort_idx = np.argsort(valid_subset["sample_size"])
                            linestyle = ["-", "--", "-.", ":"][idx % 4]
                            ax.plot(
                                valid_subset["sample_size"].iloc[sort_idx],
                                valid_subset["power"].iloc[sort_idx],
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
                ax.set_aspect('auto') 
                
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
            plt.savefig(fname2, bbox_inches="tight", dpi=150)  
            plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("Starting benchmark execution...")
    df_results = run_benchmark()
    print(f"Benchmark completed. Generated {len(df_results)} test results.")
    df_results.to_csv("results/ci_benchmark_raw_result.csv", index=False)

    df_summary = compute_summary(df_results)
    print(f"Summary computed with {len(df_summary)} summary rows.")
    df_summary.to_csv("results/ci_benchmark_summaries.csv", index=False)
    print(df_summary.head())
    print(
        "\nDetailed results and summary saved to ci_benchmark_raw_result.csv and ci_benchmark_summaries.csv"
    )

    print("Generating plots...")
    plot_benchmarks(df_summary)
    print("Plots generated successfully.")
    
    # Making a copy of the result to the web directory
    web_results_dir = os.path.join("web", "results")
    os.makedirs(web_results_dir, exist_ok=True)
    src = os.path.join("results", "ci_benchmark_summaries.csv")
    dst = os.path.join(web_results_dir, "ci_benchmark_summaries.csv")
    shutil.copyfile(src, dst)
    print(f"Copied summary CSV to {dst} for web UI.")