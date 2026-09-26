# proximitylib/proximity.py

# =============================
# Imports
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from fractions import Fraction
from collections import Counter
from functools import reduce
from math import gcd
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from tabulate import tabulate
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

# =============================
# 0) Parameter sets
# =============================

param_sets = {
    1: dict(
    sharpness_base=2.0, sharpness_growth=0.0, decay=0.0,
    max_freq=1, r_max=1, weight_exponent=0.0, freq_sharpness_exp=0.0,
    threshold=0.01
),
    2: dict(
    sharpness_base=6.0, sharpness_growth=0.0, decay=0.0,
    max_freq=1, r_max=5, weight_exponent=0.0, freq_sharpness_exp=0.0,
    threshold=0.01
),
    3: dict(
    sharpness_base=10.0, sharpness_growth=0.0, decay=0.0,
    max_freq=2, r_max=5, weight_exponent=0.0, freq_sharpness_exp=1.0,
    threshold=0.01
)
}

# =============================
# 1) Proximity-Function
# =============================

def proximity_max_freq_gaussian(
    x,
    sharpness_base=10.0,
    sharpness_growth=0.0,
    decay=0.0,
    max_freq=2,
    r_max=None,
    weight_exponent=0.0,
    freq_sharpness_exp=0.0,
    threshold=0.01,
):
    x = np.array(x, dtype=float)
    if r_max is None:
        r_max = int(np.ceil(np.max(x)))
    N = len(x)
    result = np.zeros(N)
    best_frac = [(0, 0)] * N

    for f in range(1, max_freq + 1):
        k_vals = np.arange(f, int(np.floor(r_max * f)) + 1)
        mu_all = k_vals / f
        sharpness_f = sharpness_base * (f ** freq_sharpness_exp)
        sigma_all = 1.0 / (sharpness_f * (k_vals ** sharpness_growth) + 1e-9)
        weight_f = 1.0 / (f ** weight_exponent)
        mu_decay_all = 1.0 / (mu_all ** decay)

        diff = x[:, None] - mu_all[None, :]
        g = weight_f * mu_decay_all[None, :] * np.exp(-0.5 * (diff / sigma_all[None, :])**2)
        best_idx = np.argmax(g, axis=1)
        best_val = g[np.arange(N), best_idx]
        update_idx = best_val > result
        result[update_idx] = best_val[update_idx]
        for idx in np.where(update_idx)[0]:
            best_frac[idx] = (int(k_vals[best_idx[idx]]), f)

    if result.max() > 0:
        result /= result.max()
    result[result < threshold] = threshold
    return result, best_frac

def generate_null_sequences(iois, num_sequences=1000, models=("expon", "uniform", "resampled"), seed=42):
    """
    Generates null sequences based on the given IOIs and selected null models.

    Parameters
    ----------
    iois : array-like
        Input-Inter-Onset-Interval-Sequence.
    num_sequences : int
        Number of the null sequences to generate per model.
    models : tuple of str
        List of desired null models. Possible values:
        ("expon", "uniform", "resampled")
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with model names as keys and generated null sequences
        as values (shape: num_sequences x len(iois)).
    """
    rng = np.random.default_rng(seed)
    iois = np.asarray(iois, dtype=float)
    n = len(iois)
    max_ioi, mean_ioi = np.max(iois), np.mean(iois)

    null_sequences = {}
    for model in models:
        if model == "expon":
            sims = rng.exponential(scale=mean_ioi, size=(num_sequences, n))
        elif model == "uniform":
            sims = rng.uniform(0.01, max_ioi, size=(num_sequences, n))
        elif model == "resampled":
            sims = rng.choice(iois, size=(num_sequences, n), replace=True)
        else:
            raise ValueError(f"Unknown null model: {model}")
        null_sequences[model] = sims

    return null_sequences

# =============================
# A) FUNCTIONS FOR SINGLE SEQUENCE ANALYSIS 
# (Called from catalogue.py)
# =============================

def build_pairwise_proximity_df(iois, params, threshold=0.01):
    n = len(iois)
    
    # --- Ratio-Matrix ---
    ratios = np.ones((n, n), dtype=float)
    i_idx, j_idx = np.triu_indices(n, k=1)
    ratios[i_idx, j_idx] = np.maximum(iois[i_idx], iois[j_idx]) / np.minimum(iois[i_idx], iois[j_idx])
    ratios[j_idx, i_idx] = ratios[i_idx, j_idx]
    
    # --- Proximity-Matrix ---
    ratios_flat = ratios.flatten()
    prox_flat, best_frac_flat = proximity_max_freq_gaussian(ratios_flat, **params)
    prox_mat = prox_flat.reshape(n, n)
    
    # --- Peak-Labels ---
    peak_labels = np.full((n, n), "No Proximity", dtype=object)
    for idx, val in enumerate(prox_flat):
        if val > threshold:
            i, j = divmod(idx, n)
            k_val, f_val = best_frac_flat[idx]
            frac = Fraction(k_val, f_val).limit_denominator()
            peak_labels[i, j] = f"{frac.numerator}:{frac.denominator}"

    # --- DataFrame ---
    df_pairs = pd.DataFrame({
        "i": np.repeat(np.arange(n), n),
        "j": np.tile(np.arange(n), n),
        "ioi_i": np.repeat(iois, n),
        "ioi_j": np.tile(iois, n),
        "ratio": ratios.flatten(),
        "prox_value": prox_flat,
        "peak_label": peak_labels.flatten(),
        "distance": np.abs(np.repeat(np.arange(n), n) - np.tile(np.arange(n), n)),
    })
    
    return df_pairs, ratios, prox_mat

def analyze_single_sequence(iois, params, num_sequences=1000, models=("expon", "uniform", "resampled"),
                             seed=42, bins=50, plot=True):
    """
    Assesses a single IOI sequence against selected null models
    (e.g., Exponential, Uniform, Resampled) and creates histogram with
    mean, 95% quantiles, z-score, and p-value.
    """
    rng = np.random.default_rng(seed)
    iois = np.asarray(iois, dtype=float)
    sequence_length = len(iois)
    if sequence_length < 2:
        raise ValueError("IOI sequence must contain at least two values.")

    # -----------------------------
    # Real Proximity-Scores
    # -----------------------------
    ratios_real = np.maximum(iois[:, None], iois[None, :]) / np.minimum(iois[:, None], iois[None, :])
    prox_mat_real, _ = proximity_max_freq_gaussian(ratios_real.flatten(), **params)
    prox_mat_real = prox_mat_real.reshape(ratios_real.shape)
    triu_idx = np.triu_indices_from(prox_mat_real, k=1)
    proximity_real = prox_mat_real[triu_idx]
    mean_real = np.mean(proximity_real)

    # -----------------------------
    # Compute Null Proximity-Scores
    # -----------------------------
    null_sequences = generate_null_sequences(iois, num_sequences=num_sequences,
                                             models=models, seed=seed)

    stats = {}
    proximities = {}
    mean_values = {}

    for model, sims in null_sequences.items():
        model_means = []
        all_vals = []
        for sim_iois in sims:
            ratios = np.maximum(sim_iois[:, None], sim_iois[None, :]) / np.minimum(sim_iois[:, None], sim_iois[None, :])
            prox_mat, _ = proximity_max_freq_gaussian(ratios.flatten(), **params)
            prox_mat = prox_mat.reshape(ratios.shape)
            vals = prox_mat[triu_idx]
            all_vals.append(vals)
            model_means.append(np.mean(vals))
        all_vals = np.concatenate(all_vals)
        proximities[model] = all_vals
        mean_values[model] = np.array(model_means)

        mean_null = np.mean(model_means)
        std_null = np.std(model_means, ddof=1)
        zscore = (mean_real - mean_null) / (std_null + 1e-12)
        pval = (np.sum(mean_values[model] >= mean_real) + 1) / (num_sequences + 1)
        quant95 = np.percentile(mean_values[model], 95)

        stats[model] = dict(
            mean_null=mean_null,
            std_null=std_null,
            zscore=zscore,
            pval=pval,
            quant95=quant95,
        )

    # --- real distance correlation ---
    df_pairs, ratios_real, prox_mat_real = build_pairwise_proximity_df(iois, params)
    triu_idx = np.triu_indices_from(prox_mat_real, k=1)
    proximity_real = prox_mat_real[triu_idx]
    mean_real = np.mean(proximity_real)

    # Calculate distance correlation for real data
    df_pairs["distance"] = np.abs(df_pairs["i"] - df_pairs["j"])
    valid = df_pairs["i"] != df_pairs["j"]  # only i != j
    if valid.sum() > 0:
        distance_corr_real, distance_corr_real_p = pearsonr(
            df_pairs.loc[valid, "distance"], df_pairs.loc[valid, "prox_value"]
        )
    else:
        distance_corr_real, distance_corr_real_p = np.nan, np.nan

    # --- null distance correlations ---
    distance_corr_nulls = {}
    for model, sims in null_sequences.items():
        model_corrs = []
        for sim_iois in sims:
            df_null_pairs, _, prox_mat_null = build_pairwise_proximity_df(sim_iois, params)
            df_null_pairs["distance"] = np.abs(df_null_pairs["i"] - df_null_pairs["j"])
            valid_null = df_null_pairs["i"] != df_null_pairs["j"]
            if valid_null.sum() > 0:
                corr, _ = pearsonr(
                    df_null_pairs.loc[valid_null, "distance"],
                    df_null_pairs.loc[valid_null, "prox_value"]
                )
                model_corrs.append(corr)
            else:
                model_corrs.append(np.nan)
        distance_corr_nulls[model] = np.array(model_corrs)

    # -----------------------------
    # Plot (optional)
    # -----------------------------
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(12, 6))
        colors = {"Real": "blue", "expon": "orange", "uniform": "gray", "resampled": "green"}
        all_nulls = np.concatenate([v for v in mean_values.values()])
        bin_edges = np.linspace(np.min(all_nulls), np.max(all_nulls), bins + 1)

        plt.hist(proximity_real, bins=bin_edges, density=True, alpha=0.5, color=colors["Real"], label="Real")
        sns.kdeplot(proximity_real, bw_adjust=0.5, color=colors["Real"], linewidth=2)

        for model in models:
            plt.hist(mean_values[model], bins=bin_edges, density=True, alpha=0.3, color=colors[model],
                     label=f"{model.capitalize()} Null (mean)")
            sns.kdeplot(mean_values[model], bw_adjust=0.5, color=colors[model], linewidth=2)

            plt.axvline(stats[model]["mean_null"], color=colors[model], linestyle="--", linewidth=2,
                        label=f"{model} mean ≈ {stats[model]['mean_null']:.3f}")
            plt.axvline(stats[model]["quant95"], color=colors[model], linestyle=":", linewidth=2,
                        label=f"{model} 95%-quantile ≈ {stats[model]['quant95']:.3f}")

        plt.axvline(mean_real, color=colors["Real"], linestyle="--", linewidth=2,
                    label=f"real mean ≈ {mean_real:.3f}")
        plt.xlabel("Proximity-Scores")
        plt.ylabel("Density")
        plt.title("Proximity-Scores: Real vs Null Models", fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()

        print("\n=== Results ===")
        for model, vals in stats.items():
            print(f"\n{model.upper()}:")
            for k, v in vals.items():
                print(f"  {k}: {v:.4f}")

    # -----------------------------
    # Return results
    # -----------------------------
    return {
        "proximity_real": proximity_real,
        "null_proximities": proximities,
        "null_means": mean_values,
        "distance_corr_real": distance_corr_real,
        "distance_corr_real_p": distance_corr_real_p,
        "distance_corr_nulls": distance_corr_nulls,
        "results": {"mean_real": mean_real, **stats},
    }

def new_plot_sequence_analysis(
    filename,
    base_dir,
    param_choice=1,
    null_models=("expon", "uniform", "resampled"),
    num_sequences=200,
    seed=42,
    bins=25,
    top_k=5,
    figsize=(20, 28),
    only_page=False,
    safe_fig=False,
    save_dir=None,
):
    """
    Creates a comprehensive figure analyzing a single IOI sequence from a CSV file.
    The figure includes:
    - Proximity function with IOI ratios
    - Distribution of IOI ratios
    - Proximity matrix (recurrence plot)
    - Histograms comparing real proximity scores to null models
    - Scatter plot of distance vs. proximity (real data)
    - Correlation of distance and proximity (real vs. null models)

    Parameters
    ----------
    filename : str
        Name of the CSV file containing IOI data.
    base_dir : str
        Base directory where the CSV file is located.
    param_choice : int
        Choice of parameter set (1, 2, or 3).
    null_models : tuple of str
        Null models to use for comparison (e.g., ("expon", "uniform", "resampled")).
    num_sequences : int
        Number of null sequences to generate per model.
    seed : int
        Random seed for reproducibility.
    bins : int
        Number of bins for histograms.
    top_k : int
        Number of top beat candidates to display.
    figsize : tuple
        Figure size.
    only_page : bool
        If True, only returns the figure object without displaying or saving.
    safe_fig : bool
        If True, saves the figure to the specified save_dir.
    save_dir : str or None
        Directory to save the figure if safe_fig is True.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.gridspec as gridspec
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    # === Load Parameters and File ===
    path = Path(base_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)
    iois = df["IOI"].dropna().values
    onsets = df["onset"].dropna().values

    params = param_sets[param_choice]
    print(f"\nUsing parameter dict: default_params_{param_choice}")

    # === Pairwise Proximity ===
    df_pairs, ratios, prox_max = build_pairwise_proximity_df(iois, params)
    x_min = max(1.0, df_pairs["ratio"].min() * 0.95)
    x_max = df_pairs["ratio"].max() * 1.05
    x_range = np.linspace(x_min, x_max, 2000)
    y_curve, _ = proximity_max_freq_gaussian(x_range, **params)
    mean_prox = np.mean(y_curve)

    pivot_vals = df_pairs.pivot(index="i", columns="j", values="prox_value") * 100
    pivot_labels = df_pairs.pivot(index="i", columns="j", values="peak_label")

    # === Color peaks ===
    unique_labels = sorted(set(df_pairs["peak_label"]))
    tab10 = plt.cm.tab10.colors
    fixed_colors = {
        "1:1": tab10[0], "2:1": tab10[1], "3:1": tab10[2], "4:1": tab10[3], "5:1": tab10[4],
        "3:2": tab10[6], "5:2": tab10[9], "7:2": tab10[7], "9:2": tab10[8], 
        "4:3": tab10[5], "5:3": (0.5, 0.0, 0.5)
    }

    label_to_cmap = {}
    for label, color in fixed_colors.items():
        if label in unique_labels:
            cmap_colors = [(1, 1, 1), plt.cm.colors.to_rgb(color)]
            label_to_cmap[label] = plt.cm.colors.LinearSegmentedColormap.from_list(f"{label}_cmap", cmap_colors)
    remaining_labels = [lab for lab in unique_labels if lab not in label_to_cmap]
    palette = sns.color_palette("Set2", len(remaining_labels))
    for label, base_color in zip(remaining_labels, palette):
        cmap_colors = [(1, 1, 1), base_color]
        label_to_cmap[label] = plt.cm.colors.LinearSegmentedColormap.from_list(f"{label}_cmap", cmap_colors)

    # === Nullmodel Analysis ===
    results = analyze_single_sequence(
        iois,
        params=params,
        num_sequences=num_sequences,
        seed=seed,
        bins=bins,
        plot=False,
        models=null_models
    )

    # === Figure ===
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(7, 2, figure=fig, width_ratios=[1, 1],
                           height_ratios=[0.5, 1, 1, 1, 1, 1, 1], hspace=0.5, wspace=0.3)

    # === Header ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.text(0.01, 1.0, f"IOI-Analyse:\n{filename}", transform=ax_header.transAxes,
                   fontsize=22, fontweight='bold', va='top', ha='left')

    res = results["results"]
    info_lines = [
        f"Number of IOIs: {len(iois)} | min: {np.min(iois):.3f}, max: {np.max(iois):.3f}, Ø: {np.mean(iois):.3f}",
        f"PAIR_THRESHOLD: {params['threshold']}",
        f"Parameters: sharpness_base={params['sharpness_base']}, sharpness_growth={params['sharpness_growth']}, decay={params['decay']}",
        f"max_freq={params['max_freq']}, weight_exponent={params['weight_exponent']}, freq_sharpness_exp={params['freq_sharpness_exp']}",
    ]
    info_lines.append(f"Mean Real: {res['mean_real']:.3f}")
    for model, vals in res.items():
        if isinstance(vals, dict) and "mean_null" in vals:
            info_lines.append(
                f"{model.capitalize()} → Mean: {vals['mean_null']:.3f} | "
                f"Std: {vals['std_null']:.3f} | Z: {vals['zscore']:.3f} | "
                f"p: {vals['pval']:.3f} | Q95: {vals['quant95']:.3f}"
            )
    # Distance-Proximity Correlation
    dcorr_real = results.get("distance_corr_real", np.nan)
    dcorr_p = results.get("distance_corr_real_p", np.nan)
    info_lines.append(f"Distance-Proximity (Real): r = {dcorr_real:.3f}, p = {dcorr_p:.3f}")

    ax_header.text(0.99, 1.0, "\n".join(info_lines),
                   transform=ax_header.transAxes, fontsize=11, va='top', ha='right')

    # === Proximity-Function ===
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(x_range, y_curve, label=f"Proximity (mean ≈ {mean_prox:.3f})", alpha=0.5, color="grey")
    ax1.scatter(df_pairs["ratio"], df_pairs["prox_value"], s=10, alpha=0.7, color="red", label="IOI-Ratios")
    ax1.set_xlim(x_min, x_max)
    ax1.set_title("Proximity-Function with IOI Ratios\n", fontweight='bold')
    ax1.set_xlabel("Ratio")
    ax1.set_ylabel("Proximity")
    ax1.grid(True)
    ax1.legend()

    # === IOI-Ratio Distribution ===
    ax2 = fig.add_subplot(gs[2, :])
    sns.kdeplot(df_pairs["ratio"], fill=True, color="green", alpha=0.5, lw=2, bw_adjust=0.1, ax=ax2)
    sns.histplot(df_pairs["ratio"], bins=50, color="gray", alpha=0.3, stat="density", ax=ax2)
    ax2.set_title("Distribution of IOI Ratios\n", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Ratio (IOI_i / IOI_j)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(x_min, x_max)
    ax2.grid(True)

    # === Proximity-Matrix (Recurrence) ===
    ax3 = fig.add_subplot(gs[3:5, 0])
    n = len(iois)
    for (i, j), val in np.ndenumerate(pivot_vals.values):
        label = pivot_labels.iloc[i, j]
        cmap = label_to_cmap[label]
        color = cmap(val / 100) if not np.isnan(val) else (1, 1, 1, 1)
        ax3.add_patch(plt.Rectangle([j, i], 1, 1, facecolor=color, edgecolor='black'))
        if n <= 20 and not np.isnan(val):
            ax3.text(j+0.5, i+0.5, f"{float(val.item()):.1f}", ha='center', va='center', fontsize=6, color="black")

    ax3.set_xlim(0, n)
    ax3.set_ylim(0, n)
    ax3.set_aspect('equal')
    ax3.set_xticks(np.arange(n)+0.5)
    ax3.set_yticks(np.arange(n)+0.5)
    ax3.set_xticklabels(range(n), rotation=45, ha="right")
    ax3.set_yticklabels(range(n))
    ax3.set_xlabel("j")
    ax3.set_ylabel("i")
    ax3.set_title("Proximity-Matrix of IOI Ratios\n", fontweight='bold')

    # === Colorbars for Peak Labels ===
    pos = ax3.get_position()
    legend_labels = [lab for lab in unique_labels if lab != "No Proximity"]
    cbar_width = 0.05
    spacing = 0.01
    n_labels = len(legend_labels)
    bar_height = (pos.height - (n_labels-1)*spacing) / n_labels
    for idx, label in enumerate(legend_labels):
        cmap = label_to_cmap[label]
        norm = plt.Normalize(vmin=0, vmax=100)
        cbar_y = pos.y0 + pos.height - (idx+1)*bar_height - idx*spacing
        cbar_ax = fig.add_axes([pos.x1 + 0.01, cbar_y, cbar_width, bar_height])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation="horizontal")
        cb.set_ticks([0, 50, 100])
        cb.ax.tick_params(labelsize=9)
        fig.text(pos.x1 + 0.07, cbar_y + bar_height/2, label, va='center', fontsize=10)

    # === Histogramm ===
    ax4 = fig.add_subplot(gs[3:5, 1])
    colors = {"Real": "blue", "expon": "orange", "uniform": "gray", "resampled": "green"}
    bin_edges = np.linspace(0, 1, bins)

    ax4.hist(results["proximity_real"], bins=bin_edges, density=True, alpha=0.5, color="blue", label="Real")
    sns.kdeplot(results["proximity_real"], bw_adjust=0.4, color="blue", linewidth=2, ax=ax4)

    for model, prox_vals in results["null_proximities"].items():
        color = colors.get(model, "black")
        ax4.hist(prox_vals, bins=bin_edges, density=True, alpha=0.3, color=color, label=f"{model.capitalize()} Null")
        sns.kdeplot(prox_vals, bw_adjust=0.4, color=color, linewidth=2, ax=ax4)
        if model in res:
            mstats = res[model]
            ax4.axvline(mstats["mean_null"], color=color, linestyle="--", linewidth=2)
            ax4.axvline(mstats["quant95"], color=color, linestyle=":", linewidth=2)
    ax4.axvline(res["mean_real"], color="blue", linestyle="--", linewidth=2)
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Proximity-Scores")
    ax4.set_ylabel("Density")
    ax4.set_title("Histogram: Real vs Nullmodels\n", fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # === Scatterplot: Distance vs. Proximity (Real) ===
    ax_scatter = fig.add_subplot(gs[5:7, 0])
    if "distance" in df_pairs.columns:
        sns.scatterplot(x=df_pairs["distance"], y=df_pairs["prox_value"], alpha=0.4, color='purple', ax=ax_scatter)
        sns.regplot(x=df_pairs["distance"], y=df_pairs["prox_value"], scatter=False, color='red', ax=ax_scatter)
        ax_scatter.set_xlabel("Distance (Δ Index)")
        ax_scatter.set_ylabel("Proximity")
        ax_scatter.set_title("Real: Proximity vs. Distance of IOIs\n", fontweight='bold')
        ax_scatter.grid(True)

    # === Correlation: Distance vs. Proximity (Real) ===
    ax_corr = fig.add_subplot(gs[5:7, 1])
    colors_corr = {"expon": "orange", "uniform": "gray", "resampled": "green"}
    for model, corr_vals in results["distance_corr_nulls"].items():
        color = colors_corr.get(model, "black")
        sns.histplot(corr_vals, bins=20, color=color, alpha=0.4, label=f"{model.capitalize()} Null", ax=ax_corr)
        sns.kdeplot(corr_vals, color=color, linewidth=2, ax=ax_corr)
    ax_corr.axvline(results["distance_corr_real"], color="blue", linestyle="--", linewidth=2,
                    label=f"Real r = {results['distance_corr_real']:.3f}")
    ax_corr.set_xlabel("Distance-Proximity Correlation (r)")
    ax_corr.set_ylabel("Density")
    ax_corr.set_title("Distance-Proximity Correlation: Real vs Nullmodels\n", fontweight='bold')
    ax_corr.legend()
    ax_corr.grid(True, alpha=0.3)

    plt.tight_layout()

    # === Save Figure ===
    if safe_fig:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # filename: e. g. heterochrony-1-2_jitter-0050_07_default_params_2.png
        safe_name = Path(filename).stem.replace(" ", "_")
        save_name = f"{safe_name}_params_{param_choice}.png"
        save_path = save_dir / save_name

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path.resolve()}")

    plt.show()

    if not only_page:
        return {"iois": iois, "onsets": onsets, "results": results, "params": params}

# =============================
# B) FUNCTIONS FOR DATASET/SEQUENCE RHYTHMICITY ANALYSIS 
# (Called from peak_tables.ipynb)
# =============================

def compute_peak_significance_table(iois, params, num_null=200,
                                    models=("expon", "uniform", "resampled"),
                                    seed=42, source="", alpha=0.05):
    """
    Identifies significant proximity peaks in an IOI sequence by comparing real proximity scores
    against null models. Returns a DataFrame summarizing significant peaks.
    """
    rng = np.random.default_rng(seed)
    iois = np.asarray(iois, dtype=float)
    n = len(iois)
    if n < 2:
        return pd.DataFrame(), {}

    triu_idx = np.triu_indices(n, k=1)
    ratios_real = np.maximum(iois[:, None], iois[None, :]) / np.minimum(iois[:, None], iois[None, :])
    ratios_real = ratios_real[triu_idx]
    prox_real, best_frac_real = proximity_max_freq_gaussian(ratios_real, **params)
    threshold = params.get("threshold", 0.01)

    peak_labels = [f"{k}:{f}" if (k,f)!=(0,0) else "No Proximity" for k,f in best_frac_real]
    peaks = sorted(set([p for p in peak_labels if p != "No Proximity"]))

    peak_real_scores = {
        peak: np.array([prox_real[i] if peak_labels[i]==peak else threshold for i in range(len(prox_real))])
        for peak in peaks
    }

    # ===== Nullmodels =====
    null_sequences = generate_null_sequences(iois, num_sequences=num_null, models=models, seed=seed)
    null_scores = {model: {peak: [] for peak in peaks} for model in models}

    for model, sims in null_sequences.items():
        for sim_iois in sims:
            ratios_null = np.maximum(sim_iois[:, None], sim_iois[None, :]) / np.minimum(sim_iois[:, None], sim_iois[None, :])
            ratios_null = ratios_null[triu_idx]
            prox_null, best_frac_null = proximity_max_freq_gaussian(ratios_null, **params)
            peak_labels_null = [f"{k}:{f}" if (k,f)!=(0,0) else "No Proximity" for k,f in best_frac_null]
            for peak in peaks:
                vals = np.array([prox_null[i] if peak_labels_null[i]==peak else threshold for i in range(len(prox_null))])
                null_scores[model][peak].append(np.mean(vals))

    # ===== Results =====
    results = []
    for peak in peaks:
        peak_data = {"peak": peak}
        sig_models = []
        for model in models:
            pvals = np.array(null_scores[model][peak])
            mean_null = np.mean(pvals)
            pval = (np.sum(pvals >= np.mean(peak_real_scores[peak])) + 1) / (num_null + 1)
            peak_data[f"{model}_mean_null"] = mean_null
            peak_data[f"{model}_pval"] = pval
            if pval < alpha:
                sig_models.append(model)

        peak_data[f"significance_alpha: {alpha}"] = ", ".join(sig_models) if sig_models else "n.s."
        peak_data["mean_real"] = np.mean(peak_real_scores[peak])
        peak_data["level"] = "sequence"
        peak_data["source"] = source
        results.append(peak_data)

    df = pd.DataFrame(results).sort_values("mean_real", ascending=False).reset_index(drop=True)
    col_order = ["peak", f"significance_alpha: {alpha}", "level", "source"] + \
                 [c for c in df.columns if c not in ["peak", f"significance_alpha: {alpha}", "level", "source"]]

    # ===== Distance-Proximity Correlation =====
    from scipy.stats import pearsonr
    distances = np.abs(triu_idx[0] - triu_idx[1])
    valid = distances != 0
    if valid.sum() > 1 and np.std(prox_real[valid]) > 1e-12:
        distance_corr_real, distance_corr_real_p = pearsonr(distances[valid], prox_real[valid])
    else:
        distance_corr_real, distance_corr_real_p = np.nan, np.nan

    distance_corr_nulls = {model: [] for model in models}
    for model, sims in null_sequences.items():
        for sim_iois in sims:
            ratios_null = np.maximum(sim_iois[:, None], sim_iois[None, :]) / np.minimum(sim_iois[:, None], sim_iois[None, :])
            ratios_null = ratios_null[triu_idx]
            prox_null, _ = proximity_max_freq_gaussian(ratios_null, **params)
            if valid.sum() > 1 and np.std(prox_null[valid]) > 1e-12:
                corr_null, _ = pearsonr(distances[valid], prox_null[valid])
            else:
                corr_null = np.nan
            distance_corr_nulls[model].append(corr_null)

    # --- p-values for dist_corr ---
    distance_corr_dict = {
        "distance_corr_real": float(distance_corr_real),
        "distance_corr_real_p": float(distance_corr_real_p)
    }
    for model in models:
        null_corrs = np.array(distance_corr_nulls[model])
        if np.isnan(distance_corr_real):
            pval_corr = np.nan
        else:
            median_null = np.nanmedian(null_corrs)
            if distance_corr_real >= median_null:
                pval_corr = (np.sum(null_corrs >= distance_corr_real) + 1) / (len(null_corrs) + 1)
            else:
                pval_corr = (np.sum(null_corrs <= distance_corr_real) + 1) / (len(null_corrs) + 1)
        distance_corr_dict[f"distance_corr_{model}_p"] = float(pval_corr)

    return df[col_order], distance_corr_dict

def analyze_dataset_peaks_table(path, params, num_null=200, alpha=0.05, models=("expon", "uniform", "resampled"), seed=42):
    path = Path(path)
    """Analyzes a dataset of IOIs and identifies significant proximity peaks."""

    if path.is_file():
        df = pd.read_csv(path)
        iois = df["IOI"].dropna().values
        return compute_peak_significance_table(iois, params, num_null=num_null, source=path.name, alpha=alpha, models=models, seed=seed)
    elif path.is_dir():
        all_peak_dfs = []
        all_distance_corr_dicts = []

        for file in path.glob("*.csv"):
            df = pd.read_csv(file)
            iois = df["IOI"].dropna().values
            peak_df, dist_dict = compute_peak_significance_table(iois, params, num_null=num_null,
                                                                source=file.name, alpha=alpha, models=models, seed=seed)
            if not peak_df.empty:
                all_peak_dfs.append(peak_df)
                all_distance_corr_dicts.append(dist_dict)

        if not all_peak_dfs:
            return pd.DataFrame()

        df_concat = pd.concat(all_peak_dfs, ignore_index=True)
        # --- Average distance-corr values ---
        distance_corr_keys = ["distance_corr_real", "distance_corr_real_p",
                            "distance_corr_expon_p", "distance_corr_uniform_p",
                            "distance_corr_resampled_p"]

        distance_corr_means = {}
        for key in distance_corr_keys:
            vals = [d[key] for d in all_distance_corr_dicts if key in d]
            distance_corr_means[key] = float(np.nanmean(vals)) if vals else np.nan

        grouped = df_concat.groupby("peak")
        final_rows = []
        for peak, g in grouped:
            row = {
                "peak": peak,
                f"significance_alpha: {alpha}": ", ".join([model for model in ["expon","uniform", "resampled"]
                                                    if np.sum(g[f"{model}_pval"] < alpha) > len(g)/2]),
                "mean_real": g["mean_real"].mean(),
                "expon_mean_null": g["expon_mean_null"].mean(),
                "expon_pval": g["expon_pval"].mean(),
                "uniform_mean_null": g["uniform_mean_null"].mean(),
                "uniform_pval": g["uniform_pval"].mean(),
                "resampled_mean_null": g["resampled_mean_null"].mean(),
                "resampled_pval": g["resampled_pval"].mean(),
                "level": "dataset",
                "source": path.name
            }
            if row[f"significance_alpha: {alpha}"] == "":
                row[f"significance_alpha: {alpha}"] = "n.s."
            final_rows.append(row)

        df = pd.DataFrame(final_rows).sort_values("mean_real", ascending=False)
        col_order = ["peak", f"significance_alpha: {alpha}", "level", "source"] + \
            [c for c in df.columns if c not in ["peak", f"significance_alpha: {alpha}", "level", "source"]]

        return df[col_order], distance_corr_means
    else:
        raise ValueError(f"{path} is neither file nor folder!")

def summarize_results_table(df, distance_corr_dict=None, alpha=0.05):
    """
    Summarizes the peak significance results into a readable table format.
    """

    summaries = []

    for _, row in df.iterrows():
        filename = str(row["source"])
        dtype = str(row["level"])

        # --- Classification of peak ---
        peak = row["peak"]
        sig_models = []
        for model in ["expon", "uniform", "resampled"]:
            pval = row.get(f"{model}_pval", np.nan)
            if not pd.isna(pval) and pval < alpha:
                sig_models.append(f"True: {peak}: {model} ({pval:.4f})")
        peak_sig_text = "\n".join(sig_models) if sig_models else "False"

        # --- Determine category ---
        if peak == "1:1":
            category = "Isochrony"
        elif peak.endswith(":1") and not peak.startswith("1:"):
            category = "Simple Heterochrony"
        else:
            category = "Complex Heterochrony"

        summaries.append({
            "Filename": filename,
            "Type": dtype,
            "Isochrony": "False",
            "Simple Heterochrony": "False",
            "Complex Heterochrony": "False",
            category: peak_sig_text
        })

    df_summary = pd.DataFrame(summaries)

    # -- Distance-Proximity Correlation ---
    if distance_corr_dict is not None:
        r_real = distance_corr_dict.get("distance_corr_real", np.nan)
        r_classic_p = distance_corr_dict.get("distance_corr_real_p", np.nan)
        corr_col_name = f"Distance-Proximity Correlation\n(pearson r={r_real:.3f}, classical p={r_classic_p:.3f})"

        # test significance
        corr_sig_texts = []
        for model in ["expon", "uniform", "resampled"]:
            pval = distance_corr_dict.get(f"distance_corr_{model}_p", np.nan)
            if not pd.isna(pval) and pval < alpha:
                corr_sig_texts.append(f"True: {model} (p={pval:.4f})")
        corr_text = "\n".join(corr_sig_texts) if corr_sig_texts else "False"

        df_summary[corr_col_name] = corr_text
    else:
        corr_col_name = "Distance-Proximity correlation"
        df_summary[corr_col_name] = "False"

    # --- Merge entries per file and type ---
    def merge_vals(series):
        vals = [str(v).strip() for v in series if str(v) != "False" and pd.notna(v)]
        unique_vals = sorted(set(vals))
        return "\n".join(unique_vals) if unique_vals else "False"

    df_summary = df_summary.groupby(["Filename", "Type"], as_index=False).agg({
        "Isochrony": merge_vals,
        "Simple Heterochrony": merge_vals,
        "Complex Heterochrony": merge_vals,
        corr_col_name: merge_vals
    })

    # --- Final column order ---
    return df_summary[["Filename", "Type", "Isochrony", "Simple Heterochrony", "Complex Heterochrony", corr_col_name]]

def print_peak_table(df, title=None):
    """
    Prints the peak significance DataFrame in a formatted table.
    """
    from tabulate import tabulate

    if title:
        print(f"\n{title}\n" + "=" * len(title))
    
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: print_peak_table expects a DataFrame, got:", type(df))
        return

    # Format float columns
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()

    # Print the table with formatted floats
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False,
            floatfmt=".4f"  # Format floats to 4 decimal places
        )
    )

# =============================
# C) EXPLORATORY PLOTTING-WIDGET OF PROXIMITY FUNCTION
# (Called from function_explorer.ipynb)
# =============================

def plot_proximity_max_freq_gaussian(
    sharpness_base=12.0,
    sharpness_growth=0.5,
    decay=0.2,
    max_freq=3,
    r_max=4.0,
    weight_exponent=0.5,
    freq_sharpness_exp=0.5,
):
    x_vals = np.linspace(1, r_max + 0.5, 2000)
    y_vals, _ = proximity_max_freq_gaussian(
        x_vals,
        sharpness_base=sharpness_base,
        sharpness_growth=sharpness_growth,
        decay=decay,
        max_freq=max_freq,
        r_max=r_max,
        weight_exponent=weight_exponent,
        freq_sharpness_exp=freq_sharpness_exp,
    )

    mean_prox = simpson(y_vals, x=x_vals) / (x_vals[-1] - x_vals[0])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Optimized Gaussian Proximity Function", fontsize=16, y=1.02)

    axes[0].plot(x_vals, y_vals, label=f"Proximity (∫mean ≈ {mean_prox:.3f})")
    axes[0].axhline(mean_prox, color='red', linestyle='--', linewidth=2, label=f"∫mean ≈ {mean_prox:.3f}")
    axes[0].set_xlabel("Ratio x")
    axes[0].set_ylabel("Proximity")
    axes[0].set_title("Proximity Function")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    counts, _, _ = axes[1].hist(y_vals, bins=50, range=(0, 1), alpha=0.4, color="skyblue", density=True)
    try:
        kde = gaussian_kde(y_vals)
        x_dens = np.linspace(0, 1, 5000)
        kde_vals = kde(x_dens)
        scaling_factor = max(counts) / max(kde_vals)
        axes[1].plot(x_dens, kde_vals * scaling_factor, color="darkblue", lw=2, label="Density (KDE)")
    except np.linalg.LinAlgError:
        print("Warning: KDE computation failed due to singular data covariance.")

    axes[1].axvline(mean_prox, color='red', linestyle='--', linewidth=2, label=f"∫mean ≈ {mean_prox:.3f}")
    axes[1].set_title("Proximity Distribution")
    axes[1].set_xlabel("Proximity")
    axes[1].set_ylabel("Normalized frequency")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()