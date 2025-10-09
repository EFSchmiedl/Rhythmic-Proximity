# proximity.py (die libary)

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
from scipy.stats import gaussian_kde

# =============================
# 0) Parametersets
# =============================

param_sets = {
    "iso": dict(
    sharpness_base=6.0, sharpness_growth=0.0, decay=0.0,
    max_freq=1, x_max=1, weight_exponent=0.0, freq_sharpness_exp=0.0,
    output_max=1.0, threshold=0.01
),
    1: dict(
    sharpness_base=6.0, sharpness_growth=0.1, decay=0.1,
    max_freq=1, x_max=None, weight_exponent=0.0, freq_sharpness_exp=0.0,
    output_max=1, threshold=0.01
),
    2: dict(
    sharpness_base=10.0, sharpness_growth=0.1, decay=0.1,
    max_freq=2, x_max=None, weight_exponent=0.0, freq_sharpness_exp=1.0,
    output_max=1, threshold=0.01
),
    3: dict(
    sharpness_base=14.0, sharpness_growth=0.1, decay=0.1,
    max_freq=3, x_max=None, weight_exponent=0.0, freq_sharpness_exp=1.2,
    output_max=1, threshold=0.01
),
    4: dict(
    sharpness_base=14.0, sharpness_growth=0.1, decay=0.1,
    max_freq=4, x_max=None, weight_exponent=0.0, freq_sharpness_exp=1.5,
    output_max=1, threshold=0.01
)
}

# =============================

def proximity_max_freq_gaussian_table(
    x,
    sharpness_base=10.0,
    sharpness_growth=1.0,
    decay=0.1,
    max_freq=4,
    x_max=None,
    weight_exponent=1.0,
    freq_sharpness_exp=1.0,
    output_max=1.0,
    threshold=0.01,
):
    x = np.array(x, dtype=float)
    if x_max is None:
        x_max = int(np.ceil(np.max(x)))
    N = len(x)
    result = np.zeros(N)
    best_frac = [(0, 0)] * N

    for f in range(1, max_freq + 1):
        k_vals = np.arange(f, x_max * f + 1)
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
    result *= output_max
    result[result < threshold] = threshold
    return result, best_frac

def build_pairwise_proximity_df(iois, params, threshold=0.01):
    n = len(iois)
    
    # --- 1) Ratio-Matrix berechnen ---
    ratios = np.ones((n, n), dtype=float)  # Diagonale = 1
    i_idx, j_idx = np.triu_indices(n, k=1)
    ratios[i_idx, j_idx] = np.maximum(iois[i_idx], iois[j_idx]) / np.minimum(iois[i_idx], iois[j_idx])
    ratios[j_idx, i_idx] = ratios[i_idx, j_idx]  # untere Dreiecksmatrix spiegeln
    
    # --- 2) Proximity-Matrix berechnen ---
    ratios_flat = ratios.flatten()
    prox_flat, best_frac_flat = proximity_max_freq_gaussian_table(ratios_flat, **params)
    prox_max = prox_flat.reshape(n, n)
    
    # --- 3) Peak-Labels erstellen ---
    peak_labels = np.full((n, n), "Keine Proximity", dtype=object)
    for idx, val in enumerate(prox_flat):
        if val > threshold:
            i, j = divmod(idx, n)
            k_val, f_val = best_frac_flat[idx]
            frac = Fraction(k_val, f_val).limit_denominator()
            peak_labels[i, j] = f"{frac.numerator}/{frac.denominator}"
    
    # --- 4) DataFrame erstellen ---
    df_pairs = pd.DataFrame({
        "i": np.repeat(np.arange(n), n),
        "j": np.tile(np.arange(n), n),
        "ioi_i": np.repeat(iois, n),
        "ioi_j": np.tile(iois, n),
        "ratio": ratios.flatten(),
        "prox_value": prox_flat,
        "peak_label": peak_labels.flatten()
    })
    
    return df_pairs, ratios, prox_max

def analyze_single_sequence(iois, params, num_sequences=1000, seed=42, bins=50, plot=True):
    """
    Analysiert eine einzelne IOI-Sequenz gegen zwei Nullmodelle (Exponential, Uniform)
    und erstellt Histogramm mit Mean, 95%-Quantilen, Z-Score und p-Wert.
    
    Rückgabe: dict mit realen Proximity-Paaren, Nullmodellen und Statistik.
    """
    rng = np.random.default_rng(seed)
    sequence_length = len(iois)
    max_ioi = float(np.max(iois))
    mean_ioi = float(np.mean(iois))
    
    # -----------------------------
    # Reale Proximity-Paare
    # -----------------------------
    ratios_real = np.array([[max(iois[i], iois[j])/min(iois[i], iois[j]) 
                             for j in range(sequence_length)] for i in range(sequence_length)])
    prox_mat_real, _ = proximity_max_freq_gaussian_table(ratios_real.flatten(), **params)
    prox_mat_real = prox_mat_real.reshape(ratios_real.shape)
    triu_idx = np.triu_indices_from(prox_mat_real, k=1)
    proximity_real = prox_mat_real[triu_idx]
    median_real = np.median(proximity_real)
    
    # -----------------------------
    # Nullmodell: Exponential
    # -----------------------------
    medians_expon = []
    proximity_expon_all = []
    for _ in range(num_sequences):
        sim_iois = rng.exponential(scale=mean_ioi, size=sequence_length)
        ratios = np.array([[max(sim_iois[i], sim_iois[j])/min(sim_iois[i], sim_iois[j]) 
                            for j in range(sequence_length)] for i in range(sequence_length)])
        prox_mat, _ = proximity_max_freq_gaussian_table(ratios.flatten(), **params)
        prox_mat = prox_mat.reshape(ratios.shape)
        vals = prox_mat[triu_idx]
        proximity_expon_all.append(vals)
        medians_expon.append(np.median(vals))
    proximity_expon_all = np.concatenate(proximity_expon_all)
    median_expon = np.mean(medians_expon)
    std_expon = np.std(medians_expon, ddof=1)
    zscore_expon = (median_real - median_expon) / (std_expon + 1e-12)
    pval_expon = (np.sum(np.array(medians_expon) >= median_real) + 1) / (num_sequences + 1)
    quant95_expon = np.percentile(medians_expon, 95)
    
    # -----------------------------
    # Nullmodell: Uniform
    # -----------------------------
    medians_uniform = []
    proximity_uniform_all = []
    for _ in range(num_sequences):
        sim_iois = rng.uniform(0.01, max_ioi, size=sequence_length)
        ratios = np.array([[max(sim_iois[i], sim_iois[j])/min(sim_iois[i], sim_iois[j]) 
                            for j in range(sequence_length)] for i in range(sequence_length)])
        prox_mat, _ = proximity_max_freq_gaussian_table(ratios.flatten(), **params)
        prox_mat = prox_mat.reshape(ratios.shape)
        vals = prox_mat[triu_idx]
        proximity_uniform_all.append(vals)
        medians_uniform.append(np.median(vals))
    proximity_uniform_all = np.concatenate(proximity_uniform_all)
    median_uniform = np.mean(medians_uniform)
    std_uniform = np.std(medians_uniform, ddof=1)
    zscore_uniform = (median_real - median_uniform) / std_uniform if std_uniform > 0 else np.nan
    pval_uniform = (np.sum(np.array(medians_uniform) >= median_real) + 1) / (num_sequences + 1)
    quant95_uniform = np.percentile(medians_uniform, 95)
    
    # -----------------------------
    # Plot
    # -----------------------------
    if plot:
        plt.figure(figsize=(12,6))
        colors = {"Real": "blue", "Expon": "orange", "Uniform": "gray"}
        
        # Gemeinsamer Wertebereich über beide Nullmodelle
        all_nulls = np.concatenate([medians_expon, medians_uniform])
        bin_edges = np.linspace(np.min(all_nulls), np.max(all_nulls), bins+1)

        plt.hist(proximity_real, bins=bin_edges, density=True, alpha=0.5, color=colors["Real"], label="Real")
        plt.hist(medians_expon, bins=bin_edges, density=True, alpha=0.3, color=colors["Expon"], label="Mediane von Exponential Null")
        plt.hist(medians_uniform, bins=bin_edges, density=True, alpha=0.3, color=colors["Uniform"], label="Mediane von Uniform Null")
        
        sns.kdeplot(proximity_real, bw_adjust=0.5, color=colors["Real"], linewidth=2)
        sns.kdeplot(medians_expon, bw_adjust=0.5, color=colors["Expon"], linewidth=2)
        sns.kdeplot(medians_uniform, bw_adjust=0.5, color=colors["Uniform"], linewidth=2)
    
        # Mittelwerte & 95%-Quantile
        plt.axvline(median_real, color=colors["Real"], linestyle="--", linewidth=2, label=f"Real Median ≈ {median_real:.3f}")
        plt.axvline(median_expon, color=colors["Expon"], linestyle="--", linewidth=2, label=f"Expon Mean ≈ {median_expon:.3f}")
        plt.axvline(quant95_expon, color=colors["Expon"], linestyle=":", linewidth=2, label=f"Expon 95%-Quantil ≈ {quant95_expon:.3f}")
        plt.axvline(median_uniform, color=colors["Uniform"], linestyle="--", linewidth=2, label=f"Uniform Mean ≈ {median_uniform:.3f}")
        plt.axvline(quant95_uniform, color=colors["Uniform"], linestyle=":", linewidth=2, label=f"Uniform 95%-Quantil ≈ {quant95_uniform:.3f}")
        
        plt.xlabel("Proximity-Werte")
        plt.ylabel("Dichte")
        plt.title("Proximity-Paare: Real vs. Nullmodelle")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()
    
    # -----------------------------
    # Ergebnisse zurückgeben
    # -----------------------------
    results = {
        "median_real": median_real,
        "median_expon": median_expon,
        "std_expon": std_expon,
        "zscore_expon": zscore_expon,
        "pval_expon": pval_expon,
        "median_uniform": median_uniform,
        "std_uniform": std_uniform,
        "zscore_uniform": zscore_uniform,
        "pval_uniform": pval_uniform,
        "quant95_expon": quant95_expon,
        "quant95_uniform": quant95_uniform,
    }
    if plot:
        print("\n=== Ergebnisse ===")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        
    return {
        "proximity_real": proximity_real,
        "proximity_expon_all": proximity_expon_all,
        "proximity_uniform_all": proximity_uniform_all,
        "results": results
    }

# =============================
# BEATFINDING FUNCTIONS
# =============================
def round_to_resolution(x, resolution):
    return round(x / resolution) * resolution

def generate_beat_candidates_from_iois(iois, resolution=0.001, max_denominator=24):
    if len(iois) == 0:
        return []

    max_ioi = float(np.max(iois))
    min_ioi = float(np.min(iois)) / max_denominator  # dynamisches min_ioi

    raster = np.arange(min_ioi, max_ioi + resolution/2, resolution)
    candidates = set(round_to_resolution(val, resolution) for val in raster)

    for ioi in iois:
        if ioi < min_ioi or ioi > max_ioi:
            continue
        for denom in range(1, max_denominator + 1):
            beat = ioi / denom
            if min_ioi <= beat <= max_ioi:
                candidates.add(round_to_resolution(beat, resolution))

    for denom in range(1, max_denominator + 1):
        val = 1.0 / denom
        if min_ioi <= val <= max_ioi:
            candidates.add(round_to_resolution(val, resolution))

    # === Filter: Kandidaten müssen mindestens so viele Beats abdecken wie IOIs existieren ===
    duration = float(np.sum(iois))
    n_iois = len(iois)
    filtered_candidates = []
    for beat in sorted(candidates):
        if beat <= 0:  # ❌ Filter für 0 und negative Beats
            continue
        n_beats = int(duration // beat)
        if n_beats >= n_iois:
            filtered_candidates.append(beat)

    return filtered_candidates

def score_for_beat(beat, iois, params):
    ratios = np.maximum(iois, beat) / np.minimum(iois, beat)
    prox_vals, _ = proximity_max_freq_gaussian_table(ratios, **params)
    return float(np.mean(prox_vals))

def find_local_maxima(candidates, scores, order=2):
    indices = argrelextrema(np.array(scores), np.greater, order=order)[0]
    return [(candidates[i], scores[i]) for i in indices]

def compute_beat_rmse(onsets, beat_period, resolution=0.001):
    """
    Berechnet den besten Phasen-Shift für die Beats, basierend auf RMSE.
    
    Parameters:
        onsets (array-like): Zeitpunkte der Onsets
        beat_period (float): vermutete Beatdauer
        resolution (float): Schrittweite für Phasen-Shift-Suche
    
    Returns:
        best_shift (float): Shift, der RMSE minimiert
        best_rmse (float): Minimaler RMSE-Wert
        best_beats (np.array): Beat-Zeiten mit optimalem Shift
    """
    onsets = np.array(onsets)
    first_onset = onsets[0]
    last_onset = onsets[-1]
    
    search_range = np.arange(0, beat_period, resolution)
    best_shift = 0.0
    best_rmse = float('inf')
    best_beats = None

    for shift in search_range:
        # Beat-Zeiten, einen Beat davor und einen danach erweitern
        start_beat = first_onset - beat_period + shift
        end_beat = last_onset + beat_period
        beats = np.arange(start_beat, end_beat + resolution, beat_period)

        # Für jeden Onset den nächsten Beat suchen
        nearest_beats = beats[np.argmin(np.abs(onsets[:, None] - beats[None, :]), axis=1)]

        # RMSE berechnen
        rmse = np.sqrt(((onsets - nearest_beats)**2).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift
            best_beats = beats

    return best_shift, best_rmse, best_beats

# =============================
# 2) Nullmodelle generieren
# =============================
def build_null_ratios_table(n, model, rng, iois):
    if model == "expon":
        sim_iois = rng.exponential(scale=np.mean(iois), size=n)
    elif model == "uniform":
        sim_iois = rng.uniform(np.min(iois), np.max(iois), size=n)
    elif model == "empirical":
        sim_iois = rng.choice(iois, size=n, replace=True)
    triu_idx = np.triu_indices(n, k=1)
    ratios = np.maximum(sim_iois[:, None], sim_iois[None, :]) / np.minimum(sim_iois[:, None], sim_iois[None, :])
    return ratios[triu_idx]

# =============================
# 3) Peak-Signifikanz für eine Sequenz
# =============================
def compute_peak_significance_table(iois, params, num_null=200, seed=42, source="", alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(iois)
    if n < 2:
        return pd.DataFrame()
    triu_idx = np.triu_indices(n, k=1)
    ratios_real = np.maximum(iois[:, None], iois[None, :]) / np.minimum(iois[:, None], iois[None, :])
    ratios_real = ratios_real[triu_idx]

    prox_real, best_frac_real = proximity_max_freq_gaussian_table(ratios_real, **params)
    threshold = params.get("threshold", 0.01)

    peak_labels = [f"{k}/{f}" if (k,f)!=(0,0) else "keine Proximity" for k,f in best_frac_real]
    peaks = sorted(set([p for p in peak_labels if p != "keine Proximity"]))

    peak_real_scores = {peak: np.array([prox_real[i] if peak_labels[i]==peak else threshold
                                        for i in range(len(prox_real))])
                        for peak in peaks}

    null_scores = {model: {peak: [] for peak in peaks} for model in ["expon", "uniform", "empirical"]}

    for model in ["expon", "uniform", "empirical"]:
        for _ in range(num_null):
            ratios_null = build_null_ratios_table(n, model, rng, iois)
            prox_null, best_frac_null = proximity_max_freq_gaussian_table(ratios_null, **params)
            peak_labels_null = [f"{k}/{f}" if (k,f)!=(0,0) else "keine Proximity" for k,f in best_frac_null]
            for peak in peaks:
                vals = np.array([prox_null[i] if peak_labels_null[i]==peak else threshold
                                 for i in range(len(prox_null))])
                null_scores[model][peak].append(np.mean(vals))

    results = []
    for peak in peaks:
        peak_data = {"peak": peak}
        sig_models = []
        for model in ["expon", "uniform", "empirical"]:
            pvals = np.array(null_scores[model][peak])
            mean_null = np.mean(pvals)
            pval = (np.sum(pvals >= np.mean(peak_real_scores[peak])) + 1) / (num_null + 1)
            peak_data[f"{model}_mean_null"] = mean_null
            peak_data[f"{model}_pval"] = pval
            if pval < alpha:
                sig_models.append(model)
        peak_data[f"significance_alpha: {alpha}"] = ", ".join(sig_models) if sig_models else "n.s."

        peak_data["mean_real"] = np.mean(peak_real_scores[peak])
        # Level + Quelle
        peak_data["level"] = "sequence"
        peak_data["source"] = source
        results.append(peak_data)

    df = pd.DataFrame(results).sort_values("mean_real", ascending=False).reset_index(drop=True)
    # Spaltenreihenfolge anpassen: peak | significance_dataset | level | source | Rest
    col_order = ["peak", f"significance_alpha: {alpha}", "level", "source"] + \
        [c for c in df.columns if c not in ["peak", f"significance_alpha: {alpha}", "level", "source"]]
    return df[col_order]

# =============================
# 4) Dataset-Analyse
# =============================

def analyze_dataset_peaks_table(path, params, num_null=200, alpha=0.05):
    path = Path(path)
    if path.is_file():
        df = pd.read_csv(path)
        iois = df["IOI"].dropna().values
        return compute_peak_significance_table(iois, params, num_null=num_null, source=path.name, alpha=alpha)
    elif path.is_dir():
        all_peak_dfs = []
        for file in path.glob("*.csv"):
            df = pd.read_csv(file)
            iois = df["IOI"].dropna().values
            peak_df = compute_peak_significance_table(iois, params, num_null=num_null, source=file.name, alpha=alpha)
            if not peak_df.empty:
                all_peak_dfs.append(peak_df)
        if not all_peak_dfs:
            return pd.DataFrame()

        df_concat = pd.concat(all_peak_dfs, ignore_index=True)

        grouped = df_concat.groupby("peak")
        final_rows = []
        for peak, g in grouped:
            row = {
                "peak": peak,
                f"significance_alpha: {alpha}": ", ".join([model for model in ["expon","uniform","empirical"]
                                                    if np.sum(g[f"{model}_pval"] < alpha) > len(g)/2]),
                "mean_real": g["mean_real"].mean(),
                "expon_mean_null": g["expon_mean_null"].mean(),
                "expon_pval": g["expon_pval"].mean(),
                "uniform_mean_null": g["uniform_mean_null"].mean(),
                "uniform_pval": g["uniform_pval"].mean(),
                "empirical_mean_null": g["empirical_mean_null"].mean(),
                "empirical_pval": g["empirical_pval"].mean(),
                "level": "dataset",
                "source": path.name
            }
            if row[f"significance_alpha: {alpha}"] == "":
                row[f"significance_alpha: {alpha}"] = "n.s."
            final_rows.append(row)

        df = pd.DataFrame(final_rows).sort_values("mean_real", ascending=False)
        # Spaltenreihenfolge: peak | significance_dataset | level | source | Rest
        col_order = ["peak", f"significance_alpha: {alpha}", "level", "source"] + \
            [c for c in df.columns if c not in ["peak", f"significance_alpha: {alpha}", "level", "source"]]

        return df[col_order]
    else:
        raise ValueError(f"{path} is neither file nor folder!")

# =============================
# 5) Ergebnisse zusammenfassen
# =============================

def summarize_results_table(df, alpha=0.05):
    summaries = []
    for _, row in df.iterrows():
        filename = str(row["source"])
        dtype = str(row["level"])

        # Peak-Klassifikation
        peak = row["peak"]
        sig_models = []
        for model in ["expon", "uniform", "empirical"]:
            pval = row[f"{model}_pval"]
            if pval < alpha:
                sig_models.append(f"True: {peak}: {model} ({pval:.4f})")

        if not sig_models:
            sig_text = "False"
        else:
            sig_text = "\n".join(sig_models)

        # Kategorie bestimmen
        if peak == "1/1":
            category = "Isochrony"
        elif peak.endswith("/1") and not peak.startswith("1/"):
            category = "Simple Heterochrony"
        else:
            category = "Complex Heterochrony"

        summaries.append({
            "Filename": filename,
            "Type": dtype,
            "Isochrony": "False",
            "Simple Heterochrony": "False",
            "Complex Heterochrony": "False",
            category: sig_text
        })

    # In DataFrame packen
    df_summary = pd.DataFrame(summaries)

    # Gruppieren und Strings sicher zusammenfassen
    def merge_vals(series):
        vals = [str(v) for v in series if str(v) != "False" and pd.notna(v)]
        return "\n".join(vals) if vals else "False"

    df_summary = df_summary.groupby(["Filename", "Type"], as_index=False).agg({
        "Isochrony": merge_vals,
        "Simple Heterochrony": merge_vals,
        "Complex Heterochrony": merge_vals
    })

    # Spaltenreihenfolge sicherstellen
    return df_summary[["Filename", "Type", "Isochrony", "Simple Heterochrony", "Complex Heterochrony"]]

# =============================
# 6) Tabelle ausgeben
# =============================
def print_peak_table(df, title=None):
    if title:
        print(f"\n{title}\n" + "="*len(title))
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4f"))

def plot_proximity_max_freq_gaussian(
    sharpness_base=12.0,
    sharpness_growth=0.5,
    decay=0.2,
    max_freq=3,
    x_max=4.0,
    weight_exponent=0.5,
    freq_sharpness_exp=0.5,
):
    x_vals = np.linspace(1, x_max + 0.5, 2000)
    y_vals, _ = proximity_max_freq_gaussian_table(
        x_vals,
        sharpness_base=sharpness_base,
        sharpness_growth=sharpness_growth,
        decay=decay,
        max_freq=max_freq,
        x_max=x_max,
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
        print("KDE konnte nicht berechnet werden (konstante Werte?)")

    axes[1].axvline(mean_prox, color='red', linestyle='--', linewidth=2, label=f"∫mean ≈ {mean_prox:.3f}")
    axes[1].set_title("Proximity Distribution")
    axes[1].set_xlabel("Proximity")
    axes[1].set_ylabel("Normalized frequency")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
