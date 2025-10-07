# "richtiger" code!
# ===============================================
# IOI-Proximity Analyse: Peak-Signifikanz über Dataset/Sequenz
# inkl. expon, uniform, empirisches Nullmodell, vektorisiert
# ===============================================

import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate

# =============================
# 1) Gaussian-based Proximity-Funktion (Vektorisiert)
# =============================
def proximity_max_freq_gaussian(
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

# =============================
# 2) Nullmodelle generieren
# =============================
def build_null_ratios(n, model, rng, iois):
    if model == "expon":
        sim_iois = rng.exponential(scale=np.median(iois), size=n)
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
def compute_peak_significance(iois, params, num_null=200, seed=42, source="", alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(iois)
    if n < 2:
        return pd.DataFrame()
    triu_idx = np.triu_indices(n, k=1)
    ratios_real = np.maximum(iois[:, None], iois[None, :]) / np.minimum(iois[:, None], iois[None, :])
    ratios_real = ratios_real[triu_idx]

    prox_real, best_frac_real = proximity_max_freq_gaussian(ratios_real, **params)
    threshold = params.get("threshold", 0.01)

    peak_labels = [f"{k}/{f}" if (k,f)!=(0,0) else "keine Proximity" for k,f in best_frac_real]
    peaks = sorted(set([p for p in peak_labels if p != "keine Proximity"]))

    peak_real_scores = {peak: np.array([prox_real[i] if peak_labels[i]==peak else threshold
                                        for i in range(len(prox_real))])
                        for peak in peaks}

    null_scores = {model: {peak: [] for peak in peaks} for model in ["expon", "uniform", "empirical"]}

    for model in ["expon", "uniform", "empirical"]:
        for _ in range(num_null):
            ratios_null = build_null_ratios(n, model, rng, iois)
            prox_null, best_frac_null = proximity_max_freq_gaussian(ratios_null, **params)
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
            if pval < alpha:   # <<<<<<<<<<<< statt fest 0.05
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

def analyze_dataset_peaks(path, params, num_null=200, alpha=0.05):
    path = Path(path)
    if path.is_file():
        df = pd.read_csv(path)
        iois = df["IOI"].dropna().values
        return compute_peak_significance(iois, params, num_null=num_null, source=path.name, alpha=alpha)
    elif path.is_dir():
        all_peak_dfs = []
        for file in path.glob("*.csv"):
            df = pd.read_csv(file)
            iois = df["IOI"].dropna().values
            peak_df = compute_peak_significance(iois, params, num_null=num_null, source=file.name, alpha=alpha)
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

def summarize_results(df, alpha=0.05):
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

# =============================
# 7) Beispiel-Aufruf
# =============================
default_params = dict(
    sharpness_base=14.0, sharpness_growth=0.0, decay=0.0,
    max_freq=4, x_max=10, weight_exponent=0.0, freq_sharpness_exp=1.5,
    output_max=1.0, threshold=0.01
)

def show_results(path, name, num_null=100, alpha=0.05):
    df = analyze_dataset_peaks(path, default_params, num_null=num_null, alpha=alpha)
    print_peak_table(df, name)  # große Tabelle

    df_summary = summarize_results(df, alpha=alpha)
    print(f"\n{name} (Summary)")
    print(tabulate(df_summary, headers="keys", tablefmt="fancy_grid", showindex=False))

    return df, df_summary

# Anwendung:
accelerando = show_results("/Users/emilschmiedl/Desktop/Proximity-Funktion/data/ergebnis-tabellen/accelerando/accelerando.csv", "Accelerando")
heterochrony_1_2 = show_results("../data/ergebnis-tabellen/heterochrony-1-2", "Heterochrony 1-2")
heterochrony_2_3 = show_results("../data/ergebnis-tabellen/heterochrony-2-3", "Heterochrony 2-3")
isochrony_small_jitter = show_results("../data/ergebnis-tabellen/isochrony_small-jitter", "Isochrony small jitter")
isochrony_big_jitter = show_results("../data/ergebnis-tabellen/isochrony_big-jitter", "Isochrony big jitter")
random_exponential = show_results("../data/ergebnis-tabellen/random exponential", "Random exponential")
random_uniform = show_results("../data/ergebnis-tabellen/random uniform", "Random uniform")
kwa = show_results("../data/Kwa-Sounds Scorpaena spp", "KWa-Sounds Scorpaena spp")
carollia = show_results("../data/Bat Carollia perspicillata (C_persp)", "Carollia perspicillata")
whale = show_results("../data/Sperm whale Physeter macrocephalus (sw)", "Whale Codas", num_null=50)
