# ===============================================
# Analyse von IOI-Verhältnissen zur Beat-Detektion 
# Mehrere Parameter-Sets ausprobieren
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson
from fractions import Fraction
from scipy.stats import gaussian_kde

# =============================
# 1) Proximity-Funktion
# =============================
def proximity_max_freq_gaussian(x, sharpness_base=10.0, sharpness_growth=1.0, decay=0.1,
                                max_freq=4, weight_exponent=1.0, freq_sharpness_exp=1.0,
                                neighbor_peaks=2, norm_x=1.0, output_max=1.0):
    x = np.atleast_1d(x).astype(float).ravel()
    gauss_max = np.zeros_like(x)
    peak_idx = np.zeros_like(x, dtype=int)  # <- neu
    freqs = np.arange(1, max_freq + 1)

    for f in freqs:
        weight = 1.0 / (f ** weight_exponent)
        k_center = np.round(x * f).astype(int)
        offsets = np.arange(-neighbor_peaks, neighbor_peaks + 1)
        k_all = k_center[:, None] + offsets[None, :]
        valid_mask = k_all > 0
        mu = k_all / f
        mu[~valid_mask] = 1.0
        sharpness = sharpness_base * (f ** freq_sharpness_exp) * (np.maximum(k_all, 1) ** sharpness_growth)
        sharpness[~valid_mask] = 1e-9
        sigma = 1.0 / (sharpness + 1e-9)
        mu_decay = np.where(valid_mask, 1.0 / (mu ** decay + 1e-9), 0.0)
        diff = x[:, None] - mu
        g = np.exp(-0.5 * (diff / sigma) ** 2) * mu_decay * weight
        g_max = np.max(g, axis=1)
        better = g_max > gauss_max
        gauss_max[better] = g_max[better]
        peak_idx[better] = f   # <- hier merken, bei welcher f das war

    max_at_norm_x = np.max([1.0 / (f ** weight_exponent) for f in freqs])
    gauss_max = gauss_max / (max_at_norm_x + 1e-9) * output_max
    return gauss_max, peak_idx, freqs

# =============================
# 2) Funktion für IOI-Paare
# =============================
def build_pairwise_proximity_df(iois, params, threshold=0.01):
    n = len(iois)
    ratios = np.array([[max(iois[i], iois[j]) / min(iois[i], iois[j]) 
                        for j in range(n)] for i in range(n)])
    
    prox_max_flat, peak_idx_flat, freqs = proximity_max_freq_gaussian(ratios.flatten(), **params)
    prox_max = prox_max_flat.reshape(ratios.shape)
    peak_idx = peak_idx_flat.reshape(ratios.shape)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(ratios[i, j])
            s = float(prox_max[i, j])
            if s > threshold:
                best_f = peak_idx[i, j]   # direkt die Frequenz, kein Index mehr
                p = int(round(r * best_f))
                frac = Fraction(p, best_f)
                label = f"{frac.numerator}/{frac.denominator}"
            else:
                label = "Keine Proximity"

            rows.append({
                "i": i,
                "j": j,
                "ioi_i": float(iois[i]),
                "ioi_j": float(iois[j]),
                "ratio": r,
                "proximity": s,
                "peak_label": label
            })

    df_pairs = pd.DataFrame(rows)
    return df_pairs, ratios, prox_max

# =============================
# 3) Daten einlesen
# =============================
PAIR_THRESHOLD = 0.01
base_dir = Path("/Users/emilschmiedl/Desktop/Proximity-Funktion/data/theoretical_sequences/baseIOI-0.50/100beats/")
filenames = ["random_exponential_03.csv",
             "heterochrony-2-3_jitter-0050_07.csv",
             "isochrony_jitter-0050_08.csv"] 
colors = ["red", "blue", "green"]

# =============================
# 4) Verschiedene Parameter-Sets definieren
# =============================
param_sets = [
    dict(sharpness_base=6.0, sharpness_growth=0, decay=100,
         max_freq=1, weight_exponent=1.0, freq_sharpness_exp=1.0,
         neighbor_peaks=2, norm_x=1.0, output_max=1.0),

    dict(sharpness_base=8.0, sharpness_growth=1.0, decay=0.1,
         max_freq=2, weight_exponent=0, freq_sharpness_exp=0,
         neighbor_peaks=2, norm_x=1.0, output_max=1.0),

    dict(sharpness_base=8.0, sharpness_growth=0.5, decay=0.2,
         max_freq=3, weight_exponent=0.5, freq_sharpness_exp=1.0,
         neighbor_peaks=2, norm_x=1.0, output_max=1.0)
]
# =============================
# 5) Alle Param-Sets analysieren & nebeneinander plotten
# =============================

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

legend_handles = []
legend_labels = []

for idx, (ax, params) in enumerate(zip(axes, param_sets), start=1):
    dfs = []
    for fname in filenames:
        seq_path = base_dir / fname
        if not seq_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {seq_path.resolve()}")
        df = pd.read_csv(seq_path)
        iois = df["IOI"].dropna().values
        df_pairs, ratios, prox_max = build_pairwise_proximity_df(iois, params, threshold=PAIR_THRESHOLD)
        df_pairs["source"] = fname
        dfs.append(df_pairs)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- Densityplot + Proximity-Kurve ---
    x_min = max(1.0, df_all["ratio"].min() * 0.95)
    x_max = df_all["ratio"].max() * 1.05
    x_range = np.linspace(x_min, x_max, 2000)
    y_curve, _, _ = proximity_max_freq_gaussian(x_range, **params)
    mean_prox = simpson(y_curve, x=x_range) / (x_range[-1] - x_range[0])

    # Proximity-Kurve
    curve_handle, = ax.plot(
        x_range, y_curve,
        alpha=0.7, color="gray",
        label=f"Proximity-curve\n(∫mean ≈ {mean_prox:.3f})"
    )

    # Density-Plots für Ratios
    for fname, color in zip(filenames, colors):
        df_seq = df_all[df_all["source"] == fname]
        ratios = df_seq["ratio"].values
        if len(ratios) > 1:  # KDE braucht >1 Datenpunkte
            kde = gaussian_kde(ratios)
            y_kde = kde(x_range)
            y_kde = y_kde / y_kde.max() * y_curve.max() * 0.8
            h, = ax.plot(x_range, y_kde, color=color, lw=2, label=f"density {fname}")
            # Nur im ersten Subplot Handles sammeln (damit Legende nicht x3 lang wird)
            if idx == 1:
                legend_handles.append(h)
                legend_labels.append(f"ratio-density: {fname}")

    # Untertitel für jedes Param-Set definieren
    param_subtitles = {
        1: "isochrony only",
        2: "simple heterochrony, 2nd harmonic, punished high ratios",
        3: "high complexity, multiple harmonics, mild decay"
    }

    # Haupttitel
    ax.set_title(f"Param-Set {idx}", fontsize=12, fontweight="bold", pad=18)

    # Untertitel (leicht darunter, kleiner und grau)
    ax.text(
        0.5, 1.02,   # 1.02 = leicht über dem Plot, aber unterhalb des Titels
        param_subtitles.get(idx, ""),
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9, color="gray"
    )

    ax.set_xlabel("ratio")
    if idx == 1:
        ax.set_ylabel("proximity-funciton (grey) / ratio-density (color)")
    ax.set_xlim(1, 5)
    ax.grid(True)

        # Param-Set als Text in die obere rechte Ecke
    param_text = "\n".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in params.items()])
    ax.text(
        0.98, 0.98, param_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    # Proximity-Kurve auch für Legende nur einmal merken
    if idx == 1:
        legend_handles.insert(0, curve_handle)
        legend_labels.insert(0, curve_handle.get_label())

# Gemeinsame Legende unten
fig.legend(
    legend_handles, legend_labels,
    loc="lower center", ncol=len(legend_labels),
    frameon=True, fontsize=9
)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Platz für Legende unten
plt.show()