# ausreichende Initialisierung ohne Plots
# === 0) new Initialisierung ===

# ===============================================
# Analyse von IOI-Verhältnissen zur Beat-Detektion 
# ===============================================

# =============================
# 0) Imports
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

# =============================
# 1) Proximity-Funktion (Vektorisierte Version)
# =============================

import numpy as np

def proximity_max_freq_gaussian(
    x,
    sharpness_base=10.0,
    sharpness_growth=1.0,
    decay=0.1,
    max_freq=4,
    weight_exponent=1.0,
    freq_sharpness_exp=1.0,
    neighbor_peaks=2,   # wie viele Peaks links+rechts von x getestet werden
    norm_x=1.0,
    output_max=1.0
):
    """
    Optimierte Gaussian-Proximity-Funktion:
    - prüft nur die Ratio-Peaks, die in der Nähe von x liegen (k/f ± neighbor_peaks)
    - x: Skalar oder Array
    - Rückgabe: Array gleicher Form
    """
    x = np.array(x, dtype=float)
    all_freq_values = []

    for f in range(1, max_freq + 1):
        weight = 1.0 / (f ** weight_exponent)
        gauss_max = np.zeros_like(x, dtype=float)

        # nächstliegender Peak-Index zu jedem x
        k_center = np.round(x * f).astype(int)

        for offset in range(-neighbor_peaks, neighbor_peaks + 1):
            k = k_center + offset
            valid = k > 0  # negative oder 0 Peaks sind Unsinn

            if not np.any(valid):
                continue

            mu = k[valid] / f
            sharpness = sharpness_base * (f ** freq_sharpness_exp) * (k[valid] ** sharpness_growth)
            sigma = 1.0 / (sharpness + 1e-9)
            mu_decay = 1.0 / (mu ** decay)

            # Broadcasting über alle validen Peaks
            g = np.exp(-0.5 * ((x[..., None] - mu[None, ...]) / sigma[None, ...]) ** 2)
            contrib = weight * mu_decay * g

            # Max über Peaks in Nachbarschaft
            gauss_max = np.maximum(gauss_max, np.max(contrib, axis=-1))

        all_freq_values.append(gauss_max)

    result = np.maximum.reduce(all_freq_values)

    # Normierung
    max_at_norm_x = np.max([
        (1.0 / (f ** weight_exponent))
        for f in range(1, max_freq + 1)
    ])
    return result / (max_at_norm_x + 1e-9) * output_max

# =============================
# 2) Funktion für IOI-Paare
# =============================

def build_pairwise_proximity_df(iois, params, threshold=0.01):
    n = len(iois)
    ratios = np.array([[max(iois[i], iois[j]) / min(iois[i], iois[j]) 
                        for j in range(n)] for i in range(n)])
    prox_max = proximity_max_freq_gaussian(ratios, **params)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(ratios[i, j])
            s = float(prox_max[i, j])
            if s > threshold:
                # Bestes f und k rekonstruieren
                best_f = None
                best_frac = None
                best_diff = np.inf
                for f in range(1, params["max_freq"] + 1):
                    k = round(r * f)
                    frac = Fraction(k, f).limit_denominator()
                    diff = abs(r - k / f)
                    if diff < best_diff:
                        best_diff = diff
                        best_frac = frac
                        best_f = f
                label = f"{best_frac.numerator}/{best_frac.denominator}"
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
# 4) Parameter, Daten laden & Pairs bauen
# =============================
params = dict(
    sharpness_base=8.0,
    sharpness_growth=1.0,
    decay=0.1,
    max_freq=4,
    weight_exponent=1.0,
    freq_sharpness_exp=1.0,
    neighbor_peaks=2, 
    norm_x=1.0,
    output_max=1.0
)

PAIR_THRESHOLD = 0.01

filename = "heterochrony-2-3_jitter-0100_07.csv"
base_dir = Path("/Users/emilschmiedl/Desktop/Proximity-Funktion/data/theoretical_sequences/baseIOI-0.50/20beats/")
seq_path = base_dir / filename
if not seq_path.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {seq_path.resolve()}")

df = pd.read_csv(seq_path)

# Nur gültige IOIs verwenden
iois = df["IOI"].dropna().values
onsets = df["onset"].dropna().values

# Pairs sofort neu berechnen – kein alter Cache!
df_pairs, ratios, prox_max = build_pairwise_proximity_df(iois, params, threshold=PAIR_THRESHOLD)

# =============================
# Anwendung auf Daten
# =============================

# Dynamische Grenzen aus den Daten bestimmen
x_min = max(1.0, df_pairs["ratio"].min() * 0.95)  # kleiner Puffer nach unten
x_max = df_pairs["ratio"].max() * 1.05            # Puffer nach oben

# Blaue Kurve über den ganzen Plotbereich
x_range = np.linspace(x_min, x_max, 2000)
y_curve = proximity_max_freq_gaussian(x_range, **params)

# Mittelwert für diesen Bereich
mean_prox = simpson(y_curve, x=x_range) / (x_range[-1] - x_range[0])

# === 2) Recurrence-Plot (Peak-Label-Farbskalen) ===

# === all_pairs_df erstellen ===
records = []
n = len(iois)
for i in range(n):
    for j in range(n):
        ratio = max(iois[i], iois[j]) / min(iois[i], iois[j])
        prox_val = proximity_max_freq_gaussian([ratio], **params)[0]
        if prox_val > PAIR_THRESHOLD:
            frac = Fraction(ratio).limit_denominator(params["max_freq"])
            peak_label = f"{frac.numerator}/{frac.denominator}"
        else:
            peak_label = "Keine Proximity"
        records.append({
            "i": i,
            "j": j,
            "ioi_i": iois[i],
            "ioi_j": iois[j],
            "ratio": ratio,
            "prox_value": prox_val,
            "peak_label": peak_label
        })
all_pairs_df = pd.DataFrame.from_records(records)

# === Plot vorbereiten ===
pivot_vals = all_pairs_df.pivot(index="i", columns="j", values="prox_value") * 100
pivot_labels = all_pairs_df.pivot(index="i", columns="j", values="peak_label")

unique_labels = sorted(set(all_pairs_df["peak_label"]))

# --- Fixe Farbzuordnung für kleine Brüche ---
tab10 = plt.cm.tab10.colors  

fixed_colors = {
    "1/1": tab10[0],  # blau
    "2/1": tab10[1],  # orange
    "3/1": tab10[2],  # grün
    "4/1": tab10[3],  # rot
    "5/1": tab10[4],  # lila
    "1/2": tab10[5],  # braun
    "3/2": tab10[6],  # pink
    "5/2": tab10[7],  # grau
    "1/3": tab10[8],  # olivgrün
    "2/3": tab10[9],  # hellblau
    "4/3": (0.5, 0.0, 0.5)  # extra-farbe (dunkelviolett), da tab10 nur 10 hat
}

label_to_cmap = {}

# Zuerst feste Farben vergeben
for label, color in fixed_colors.items():
    if label in unique_labels:
        cmap_colors = [(1,1,1), plt.cm.colors.to_rgb(color)]
        label_to_cmap[label] = plt.cm.colors.LinearSegmentedColormap.from_list(f"{label}_cmap", cmap_colors)

# Rest wie vorher aus Palette
remaining_labels = [lab for lab in unique_labels if lab not in label_to_cmap]
palette = sns.color_palette("Set2", len(remaining_labels))
for label, base_color in zip(remaining_labels, palette):
    cmap_colors = [(1,1,1), base_color]
    label_to_cmap[label] = plt.cm.colors.LinearSegmentedColormap.from_list(f"{label}_cmap", cmap_colors)

# === 3) Kombiniertes Histogramm mit Anteilen + 95%-Quantil ===
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
    prox_mat_real = proximity_max_freq_gaussian(ratios_real, **params)
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
        prox_mat = proximity_max_freq_gaussian(ratios, **params)
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
        prox_mat = proximity_max_freq_gaussian(ratios, **params)
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

results = analyze_single_sequence(iois, params, num_sequences=100, seed=42, bins=25, plot=False)

# === 4) Beatfinding via Kandidaten und lokalen Maxima ===

# =============================
# 2) Hilfsfunktionen
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
    prox_vals = proximity_max_freq_gaussian(ratios, **params)
    return float(np.mean(prox_vals))

def find_local_maxima(candidates, scores, order=2):
    indices = argrelextrema(np.array(scores), np.greater, order=order)[0]
    return [(candidates[i], scores[i]) for i in indices]

# =============================
# 7) Beat-Kandidaten
# =============================
beat_candidates = generate_beat_candidates_from_iois(iois)
scores = [score_for_beat(beat, iois, params) for beat in beat_candidates]
local_maxima = find_local_maxima(beat_candidates, scores, order=5)
local_maxima_top5 = sorted(local_maxima, key=lambda x: -x[1])[:5]

# =============================
# 8) Top-5 Onset-Fit
# =============================

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

# -----------------------------
# Figure vorbereiten
# -----------------------------
fig = plt.figure(figsize=(20, 28))

# GridSpec: jetzt 7 Zeilen
gs = gridspec.GridSpec(
    7, 2, figure=fig,
    width_ratios=[1, 1],
    height_ratios=[0.5, 1, 1, 1, 1, 1, 1],
    hspace=0.5, wspace=0.3
)

# -----------------------------
# Header
# -----------------------------
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis("off")
ax_header.text(
    0.01, 1.0,
    f"IOI-Analyse:\n{filename}",
    transform=ax_header.transAxes,
    fontsize=22, fontweight='bold',
    va='top', ha='left'
)

# Rechter Zusatz oben rechts
info_lines = [
    f"IOI-Länge: {len(iois)} | min: {np.min(iois):.3f}, max: {np.max(iois):.3f}, Ø: {np.mean(iois):.3f}",
    f"PAIR_THRESHOLD: {PAIR_THRESHOLD}",
    f"Params: sharpness_base={params['sharpness_base']}, sharpness_growth={params['sharpness_growth']}, decay={params['decay']}",
    f"max_freq={params['max_freq']}, weight_exponent={params['weight_exponent']}, freq_sharpness_exp={params['freq_sharpness_exp']}",
    f"norm_x={params['norm_x']}, output_max={params['output_max']}",
    f"Top-5 Beats: " + ", ".join([f"{b:.3f}s ({s:.2f})" for b,s in local_maxima_top5[:5]]),
    f"Median Real: {results['results']['median_real']:.3f} | Median Expon: {results['results']['median_expon']:.3f} | Median Uniform: {results['results']['median_uniform']:.3f}",
    f"Std Expon: {results['results']['std_expon']:.3f} | Z-Score Expon: {results['results']['zscore_expon']:.3f} | p-Wert Expon: {results['results']['pval_expon']:.3f} | 95%-Quantil Expon: {results['results']['quant95_expon']:.3f}", 
    f"Std Uniform: {results['results']['std_uniform']:.3f} | Z-Score Uniform: {results['results']['zscore_uniform']:.3f} | p-Wert Uniform: {results['results']['pval_uniform']:.3f} | 95%-Quantil Uniform: {results['results']['quant95_uniform']:.3f}"
]

ax_header.text(
    0.99, 1.0,                       # oben rechts
    "\n".join(info_lines),
    transform=ax_header.transAxes,
    fontsize=11,
    va='top',
    ha='right'
)

# -----------------------------
# 1) Proximity-Funktion
# -----------------------------
# Gemeinsame X-Achsen-Grenzen berechnen
x_min = max(1.0, df_pairs["ratio"].min() * 0.95)
x_max = df_pairs["ratio"].max() * 1.05

# -----------------------------
# 1) Proximity-Funktion
# -----------------------------
ax1 = fig.add_subplot(gs[1, :])
x_range = np.linspace(x_min, x_max, 2000)
y_curve = proximity_max_freq_gaussian(x_range, **params)
mean_prox = np.mean(y_curve)
ax1.plot(x_range, y_curve, label=f"Proximity (mean ≈ {mean_prox:.3f})", alpha=0.5, color="grey")
ax1.scatter(df_pairs["ratio"].values, df_pairs["proximity"].values, s=10, alpha=0.7, color="red", label="IOI-Verhältnisse")
ax1.set_title("Proximity-Funktion mit IOI-Verhältnissen", fontweight='bold')
ax1.set_xlabel("Verhältnis")
ax1.set_ylabel("Proximity")
ax1.grid(True)
ax1.legend()
ax1.set_xlim(x_min, x_max)   # <--- hier!

# -----------------------------
# 2) Verteilung IOI-Verhältnisse (neu!)
# -----------------------------
ax2 = fig.add_subplot(gs[2, :])
sns.kdeplot(df_pairs["ratio"], fill=True, color="green", alpha=0.5, lw=2, bw_adjust=0.1, ax=ax2)
sns.histplot(df_pairs["ratio"], bins=50, color="gray", alpha=0.3, stat="density", ax=ax2)

ax2.set_title("Verteilung der IOI-Verhältnisse", fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Verhältnis (IOI_i / IOI_j)")
ax2.set_ylabel("Dichte")
ax2.set_xlim(x_min, x_max)
ax2.grid(True)

# -----------------------------
# 3) Recurrence-Matrix links
# -----------------------------
ax3 = fig.add_subplot(gs[3:5, 0])
# -----------------------------
pivot_vals = all_pairs_df.pivot(index="i", columns="j", values="prox_value") * 100
pivot_labels = all_pairs_df.pivot(index="i", columns="j", values="peak_label")
n = len(iois)

for (i, j), val in np.ndenumerate(pivot_vals.values):
    label = pivot_labels.iloc[i, j]
    cmap = label_to_cmap[label]

    # Falls NaN -> weiß färben
    if np.isnan(val):
        color = (1, 1, 1, 1)
    else:
        color = cmap(val / 100)

    rect = plt.Rectangle([j, i], 1, 1, facecolor=color, edgecolor='black')
    ax3.add_patch(rect)

    # Werte in kleine Matrizen schreiben
    if n <= 20 and not np.isnan(val):
        ax3.text(
            j + 0.5, i + 0.5,
            f"{float(val.item()):.1f}",
            ha='center', va='center',
            fontsize=6, color="black"
        )


ax3.set_xlim(0, n)
ax3.set_ylim(0, n)
ax3.set_aspect('equal')
ax3.set_xticks(np.arange(n)+0.5)
ax3.set_yticks(np.arange(n)+0.5)
ax3.set_xticklabels(range(n), rotation=45, ha="right")
ax3.set_yticklabels(range(n))
ax3.set_xlabel("j")
ax3.set_ylabel("i")
ax3.set_title("Proximity-Matrix der IOI-Verhältnisse\n", fontweight='bold')

# aktuelle Position auslesen
pos = ax3.get_position()

# leicht nach links verschieben (x0 und x1 kleiner machen)
shift = 0.0227   # Feinjustieren!
ax3.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])

# neue Position nach dem Verschieben holen
pos = ax3.get_position()

# Farbskalen rechts der Matrix
unique_labels = sorted(set(all_pairs_df["peak_label"]))
legend_labels = [lab for lab in unique_labels if lab != "Keine Proximity"]

cbar_width = 0.05
spacing = 0.01
n_labels = len(legend_labels)
bar_height = (pos.height - (n_labels-1)*spacing) / n_labels

for idx, label in enumerate(legend_labels):
    cmap = label_to_cmap[label]
    norm = plt.Normalize(vmin=0, vmax=100)
    cbar_y = pos.y0 + pos.height - (idx+1)*bar_height - idx*spacing
    cbar_ax = fig.add_axes([pos.x1 + 0.01, cbar_y, cbar_width, bar_height])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal"
    )
    cb.set_ticks([0, 50, 100])
    cb.ax.tick_params(labelsize=9)
    fig.text(pos.x1 + 0.07, cbar_y + bar_height/2, label, va='center', fontsize=10)

# -----------------------------
# 4) Histogramm rechts
# -----------------------------
ax4 = fig.add_subplot(gs[3:5, 1])
colors = {"Real": "blue", "Expon": "orange", "Uniform": "gray"}
bin_edges = np.linspace(0, 1, 25)
ax4.hist(results["proximity_real"], bins=bin_edges, density=True, alpha=0.5, color=colors["Real"], label="Real")
ax4.hist(results["proximity_expon_all"], bins=bin_edges, density=True, alpha=0.3, color=colors["Expon"], label="Exponential Null")
ax4.hist(results["proximity_uniform_all"], bins=bin_edges, density=True, alpha=0.3, color=colors["Uniform"], label="Uniform Null")
sns.kdeplot(results["proximity_real"], bw_adjust=0.4, color=colors["Real"], linewidth=2, ax=ax4)
sns.kdeplot(results["proximity_expon_all"], bw_adjust=0.4, color=colors["Expon"], linewidth=2, ax=ax4)
sns.kdeplot(results["proximity_uniform_all"], bw_adjust=0.4, color=colors["Uniform"], linewidth=2, ax=ax4)
ax4.axvline(results["results"]["median_real"], color=colors["Real"], linestyle="--", linewidth=2, label="Median Real")
ax4.axvline(results["results"]["median_expon"], color=colors["Expon"], linestyle="--", linewidth=2, label="Median Expon")
ax4.axvline(results["results"]["quant95_expon"], color=colors["Expon"], linestyle=":", linewidth=2, label="95%-Quantil Expon")
ax4.axvline(results["results"]["median_uniform"], color=colors["Uniform"], linestyle="--", linewidth=2, label="Median Uniform")
ax4.axvline(results["results"]["quant95_uniform"], color=colors["Uniform"], linestyle=":", linewidth=2, label="95%-Quantil Uniform")
ax4.set_xlim(0, 1)
ax4.set_xlabel("Proximity-Werte")
ax4.set_ylabel("Dichte")
ax4.set_title("Histogramm: Real vs Nullmodelle\n", fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

target_width = 0.3515294117647057  # gemessene Breite der Matrix + Skala

pos_other = ax4.get_position()
ax4.set_position([pos_other.x0, pos_other.y0, target_width, pos_other.height])

# aktuelle Position auslesen
pos = ax4.get_position()

# leicht nach links verschieben (x0 und x1 kleiner machen)
shift_right = 0.0263   # Feinjustieren!
ax4.set_position([pos.x0 - shift_right, pos.y0, pos.width, pos.height])

# neue Position nach dem Verschieben holen
pos = ax4.get_position()

# -----------------------------
# 5) Beat-Kandidaten
# -----------------------------
ax5 = fig.add_subplot(gs[5:7, 0])
ax5.plot(beat_candidates, scores, label="Proximity Score")
for beat, score in local_maxima:
    ax5.plot(beat, score, 'o', color='green')
top_labels = [f"{beat:.3f}s ({score:.2f})" for beat, score in local_maxima_top5[:4]]
ax5.plot([], [], 'o', color='green', label="Lokale Maxima:\n" + "\n".join(top_labels))
ax5.set_xlabel("Beatdauer (s)")
ax5.set_ylabel("⟨Proximity(IOI/Beat)⟩")
ax5.set_title("Beatfinding: Grundschlag-Kandidaten\n", fontweight='bold')
ax5.grid(True)
ax5.legend()

target_width = 0.3515294117647057  # von oben gemessen

# Linke Seite (Beat-Kandidaten)
pos_left = ax5.get_position()
ax5.set_position([pos_left.x0, pos_left.y0, target_width, pos_left.height])

# Rechte Spalte: Top-Beats
height_ratios = [0.9, 0.9, 0.9]
right_gs = GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs[5:7, 1],
    hspace=0.8, height_ratios=height_ratios
)

n_top_display = min(3, len(local_maxima_top5))
ax_tops = []  # hier speichern wir die Subplots
for idx in range(n_top_display):
    ax_top = fig.add_subplot(right_gs[idx])
    ax_tops.append(ax_top)

    beat, score = local_maxima_top5[idx]
    best_shift, best_rmse, best_beats = compute_beat_rmse(onsets, beat)

    ax_top.axhline(0, color='gray', linewidth=1)
    ax_top.vlines(onsets, -0.2, 0.2, color='black', linewidth=2,
                  label='Original Onsets' if idx==0 else "")
    ax_top.vlines(best_beats, -0.1, 0.1, color='red', linestyle='--',
                  label='Beat Raster' if idx==0 else "")

    title = (f"Beatdauer: {beat:.3f}s | Score: {score:.3f} | "
             f"RMSE: {best_rmse*1000:.1f} ms | Phase-Shift: {best_shift:.3f}s")
    ax_top.set_title(title, fontsize=10)
    ax_top.set_yticks([])
    ax_top.set_xlabel("Zeit (s)")
    if idx == 0:
        ax_top.legend(loc='upper right')

# --- Breite angleichen (nach dem Zeichnen!) ---
for ax_top in ax_tops:
    pos = ax_top.get_position()
    ax_top.set_position([pos.x0 - shift_right, pos.y0, target_width, pos.height])

# Position der rechten Spalte über die Achsen bestimmen
top_ax_pos = ax_tops[0].get_position()       # ganz oben
bottom_ax_pos = ax_tops[-1].get_position()   # ganz unten

# Bounding Box für die Spalte
x0 = top_ax_pos.x0
x1 = top_ax_pos.x1
y0 = bottom_ax_pos.y0
y1 = top_ax_pos.y1

# --- Gemeinsamer Titel für die ganze Spalte ---
# Statt y1 + 0.02 -> etwas weiter unten, wie ax.set_title()
fig.text(
    (x0 + x1)/2,
    y1 + 0.01,    # etwas tiefer, damit es wie ein normaler Plot-Titel wirkt
    "Top-Beats",
    ha="center", va="bottom",
    fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.show()
