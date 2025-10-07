import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
from ipywidgets import interact, FloatSlider, IntSlider

# ---------------------------
# Optimierte Proximity-Funktion
# ---------------------------

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

# ---------------------------
# Plot-Wrapper
# ---------------------------
def plot_proximity_max_freq_gaussian(
    sharpness_base=10.0,
    sharpness_growth=1.0,
    decay=0.5,
    max_freq=6,
    x_max=5.0,
    weight_exponent=1.0,
    freq_sharpness_exp=0.5,
):
    x_vals = np.linspace(1, x_max, 2000)
    y_vals, _ = proximity_max_freq_gaussian(
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

# ---------------------------
# Interaktive Kontrolle
# ---------------------------
interact(
    plot_proximity_max_freq_gaussian,
    sharpness_base=FloatSlider(min=1, max=50, step=1, value=10, description="Sharpness Base"),
    sharpness_growth=FloatSlider(min=0, max=3, step=0.1, value=1, description="Sharpness Growth (k)"),
    decay=FloatSlider(min=0, max=3, step=0.05, value=0.5, description="Decay (mu)"),
    max_freq=IntSlider(min=1, max=10, step=1, value=6, description="Max Frequency"),
    x_max=IntSlider(min=1, max=10, step=1, value=5, description="x_max"),
    weight_exponent=FloatSlider(min=0, max=3, step=0.1, value=1, description="Weight Exponent (f)"),
    freq_sharpness_exp=FloatSlider(min=0, max=3, step=0.1, value=0.5, description="Freq Sharpness Exp (f)"),
)
