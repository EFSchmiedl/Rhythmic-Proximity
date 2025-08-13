import numpy as np
import pandas as pd
import os
from scipy.stats import truncnorm

# === Konfiguration ===
BASE_IOI_LIST = [0.4, 0.5, 0.6, 1.0]
BEAT_COUNTS = [20] 
JITTER_STD_LIST = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
REPEATS = 10
BASE_OUTPUT_DIR = 'data/theoretical_sequences'
INITIAL_ONSET = 0.1

# === Heterochrone Muster ===
HETERO_PATTERNS = {
    "heterochron-1-2": [1, 2],
    "heterochron-1-4": [1, 4],
    "heterochron-1-3-2": [1, 3, 2],
    "heterochron-triplet": [1, 1, 2],
    "heterochron-irregular": None
}

# === Hilfsfunktionen ===
def truncated_jitter(size, std=0.1, limit=0.4):
    a, b = -limit / std, limit / std
    return truncnorm.rvs(a, b, scale=std, size=size)

def save_sequence(iois, base_ioi, num_beats, name, idx=None):
    onsets = np.insert(INITIAL_ONSET + np.cumsum(iois), 0, INITIAL_ONSET)
    iois_with_nan = np.append(iois, np.nan)
    df = pd.DataFrame({'onset': onsets, 'IOI': iois_with_nan, 'label': 'a'})
    dir_path = os.path.join(BASE_OUTPUT_DIR, f"baseIOI-{base_ioi:.2f}", f"{num_beats}beats")
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{name}.csv" if idx is None else f"{name}_{idx:02}.csv"
    df.to_csv(os.path.join(dir_path, filename), index=False)

# === Hauptschleifen ===
for base_ioi in BASE_IOI_LIST:
    for num_beats in BEAT_COUNTS:
        third = num_beats // 3
        mid_idx = num_beats // 2

        # --- Isochron (perfekt)
        iois = np.full(num_beats, base_ioi)
        save_sequence(iois, base_ioi, num_beats, "isochron")

        # --- Isochron + Jitter
        for std in JITTER_STD_LIST:
            label = f"{int(std * 1000):04d}"
            for i in range(REPEATS):
                jitter_iois = base_ioi + truncated_jitter(num_beats, std)
                save_sequence(jitter_iois, base_ioi, num_beats, f"isochron_jitter-{label}", i+1)

        # --- Drittelweise Jitter (start, middle, end)
        for std in JITTER_STD_LIST:
            label = f"{int(std * 1000):04d}"
            for i in range(REPEATS):
                iois = np.full(num_beats, base_ioi)
                iois[:third] += truncated_jitter(third, std=std)
                save_sequence(iois, base_ioi, num_beats, f"jitter_start-{label}", i+1)

                iois = np.full(num_beats, base_ioi)
                mid_start = num_beats // 2 - third // 2
                iois[mid_start:mid_start+third] += truncated_jitter(third, std=std)
                save_sequence(iois, base_ioi, num_beats, f"jitter_middle-{label}", i+1)

                iois = np.full(num_beats, base_ioi)
                iois[-third:] += truncated_jitter(third, std=std)
                save_sequence(iois, base_ioi, num_beats, f"jitter_end-{label}", i+1)

        # --- Single Jitter (start, middle, end)
        for std in JITTER_STD_LIST:
            label = f"{int(std * 1000):04d}"
            for i in range(REPEATS):
                iois = np.full(num_beats, base_ioi)
                iois[0] += truncated_jitter(1, std=std)[0]
                save_sequence(iois, base_ioi, num_beats, f"jitter_start-single-{label}", i+1)

                iois = np.full(num_beats, base_ioi)
                iois[mid_idx] += truncated_jitter(1, std=std)[0]
                save_sequence(iois, base_ioi, num_beats, f"jitter_middle-single-{label}", i+1)

                iois = np.full(num_beats, base_ioi)
                iois[-1] += truncated_jitter(1, std=std)[0]
                save_sequence(iois, base_ioi, num_beats, f"jitter_end-single-{label}", i+1)

        # --- Heterochron
        for name, pattern in HETERO_PATTERNS.items():
            if pattern is None:
                pattern_array = np.random.choice([1, 2, 3], size=num_beats, replace=True) * base_ioi
            else:
                pattern_array = np.resize(np.array(pattern) * base_ioi, num_beats)
            save_sequence(pattern_array, base_ioi, num_beats, name)

            for std in JITTER_STD_LIST:
                label = f"{int(std * 1000):04d}"
                for i in range(REPEATS):
                    jittered = pattern_array + truncated_jitter(num_beats, std)
                    save_sequence(jittered, base_ioi, num_beats, f"{name}_jitter-{label}", i+1)

        # --- Random
        for i in range(REPEATS):
            iois = np.random.uniform(0.2, 1.0, size=num_beats)
            save_sequence(iois, base_ioi, num_beats, "random", i+1)

        # --- Accelerando / Ritardando / Drift
        save_sequence(base_ioi * np.linspace(1.5, 0.3, num_beats), base_ioi, num_beats, "accelerando")
        save_sequence(base_ioi * np.linspace(0.3, 1.5, num_beats), base_ioi, num_beats, "ritardando")
        save_sequence(base_ioi + np.linspace(0, 0.5, num_beats), base_ioi, num_beats, "isochron_drift")

        # --- Tempo-Shifts (alle Jitter-STD)
        base_factors = [1.0, 0.5, 1.5, (1 / 3), (2 / 3), (4 / 3), (5 / 3)]

        for std in JITTER_STD_LIST:
            label = f"{int(std * 1000):04d}"
            for i in range(REPEATS):
                num_sections = np.random.randint(2, 5)
                section_bounds = np.linspace(0, num_beats, num_sections + 1, dtype=int)

                # Beide: Tempo & IOI Jitter
                iois = np.full(num_beats, base_ioi)
                for s in range(num_sections):
                    start, end = section_bounds[s], section_bounds[s + 1]
                    factor = np.random.choice(base_factors) * np.random.uniform(0.95, 1.05)
                    iois[start:end] = base_ioi * factor
                iois += truncated_jitter(num_beats, std=std)
                save_sequence(iois, base_ioi, num_beats, f"tempo_shift_both-jitter-{label}", i+1)

                # Nur Tempo-Jitter
                iois = np.full(num_beats, base_ioi)
                for s in range(num_sections):
                    start, end = section_bounds[s], section_bounds[s + 1]
                    factor = np.random.choice(base_factors) * np.random.uniform(0.95, 1.05)
                    iois[start:end] = base_ioi * factor
                save_sequence(iois, base_ioi, num_beats, f"tempo_shift_onlytempo-jitter-{label}", i+1)

                # Nur IOI-Jitter
                iois = np.full(num_beats, base_ioi)
                for s in range(num_sections):
                    start, end = section_bounds[s], section_bounds[s + 1]
                    factor = np.random.choice(base_factors)
                    iois[start:end] = base_ioi * factor
                iois += truncated_jitter(num_beats, std=std)
                save_sequence(iois, base_ioi, num_beats, f"tempo_shift_onlyioi-jitter-{label}", i+1)

                # Exakte Tempo-Faktoren, kein Jitter (nur 1x, nicht für alle std)
                if std == JITTER_STD_LIST[0]:
                    iois = np.full(num_beats, base_ioi)
                    for s in range(num_sections):
                        start, end = section_bounds[s], section_bounds[s + 1]
                        factor = np.random.choice(base_factors)
                        iois[start:end] = base_ioi * factor
                    save_sequence(iois, base_ioi, num_beats, "tempo_shift_no-jitter", i+1)

print("\n--- ✅ Alle Sequenzen erfolgreich erzeugt ---")