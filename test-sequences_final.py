import numpy as np
import pandas as pd
import os
from scipy.stats import truncnorm

# === Configuration ===
BASE_IOI_LIST = [0.4, 0.5, 0.6, 1.0]
BEAT_COUNTS = [6, 20, 100] 
JITTER_STD_LIST = [0.001, 0.005, 0.01, 0.05, 0.1]
REPEATS = 100
BASE_OUTPUT_DIR = 'data/theoretical_sequences'
INITIAL_ONSET = 0.1

# === Seed for reproducibility ===
np.random.seed(123)
rng = np.random.default_rng()

# === Heterochron Patterns ===
HETERO_PATTERNS = {
    "heterochrony-1-2": [1, 2],
    "heterochrony-1-4": [1, 4],
    "heterochrony-1-3-2": [1, 3, 2],
    "heterochrony-2-3": [2, 3],
    "heterochrony-triplet": [1, 1, 2],
    "heterochrony-irregular": None
}

# === Helper Functions ===
def truncated_jitter(size, std=0.1, limit=0.4):
    a, b = -limit / std, limit / std
    return truncnorm.rvs(a, b, scale=std, size=size)

def save_sequence(iois, base_ioi, num_beats, condition, idx=None):
    onsets = np.insert(INITIAL_ONSET + np.cumsum(iois), 0, INITIAL_ONSET)
    iois_with_nan = np.append(iois, np.nan)

    df = pd.DataFrame({
        'onset': onsets,
        'IOI': iois_with_nan,
        'label': 'a'
    })

    # --- Directory structure ---
    dir_path = os.path.join(
        BASE_OUTPUT_DIR,
        f"baseIOI-{base_ioi:.2f}",
        f"{num_beats}beats",
        condition
    )
    os.makedirs(dir_path, exist_ok=True)

    # --- File naming ---
    if idx is None:
        filename = f"{condition}.csv"
    else:
        filename = f"{condition}_{idx:02d}.csv"

    df.to_csv(os.path.join(dir_path, filename), index=False)


# === Main Loops ===
for base_ioi in BASE_IOI_LIST:
    for num_beats in BEAT_COUNTS:
        third = num_beats // 3
        mid_idx = num_beats // 2

        # --- Isochrony ---
        iois = np.full(num_beats, base_ioi)
        save_sequence(iois, base_ioi, num_beats, "isochrony")

        # --- Isochrony + Jitter
        for std in JITTER_STD_LIST:
            label = f"{int(std * 1000):04d}"
            for i in range(REPEATS):
                jitter_iois = base_ioi + truncated_jitter(num_beats, std)
                save_sequence(jitter_iois, base_ioi, num_beats, f"isochrony_jitter-{label}", i+1)

        # --- Third-wise Jitter (start, middle, end)
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

        # --- Heterochrony
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

        # --- Random-Uniform
        for i in range(REPEATS):
            iois = np.random.uniform(0.1, 1.0, size=num_beats)
            save_sequence(iois, base_ioi, num_beats, "random_uniform", i+1)

        # --- Random-Exponential 
        for i in range(REPEATS):
            iois = rng.exponential(scale=base_ioi, size=num_beats)
            iois = np.clip(iois, 0.1, 2.0)   # Begrenzen auf [0.1, 2.0]
            save_sequence(iois, base_ioi, num_beats, "random_exponential", i+1)

        # --- Accelerando / Ritardando / Drift
        save_sequence(base_ioi * np.linspace(1.5, 0.3, num_beats), base_ioi, num_beats, "accelerando")
        save_sequence(base_ioi * np.linspace(0.3, 1.5, num_beats), base_ioi, num_beats, "ritardando")
        save_sequence(base_ioi + np.linspace(0, 0.5, num_beats), base_ioi, num_beats, "isochrony_drift")

        # --- Tempo-Shifts (all Jitter-STD)
        for std in JITTER_STD_LIST:
            label = f"{int(std * 1000):04d}"

            for i in range(REPEATS):
                
                # Random shift point (not at edges)
                shift_point = np.random.randint(1, num_beats - 1)

                # --- Both: Tempo-Jitter + IOI-Jitter
                iois = np.full(num_beats, base_ioi)
                factor = np.random.uniform(1/3, 3.0) * np.random.uniform(0.95, 1.05)
                iois[shift_point:] = base_ioi * factor
                iois += truncated_jitter(num_beats, std=std)

                save_sequence(iois, base_ioi, num_beats,
                            f"tempo_shift_both-jitter-{label}", i + 1)

                # --- Only Tempo-Jitter (±5%)
                iois = np.full(num_beats, base_ioi)
                factor = np.random.uniform(1/3, 3.0) * np.random.uniform(0.95, 1.05)
                iois[shift_point:] = base_ioi * factor

                save_sequence(iois, base_ioi, num_beats,
                            "tempo_shift_onlytempo-jitter-5percent", i + 1)

                # --- Only IOI-Jitter
                iois = np.full(num_beats, base_ioi)
                factor = np.random.uniform(1/3, 3.0)
                iois[shift_point:] = base_ioi * factor
                iois += truncated_jitter(num_beats, std=std)

                save_sequence(iois, base_ioi, num_beats,
                            f"tempo_shift_onlyioi-jitter-{label}", i + 1)

                # --- Exact Tempo-Shift (no jitter)
                iois = np.full(num_beats, base_ioi)
                factor = np.random.uniform(1/3, 3.0)
                iois[shift_point:] = base_ioi * factor

                save_sequence(iois, base_ioi, num_beats,
                            "tempo_shift_no-jitter", i + 1)
        
print("\n--- All sequences successfully generated ---")