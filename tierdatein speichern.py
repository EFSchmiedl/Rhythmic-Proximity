import pandas as pd
from pathlib import Path

# === Pfad zur CSV-Datei ===
csv_file = Path("/Users/emilschmiedl/Desktop/Bat Carollia perspicillata (C_persp).csv")  # Pfad anpassen
output_dir = Path("data/Bat Carollia perspicillata (C_persp)")    # Ausgabe-Ordner
output_dir.mkdir(parents=True, exist_ok=True)

# CSV laden
df = pd.read_csv(csv_file, sep=",")  # '\t' falls Tab-separiert, sonst sep=','

# Für jeden einzigartigen Data-Wert eine eigene Datei

# Für jeden einzigartigen Data-Wert eine eigene Datei
for data_value in df["Data"].unique():
    df_subset = df[df["Data"] == data_value].copy()
    
    # === Onset berechnen ===
    onset = [0.1]  # erster Onset bei 0.1
    for ioi in df_subset["IOI"].iloc[:-1]:
        onset.append(onset[-1] + ioi)
    df_subset["onset"] = onset

    # === Zusätzliche Zeile mit weitergeführtem Onset ===
    if len(df_subset) >= 1:
        last_onset = df_subset["onset"].iloc[-1]
        last_ioi = df_subset["IOI"].iloc[-1]

        new_row = {col: None for col in df_subset.columns}  # leere Zeile
        new_row["onset"] = last_onset + last_ioi            # nur onset weiterführen

        df_subset = pd.concat([df_subset, pd.DataFrame([new_row])], ignore_index=True)

    # Datei speichern
    out_file = output_dir / f"{data_value}.csv"
    df_subset.to_csv(out_file, index=False)
    print(f"✅ Datei erstellt: {out_file.name}")