import pandas as pd
from pathlib import Path

# === Pfad zur CSV-Datei ===
csv_file = Path("/Users/emilschmiedl/Desktop/bat Saccopteryx bilineata (S_bil).csv")  # Pfad anpassen
output_dir = Path("data/Bat Saccopteryx bilineata (S_bil)")    # Ausgabe-Ordner
output_dir.mkdir(parents=True, exist_ok=True)

# CSV laden
df = pd.read_csv(csv_file, sep=",")  # '\t' falls Tab-separiert, sonst sep=','

# Für jeden einzigartigen Data-Wert eine eigene Datei
for data_value in df["Data"].unique():
    df_subset = df[df["Data"] == data_value].copy()
    
    # === Onset berechnen ===
    onset = [0.1]  # erster Onset bei 0.1
    for ioi in df_subset["IOI"].iloc[:-1]:
        onset.append(onset[-1] + ioi)
    df_subset["Onset"] = onset

    # Datei speichern
    out_file = output_dir / f"{data_value}.csv"
    df_subset.to_csv(out_file, index=False)
    print(f"✅ Datei erstellt: {out_file.name}")
