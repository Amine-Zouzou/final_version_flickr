import os
import pandas as pd
from tqdm import tqdm

# dossier contenant tous les CSV metadata
METADATA_DIR = "../metadata"

# fichier de sortie
OUTPUT_CSV = "cluster_tour_eiffel.csv"

# mots clés pour détecter la Tour Eiffel
KEYWORDS = [
    "Tour Eiffel",
    "Eiffel Tower",
    "La Tour Eiffel",
]

results = []

files = [f for f in os.listdir(METADATA_DIR) if f.endswith(".csv")]

print("CSV trouvés :", len(files))

for file in tqdm(files):
    path = os.path.join(METADATA_DIR, file)

    try:
        df = pd.read_csv(path)

        mask = False
        for kw in KEYWORDS:
            mask = (
                mask
                | df["title"].astype(str).str.contains(kw, case=False, na=False)
                | df["description"].astype(str).str.contains(kw, case=False, na=False)
                | df["tags"].astype(str).str.contains(kw, case=False, na=False)
            )

        subset = df[mask].copy()

        if len(subset) > 0:
            subset["source_dataset"] = file
            results.append(subset)

    except Exception as e:
        print("Erreur avec :", file, "|", e)

if len(results) == 0:
    print("Aucune image trouvée")
else:
    combined = pd.concat(results, ignore_index=True)

    # supprimer doublons éventuels
    combined = combined.drop_duplicates(subset=["id"])

    print("Images trouvées :", len(combined))

    # optionnel : limiter pour tests
    # combined = combined.sample(n=min(100, len(combined)), random_state=42)

    combined.to_csv(OUTPUT_CSV, index=False)

    print("Sauvé dans :", OUTPUT_CSV)