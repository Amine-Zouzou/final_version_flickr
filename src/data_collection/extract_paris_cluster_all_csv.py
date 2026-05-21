import os
import pandas as pd
from tqdm import tqdm

METADATA_DIR = "../metadata"
OUTPUT_CSV = "cluster_paris.csv"

QUERY = "Tour Eiffel"

results = []

files = [f for f in os.listdir(METADATA_DIR) if f.endswith(".csv")]

print("CSV trouvés :", len(files))

for file in tqdm(files):

    path = os.path.join(METADATA_DIR, file)

    try:
        df = pd.read_csv(path)

        mask = (
            df["title"].astype(str).str.contains(QUERY, case=False, na=False) |
            df["description"].astype(str).str.contains(QUERY, case=False, na=False) |
            df["tags"].astype(str).str.contains(QUERY, case=False, na=False)
        )

        subset = df[mask].copy()

        if len(subset) > 0:
            subset["source_dataset"] = file
            results.append(subset)

    except Exception as e:
        print("Erreur avec :", file)

if len(results) == 0:
    print("Aucune image trouvée")
else:

    combined = pd.concat(results)

    print("Images trouvées :", len(combined))

    combined.to_csv(OUTPUT_CSV, index=False)

    print("Sauvé dans :", OUTPUT_CSV)