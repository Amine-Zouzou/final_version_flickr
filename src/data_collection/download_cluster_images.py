import os
import pandas as pd
import requests
from tqdm import tqdm

INPUT_CSV = "cluster_tour_eiffel.csv"
OUTPUT_DIR = "image_tour_eiffel"

ID_COL = "id"
URL_COL = "image_url"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
df = df.sample(n=min(50, len(df)), random_state=42)

def get_extension(url: str) -> str:
    url = str(url).lower()
    if ".png" in url:
        return ".png"
    if ".jpeg" in url:
        return ".jpeg"
    return ".jpg"

ok = 0
fail = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="download"):
    photo_id = str(row[ID_COL])
    url = row[URL_COL]

    if pd.isna(url):
        fail += 1
        continue

    ext = get_extension(url)
    out_path = os.path.join(OUTPUT_DIR, f"{photo_id}{ext}")

    if os.path.exists(out_path):
        ok += 1
        continue

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            ok += 1
        else:
            fail += 1
    except Exception:
        fail += 1

print(f"Téléchargées: {ok}")
print(f"Échecs: {fail}")
print(f"Dossier: {OUTPUT_DIR}")