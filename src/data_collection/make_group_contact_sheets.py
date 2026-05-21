import os
import math
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

GROUPS_CSV = "groups.csv"
IMAGES_DIR = "images_paris"   # ou images_test
OUTPUT_DIR = "group_contact_sheets"

THUMB_SIZE = (220, 220)
PADDING = 20
HEADER_H = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(GROUPS_CSV)

# ignorer les groupes de taille 1 si tu veux
grouped = df[df["group_size"] >= 2].groupby("group_id")

for group_id, group_df in grouped:
    photos = group_df["photo"].tolist()
    n = len(photos)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    sheet_w = cols * THUMB_SIZE[0] + (cols + 1) * PADDING
    sheet_h = rows * THUMB_SIZE[1] + (rows + 1) * PADDING + HEADER_H

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((PADDING, 10), f"Group {group_id} - {n} images", fill="black")

    for idx, photo in enumerate(photos):
        path = os.path.join(IMAGES_DIR, photo)
        if not os.path.exists(path):
            continue

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            thumb = ImageOps.pad(img, THUMB_SIZE, color="white")

            r = idx // cols
            c = idx % cols

            x = PADDING + c * (THUMB_SIZE[0] + PADDING)
            y = HEADER_H + PADDING + r * (THUMB_SIZE[1] + PADDING)

            sheet.paste(thumb, (x, y))
        except Exception:
            continue

    out_path = os.path.join(OUTPUT_DIR, f"group_{group_id}.jpg")
    sheet.save(out_path, quality=95)

print(f"Planches sauvegardées dans : {OUTPUT_DIR}")
