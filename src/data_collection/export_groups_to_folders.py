import os
import sys
import shutil
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups",  default="results/groups.csv",  help="CSV des groupes")
    parser.add_argument("--images",  default="images_test/",        help="Dossier images source")
    parser.add_argument("--output",  default="grouped_results/",    help="Dossier de sortie")
    parser.add_argument("--min-size", type=int, default=2,          help="Taille minimale de groupe")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.groups)

    copied, missing = 0, 0
    for _, row in df.iterrows():
        photo      = str(row["photo"])
        group_id   = int(row["group_id"])
        group_size = int(row["group_size"])

        if group_size < args.min_size:
            continue

        src = os.path.join(args.images, photo)
        dst_dir = os.path.join(args.output, f"group_{group_id:03d}")
        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, photo))
            copied += 1
        else:
            missing += 1

    print(f"Exportées : {copied} | Introuvables : {missing} | Dossier : {args.output}")

if __name__ == "__main__":
    main()
