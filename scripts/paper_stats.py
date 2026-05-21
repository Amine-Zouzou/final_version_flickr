"""
paper_stats.py — extrait les statistiques pour le paper à partir des outputs
de la pipeline (groups.csv, edges.csv).

Usage:
    python paper_stats.py --groups results/groups.csv --edges results/edges.csv
"""

import argparse
import time
import numpy as np
import pandas as pd


def compute_stats(groups_path: str, edges_path: str):
    groups = pd.read_csv(groups_path)
    edges  = pd.read_csv(edges_path)

    print("\n" + "="*60)
    print("DATASET STATS")
    print("="*60)
    total_images = len(groups)
    print(f"Total images processed     : {total_images}")

    print("\n" + "="*60)
    print("GROUP STATS")
    print("="*60)
    group_sizes = groups.groupby("group_id")["photo"].count()
    n_groups        = len(group_sizes)
    n_singletons    = (group_sizes == 1).sum()
    n_real_groups   = (group_sizes > 1).sum()
    mean_size       = group_sizes[group_sizes > 1].mean()
    median_size     = group_sizes[group_sizes > 1].median()
    max_size        = group_sizes.max()

    print(f"Total groups               : {n_groups}")
    print(f"Singleton groups (size=1)  : {n_singletons}  ({100*n_singletons/n_groups:.1f}%)")
    print(f"Non-singleton groups       : {n_real_groups} ({100*n_real_groups/n_groups:.1f}%)")
    print(f"Mean group size (non-sing) : {mean_size:.2f}")
    print(f"Median group size          : {median_size:.1f}")
    print(f"Largest group              : {max_size} images")

    print("\nGroup size distribution:")
    bins = [1, 2, 3, 5, 10, 20, 50, 9999]
    labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        count = ((group_sizes >= lo) & (group_sizes < hi)).sum()
        print(f"  size {label:>5} : {count} groups")

    print("\n" + "="*60)
    print("EDGE STATS")
    print("="*60)
    total_edges   = len(edges)
    valid_edges   = edges["is_valid"].sum()
    invalid_edges = total_edges - valid_edges
    print(f"Total candidate edges      : {total_edges}")
    print(f"Valid edges                : {int(valid_edges)}  ({100*valid_edges/total_edges:.1f}%)")
    print(f"Rejected edges             : {int(invalid_edges)} ({100*invalid_edges/total_edges:.1f}%)")

    valid = edges[edges["is_valid"] == 1]
    print(f"\nAmong valid edges:")
    print(f"  Mean inliers             : {valid['best_inliers'].mean():.1f}")
    print(f"  Mean inlier ratio        : {valid['best_inlier_ratio'].mean():.3f}")
    print(f"  Mean edge score          : {valid['edge_score'].mean():.2f}")
    print(f"  Best model F / H         : F={( valid['best_model']=='F').sum()} / H={(valid['best_model']=='H').sum()}")

    print("\n" + "="*60)
    print("LOCALISATION WORKLOAD REDUCTION")
    print("="*60)
    n_central = n_real_groups + n_singletons  # one central per group
    reduction = 1 - n_central / total_images
    print(f"Images before grouping     : {total_images}")
    print(f"Central images (output)    : {n_central}")
    print(f"Workload reduction         : {100*reduction:.1f}%  (factor {total_images/n_central:.1f}x)")

    print("\n" + "="*60)
    print("COPY-PASTE FOR PAPER (fill in [XX] placeholders)")
    print("="*60)
    print(f"  total images   = {total_images}")
    print(f"  total groups   = {n_groups}")
    print(f"  non-singletons = {n_real_groups}")
    print(f"  mean size      = {mean_size:.1f}")
    print(f"  max size       = {max_size}")
    print(f"  valid edges    = {int(valid_edges)} / {total_edges} ({100*valid_edges/total_edges:.1f}%)")
    print(f"  reduction      = {100*reduction:.1f}% (factor {total_images/n_central:.1f}x)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True, help="Path to groups.csv")
    parser.add_argument("--edges",  required=True, help="Path to edges.csv")
    args = parser.parse_args()
    compute_stats(args.groups, args.edges)


if __name__ == "__main__":
    main()