from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cluster import Clusterer

logger = logging.getLogger(__name__)


def _find_central_photos(groups_df: pd.DataFrame, edges_df: pd.DataFrame) -> set[str]:
    """
    Pour chaque groupe, retourne le nom de fichier de la photo avec le plus
    haut edge score moyen vers ses voisins (= photo centrale du groupe).
    """
    valid = edges_df[edges_df["is_valid"] == 1]

    node_scores: dict[str, list[float]] = {}
    for _, row in valid.iterrows():
        node_scores.setdefault(row["a"], []).append(row["edge_score"])
        node_scores.setdefault(row["b"], []).append(row["edge_score"])

    avg_score = {node: np.mean(vals) for node, vals in node_scores.items()}

    central = set()
    for _, group in groups_df.groupby("group_id"):
        best = max(group["photo"], key=lambda p: avg_score.get(p, 0.0))
        central.add(best)

    return central


def grouping(
    clusterer: Clusterer,
    df: pd.DataFrame,
    output_dir: str | None = None,
    url_col: str = "image_url",
    id_col: str = "id",
    cluster_col: str = "geo_cluster_id",
    embedding_col: str | None = "sig_lip_vect_n",
) -> pd.DataFrame:
    """
    Pour chaque geo_cluster_id présent dans df, télécharge les images dans
    un dossier temporaire (auto-supprimé), lance la pipeline, et retourne df
    enrichi avec group_id ("{geo_cluster_id}_{local_gid}"), group_size,
    et is_central (True pour la photo centrale de chaque groupe).

    output_dir=None (défaut) : aucun fichier écrit sur disque.
    Si embedding_col est fourni, DINOv2 n'est pas chargé.
    """
    cluster_ids = df[cluster_col].unique()
    logger.info(
        f"run_on_dataframe : {len(cluster_ids)} clusters | "
        f"embeddings={'fournis ('+embedding_col+')' if embedding_col else 'DINOv2'}"
    )

    mapping_rows = []
    for cid in tqdm(cluster_ids, desc="clusters"):
        sub     = df[df[cluster_col] == cid]
        out_dir = str(Path(output_dir) / f"cluster_{cid:05d}") if output_dir else None

        with tempfile.TemporaryDirectory() as img_dir:
            paths = clusterer._download_images(sub, Path(img_dir), url_col, id_col)
            if len(paths) < 2:
                logger.warning(f"Cluster {cid} : moins de 2 images, ignoré")
                continue

            if embedding_col is not None:
                id_to_emb = dict(zip(sub[id_col].astype(str), sub[embedding_col]))
                emb_list, valid_paths = [], []
                for p in paths:
                    pid = Path(p).stem
                    if pid in id_to_emb:
                        emb_list.append(id_to_emb[pid])
                        valid_paths.append(p)
                if len(valid_paths) < 2:
                    logger.warning(f"Cluster {cid} : moins de 2 embeddings alignés, ignoré")
                    continue
                embs = np.stack(emb_list)
                embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
                groups_df, edges_df = clusterer._run_pipeline(valid_paths, out_dir, embeddings=embs)
            else:
                groups_df, edges_df = clusterer._run_pipeline(paths, out_dir)

        if groups_df.empty:
            continue

        central_photos = _find_central_photos(groups_df, edges_df)

        groups_df = groups_df.copy()
        groups_df[id_col]       = groups_df["photo"].str.replace(r"\.jpe?g$", "", regex=True)
        groups_df["group_id"]   = groups_df["group_id"].apply(lambda g: f"{cid}_{g}")
        groups_df["is_central"] = groups_df["photo"].isin(central_photos)
        mapping_rows.append(groups_df[[id_col, "group_id", "is_central"]])

    if not mapping_rows:
        return df.assign(group_id=pd.NA, is_central=False)

    mapping = pd.concat(mapping_rows, ignore_index=True)
    return df.merge(mapping, on=id_col, how="left")
