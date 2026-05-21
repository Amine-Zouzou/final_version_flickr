from __future__ import annotations

import logging
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .cluster import Clusterer
from .lightglue import LightGlueMatcher

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
    df: pd.DataFrame,
    clusterer: Clusterer | None = None,
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
    if clusterer is None:
        clusterer = Clusterer()

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


def export_groups(
    df: pd.DataFrame,
    output_dir: str,
    url_col: str = "image_url",
    id_col: str = "id",
    group_col: str = "group_id",
    central_col: str = "is_central",
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    Exporte les groupes en dossiers depuis le df enrichi par grouping().

    Le df doit contenir : id_col, url_col, group_col, central_col.

    Arborescence produite :
        output_dir/
            group_<group_id>/
                <id>.jpg
                CENTRAL_<id>.jpg   ← photo centrale du groupe

    Args:
        df:              DataFrame sorti de grouping().
        output_dir:      Dossier racine de sortie (créé si inexistant).
        url_col:         Colonne URL Flickr de l'image.
        id_col:          Colonne identifiant unique de la photo.
        group_col:       Colonne group_id (ex. "12_3").
        central_col:     Colonne booléenne photo centrale.
        min_group_size:  Groupes plus petits ignorés.

    Returns:
        DataFrame résumé : group_id | n_ok | n_fail | folder
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # garder uniquement les lignes avec un group_id valide
    sub = df.dropna(subset=[group_col]).copy()
    sub[group_col] = sub[group_col].astype(str)

    # taille de chaque groupe
    sizes = sub.groupby(group_col)[id_col].transform("count")
    sub = sub[sizes >= min_group_size]

    if sub.empty:
        logger.warning("Aucun groupe de taille >= %d — rien à exporter.", min_group_size)
        return pd.DataFrame(columns=["group_id", "n_ok", "n_fail", "folder"])

    groups = sorted(sub[group_col].unique())
    logger.info("export_groups : %d groupes → %s", len(groups), out_root)

    summary = []
    for gid in tqdm(groups, desc="export"):
        rows   = sub[sub[group_col] == gid]
        folder = out_root / f"group_{gid}"
        folder.mkdir(parents=True, exist_ok=True)
        n_ok, n_fail = 0, 0

        for _, row in rows.iterrows():
            photo_id   = str(row[id_col])
            url        = str(row[url_col])
            is_central = bool(row.get(central_col, False))

            ext      = _guess_ext(url)
            filename = f"CENTRAL_{photo_id}{ext}" if is_central else f"{photo_id}{ext}"
            dest     = folder / filename

            if dest.exists():
                n_ok += 1
                continue

            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    dest.write_bytes(r.content)
                    n_ok += 1
                else:
                    logger.warning("HTTP %s pour %s", r.status_code, photo_id)
                    n_fail += 1
            except Exception as e:
                logger.warning("Échec %s : %s", photo_id, e)
                n_fail += 1

        summary.append({"group_id": gid, "n_ok": n_ok, "n_fail": n_fail, "folder": str(folder)})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_root / "export_summary.csv", index=False)
    logger.info("Résumé → %s/export_summary.csv", out_root)
    return summary_df


def _guess_ext(url: str) -> str:
    """Devine l'extension depuis l'URL, fallback .jpg."""
    clean = url.lower().split("?")[0]
    for ext in (".png", ".jpeg", ".tiff", ".tif", ".gif", ".webp"):
        if clean.endswith(ext):
            return ext
    return ".jpg"
