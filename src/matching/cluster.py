"""
Pipeline de clustering — regroupe les images du même objet.

Étapes :
1. DINOv2 → embeddings + top-k candidats par similarité cosinus
2. SIFT + RANSAC → matching géométrique et filtrage inliers
3. NetworkX → composantes connexes = groupes

Usage :
    from src.matching.cluster import Clusterer
    clusterer = Clusterer()
    groups_df, edges_df = clusterer.run("image_tour_eiffel/", output_dir="results/")
"""
from __future__ import annotations

import os
import glob
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from transformers import AutoImageProcessor, Dinov2Model

from .sift import SiftMatcher
from .lightglue import LightGlueMatcher
logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    # Embedding model (DINOv2)
    embed_model:        str   = "facebook/dinov2-large"
    # Candidate retrieval
    clip_topk:          int   = 10    # top-k neighbours per image
    clip_mutual:        bool  = False  # no mutual constraint → better recall
    clip_sim_threshold: float = 0.65  # min DINOv2 cosine similarity to attempt SIFT
    # SIFT thresholds — permissif, premier filtre rapide
    matches_good_min:   int   = 20
    inliers_min:        int   = 12
    inlier_ratio_min:   float = 0.30
    # LightGlue thresholds — strict, vérification finale
    lg_matches_good_min:  int   = 30
    lg_inliers_min:       int   = 20
    lg_inlier_ratio_min:  float = 0.40
    max_edges_per_node: int   = 3
    symmetric_check:    bool  = True   # require match to pass in both directions
    resize_max:         int   = 1400

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Clusterer:
    """
    Pipeline complète : images → groupes.

    Args:
        cfg: ClusterConfig ou dict de config
    """

    def __init__(
        self,
        cfg: ClusterConfig | dict | None = None,
    ):
        if isinstance(cfg, dict):
            cfg = ClusterConfig.from_dict(cfg)
        self.cfg = cfg or ClusterConfig()

        self.device = (
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        logger.info(f"Clusterer device: {self.device}")
        self._embed_model     = None  # chargé à la demande via _ensure_dinov2()
        self._embed_processor = None
        self._matcher = SiftMatcher({
            "matches_good_min":  self.cfg.matches_good_min,
            "inliers_min":       self.cfg.inliers_min,
            "inlier_ratio_min":  self.cfg.inlier_ratio_min,
            "resize_max":        self.cfg.resize_max,
        })
        self._verifier = LightGlueMatcher({
            "matches_good_min":  self.cfg.lg_matches_good_min,
            "inliers_min":       self.cfg.lg_inliers_min,
            "inlier_ratio_min":  self.cfg.lg_inlier_ratio_min,
        })

    # ------------------------------------------------------------------ #
    #  DINOv2 embeddings                                                   #
    # ------------------------------------------------------------------ #

    def _load_dinov2(self):
        logger.info(f"Chargement DINOv2 : {self.cfg.embed_model}")
        processor = AutoImageProcessor.from_pretrained(self.cfg.embed_model)
        model     = Dinov2Model.from_pretrained(self.cfg.embed_model).to(self.device).eval()
        return model, processor

    def _ensure_dinov2(self):
        """Charge DINOv2 uniquement si pas déjà en mémoire."""
        if self._embed_model is None:
            self._embed_model, self._embed_processor = self._load_dinov2()

    def _embed(self, path: str) -> np.ndarray | None:
        """Returns L2-normalised DINOv2 CLS token embedding for one image."""
        self._ensure_dinov2()
        try:
            img    = Image.open(path).convert("RGB")
            inputs = self._embed_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out   = self._embed_model(**inputs)
                feats = out.last_hidden_state[:, 0, :]   # CLS token
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()[0]
        except Exception as e:
            logger.warning(f"DINOv2 fail {Path(path).name}: {e}")
            return None

    def _build_candidates(self, embeddings: np.ndarray) -> list[tuple[int, int]]:
        n   = embeddings.shape[0]
        sim = embeddings @ embeddings.T

        neighbors: dict[int, set[int]] = {}
        for i in range(n):
            # argsort gives descending similarity order; apply threshold then topk
            idx = np.argsort(-sim[i])
            top = [j for j in idx if j != i and sim[i, j] >= self.cfg.clip_sim_threshold]
            neighbors[i] = set(top[:self.cfg.clip_topk])

        pairs: set[tuple[int, int]] = set()
        for i in range(n):
            for j in neighbors[i]:
                if not self.cfg.clip_mutual or i in neighbors[j]:
                    a, b = sorted((i, j))
                    pairs.add((a, b))

        return sorted(pairs)

    # ------------------------------------------------------------------ #
    #  Edge pruning                                                        #
    # ------------------------------------------------------------------ #

    def _prune_edges(self, valid_edges: pd.DataFrame) -> pd.DataFrame:
        """Keeps only max_edges_per_node best edges per node."""
        by_node: dict[str, list] = defaultdict(list)
        for _, row in valid_edges.iterrows():
            by_node[row["a"]].append(row)
            by_node[row["b"]].append(row)

        kept: set[tuple] = set()
        for _, rows in by_node.items():
            rows_sorted = sorted(
                rows,
                key=lambda r: (r["edge_score"], r["best_inliers"], r["matches_good"]),
                reverse=True,
            )
            for r in rows_sorted[:self.cfg.max_edges_per_node]:
                kept.add(tuple(sorted((r["a"], r["b"]))))

        return valid_edges[
            valid_edges.apply(
                lambda r: tuple(sorted((r["a"], r["b"]))) in kept, axis=1
            )
        ].copy()

    # ------------------------------------------------------------------ #
    #  Core pipeline                                                       #
    # ------------------------------------------------------------------ #

    def _download_images(
        self,
        df: "pd.DataFrame",
        cache,
        out_dir: "Path",
        url_col: str,
        id_col: str,
    ) -> list[str]:
        """Télécharge ou copie les images d'un DataFrame et retourne les chemins locaux."""
        import shutil
        import requests as _requests
        paths = []
        for _, row in df.iterrows():
            src   = str(row[url_col])
            fname = Path(src).name   # ex: "52123456789_abc123_b.jpg"
            fpath = out_dir / fname
            if not fpath.exists():
                if src.startswith("http"):
                    try:
                        img = cache.get(src)
                        if img:
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG")
                            fpath.write_bytes(buffer.getvalue())
                        else:
                            looger.warning(f"File not in cache: {src}")
                            continue
                        # r = _requests.get(src, timeout=20)

                        # if r.status_code == 200:
                        #     fpath.write_bytes(r.content)
                        # else:
                        #     logger.warning(f"HTTP {r.status_code} pour {row[id_col]}")
                        #     continue


                    except Exception as e:
                        logger.warning(f"Échec téléchargement {row[id_col]}: {e}")
                        continue
                else:
                    local = Path(src)
                    if not local.exists():
                        logger.warning(f"Fichier introuvable : {src}")
                        continue
                    shutil.copy2(local, fpath)
            paths.append(str(fpath))
        return paths

    def _run_pipeline(
        self,
        paths: list[str],
        output_dir: str | None,
        embeddings: "np.ndarray | None" = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Core pipeline: embeddings → candidates → matching → pruning → groups.
        Both run() and run_on_paths() delegate here.

        Args:
            embeddings: embeddings pré-calculés (N × D), alignés sur paths.
                        Si None, DINOv2 est utilisé.
        """
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if len(paths) < 2:
            logger.warning(f"Pas assez d'images valides ({len(paths)})")
            return pd.DataFrame(), pd.DataFrame()

        if embeddings is not None:
            # Embeddings fournis — on saute DINOv2
            kept_paths = paths
            embs = embeddings
        else:
            # 1. DINOv2 embeddings
            raw_embs, kept_paths = [], []
            for path in tqdm(paths, desc="DINOv2"):
                emb = self._embed(path)
                if emb is not None:
                    raw_embs.append(emb)
                    kept_paths.append(path)

            if len(kept_paths) < 2:
                logger.warning("Pas assez d'embeddings valides")
                return pd.DataFrame(), pd.DataFrame()

            embs = np.stack(raw_embs, axis=0)

        # 2. Candidate pairs
        candidate_pairs = self._build_candidates(embs)
        logger.info(f"Paires candidates : {len(candidate_pairs)}")

        # 3. SIFT + RANSAC matching
        edges_rows = []

        for i, j in tqdm(candidate_pairs, desc="matching"):
            path_a, path_b = kept_paths[i], kept_paths[j]
            name_a, name_b = os.path.basename(path_a), os.path.basename(path_b)

            result = self._matcher.match(path_a, path_b)
            if result is None:
                continue

            is_valid = result.is_valid
            # Symmetric check: verify B→A with LightGlue for higher precision
            if self.cfg.symmetric_check and is_valid:
                result_rev = self._verifier.match(path_b, path_a)
                if result_rev is None or not result_rev.is_valid:
                    logger.info(f"LightGlue symétrie échouée : {name_a} ↔ {name_b} — rejeté")
                    is_valid = False

            edges_rows.append({
                "a":                 name_a,
                "b":                 name_b,
                "matches_good":      len(result.good_matches),
                "inliers_F":         result.inliers_F,
                "ratio_F":           result.inlier_ratio_F,
                "inliers_H":         result.inliers_H,
                "ratio_H":           result.inlier_ratio_H,
                "best_model":        result.best_model,
                "best_inliers":      result.best_inliers,
                "best_inlier_ratio": result.best_inlier_ratio,
                "edge_score":        result.edge_score,
                "is_valid":          int(is_valid),
            })

        if not edges_rows:
            logger.warning("Aucun edge trouvé")
            return pd.DataFrame(), pd.DataFrame()

        # 5. Filter + prune
        edges_df = pd.DataFrame(edges_rows).sort_values(
            ["is_valid", "edge_score", "best_inliers", "matches_good"],
            ascending=[False, False, False, False],
        )
        if output_dir is not None:
            edges_df.to_csv(os.path.join(output_dir, "edges.csv"), index=False)
        logger.info(f"Edges : {len(edges_df)} total | {edges_df['is_valid'].sum()} valides")

        valid_edges = edges_df[edges_df["is_valid"] == 1].copy()
        pruned      = self._prune_edges(valid_edges)
        if output_dir is not None:
            pruned.to_csv(os.path.join(output_dir, "edges_pruned.csv"), index=False)

        # 6. Graph → groups
        G = nx.Graph()
        for p in kept_paths:
            G.add_node(os.path.basename(p))
        for _, row in pruned.iterrows():
            G.add_edge(row["a"], row["b"],
                       inliers=row["best_inliers"],
                       edge_score=row["edge_score"])

        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        out   = []
        for gid, comp in enumerate(comps, start=1):
            for name in sorted(comp):
                out.append({"photo": name, "group_id": gid, "group_size": len(comp)})

        groups_df = pd.DataFrame(out).sort_values(
            ["group_size", "group_id"], ascending=[False, True]
        )
        if output_dir is not None:
            groups_df.to_csv(os.path.join(output_dir, "groups.csv"), index=False)
        logger.info(f"Groupes : {len(comps)} | top sizes : {[len(c) for c in comps[:10]]}")

        return groups_df, edges_df

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_on_paths(
        self,
        paths: list[str],
        output_dir: str = "results/",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run pipeline on an explicit list of image paths."""
        valid_paths = [p for p in paths if Path(p).exists()]
        missing     = len(paths) - len(valid_paths)
        if missing:
            logger.warning(f"{missing} images introuvables — ignorées")
        logger.info(f"run_on_paths : {len(valid_paths)} images")
        return self._run_pipeline(valid_paths, output_dir)

    def run(
        self,
        image_dir: str,
        output_dir: str = "results/",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run pipeline on all images in a directory.

        Args:
            image_dir:  directory containing images
            output_dir: output directory for edges.csv and groups.csv
        """
        paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        paths = [p for p in paths if p.lower().endswith(
            (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        )]
        logger.info(f"Images trouvées : {len(paths)}")
        if len(paths) < 2:
            raise ValueError(f"Pas assez d'images dans {image_dir}")
        return self._run_pipeline(paths, output_dir)
