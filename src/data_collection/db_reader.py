"""
Reads photos from the flico PostgreSQL database for a given geo_cluster_id.
Requires EPFL VPN + .env file at FLICO_ROOT with DB credentials.
"""
import os
import sys
import requests
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

FLICO_ROOT = os.getenv("FLICO_ROOT", "/Users/aminezouzou/Desktop/EPFL/BA6/flico")
if FLICO_ROOT not in sys.path:
    sys.path.insert(0, FLICO_ROOT)

from dotenv import load_dotenv
load_dotenv(Path(FLICO_ROOT) / ".env")

from src.core.db import get_engine  # noqa: E402  (flico)

ALLOWED_URL_COLS = {
    "url_sq", "url_t", "url_s", "url_q", "url_m",
    "url_n", "url_z", "url_c", "url_l", "url_o",
}


def get_photos_in_cluster(cluster_id: int, url_col: str = "url_l") -> pd.DataFrame:
    """Returns a DataFrame of photos belonging to a geo_cluster."""
    if url_col not in ALLOWED_URL_COLS:
        raise ValueError(f"url_col invalide: {url_col}")
    query = text(f"""
        SELECT
            p.id,
            p.owner_nsid,
            p.{url_col}       AS image_url,
            p.latitude,
            p.longitude,
            p.title,
            mlp.geo_cluster_id
        FROM photo p
        JOIN machine_learning_photo mlp
            ON p.owner_nsid = mlp.owner_nsid AND p.id = mlp.id
        WHERE mlp.geo_cluster_id = :cluster_id
          AND p.{url_col} IS NOT NULL
    """)
    engine = get_engine("trainer")
    df = pd.read_sql_query(query, engine, params={"cluster_id": cluster_id})
    logger.info(f"Cluster {cluster_id} : {len(df)} photos trouvées")
    return df


def download_cluster_images(cluster_id: int, output_dir: str, url_col: str = "url_l") -> list[str]:
    """Downloads images for a cluster and returns list of local paths."""
    if url_col not in ALLOWED_URL_COLS:
        raise ValueError(f"url_col invalide: {url_col}")
    df = get_photos_in_cluster(cluster_id, url_col=url_col)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for _, row in df.iterrows():
        fname = f"{row['id']}.jpg"
        fpath = out / fname
        if fpath.exists():
            paths.append(str(fpath))
            continue
        try:
            r = requests.get(row["image_url"], timeout=20)
            if r.status_code == 200:
                fpath.write_bytes(r.content)
                paths.append(str(fpath))
            else:
                logger.warning(f"HTTP {r.status_code} pour {row['id']}")
        except Exception as e:
            logger.warning(f"Échec téléchargement {row['id']}: {e}")

    logger.info(f"Cluster {cluster_id} : {len(paths)}/{len(df)} images téléchargées")
    return paths


def list_geo_clusters(limit: int = 20) -> pd.DataFrame:
    """Returns a sample of available geo_clusters."""
    query = text("""
        SELECT gc.id, gc.avg_latitude, gc.avg_longitude,
               COUNT(mlp.id) AS photo_count
        FROM geo_cluster gc
        JOIN machine_learning_photo mlp ON mlp.geo_cluster_id = gc.id
        GROUP BY gc.id, gc.avg_latitude, gc.avg_longitude
        HAVING COUNT(mlp.id) >= 2
        ORDER BY photo_count DESC
        LIMIT :limit
    """)
    engine = get_engine("trainer")
    return pd.read_sql_query(query, engine, params={"limit": limit})
