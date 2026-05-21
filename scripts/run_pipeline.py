import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  default="images_test/",  help="Dossier images")
    parser.add_argument("--output",  default="results/",      help="Dossier de sortie")
    parser.add_argument("--topk",    type=int, default=5)
    parser.add_argument("--dev",     action="store_true")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.images)
    paths = sorted([
        str(p) for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    ])
    logger.info(f"Images trouvées : {len(paths)}")

    if args.dev:
        paths = paths[:20]
        logger.info(f"Mode dev : {len(paths)} images")

    from src.matching.cluster import Clusterer
    matcher = Clusterer(cfg={
        "clip_topk": args.topk,
    })

    groups, edges = matcher.run_on_paths(paths, output_dir=args.output)

    if not groups.empty:
        print(groups[groups["group_size"] > 1])
    else:
        print("Aucun groupe trouvé")

    logger.info("Pipeline terminée.")


if __name__ == "__main__":
    main()