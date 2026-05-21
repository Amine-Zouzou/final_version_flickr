"""
Match viewer — visualise les correspondances entre 2 images.

Couleurs : vert = inlier, rouge = outlier.

Usage CLI :
    python -m src.visualization.match_viewer img1.jpg img2.jpg
    python -m src.visualization.match_viewer img1.jpg img2.jpg --output result.png --max-matches 50 --only-inliers --open

Usage Python :
    from src.visualization.match_viewer import MatchViewer
    viewer = MatchViewer()
    out = viewer.render("img1.jpg", "img2.jpg", output_path="result.png")
    print(out["stats"])
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from src.matching.sift import SiftMatcher, MatchResult
import logging
logger = logging.getLogger(__name__)

COLOR_INLIER  = (50,  200,  50)
COLOR_OUTLIER = (50,   50, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class MatchViewer:
    def __init__(
        self,
        cfg: dict | None = None,
        max_matches: int | None = 100,
        only_inliers: bool = False,
        draw_keypoints: bool = True,
        line_thickness: int = 1,
    ):
        self.matcher        = SiftMatcher(cfg)
        self.max_matches    = max_matches
        self.only_inliers   = only_inliers
        self.draw_keypoints = draw_keypoints
        self.thickness      = line_thickness

    def _load_color(self, path: str) -> np.ndarray | None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Impossible de lire : {path}")
            return None
        h, w = img.shape[:2]
        if max(h, w) > 1400:
            s = 1400 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        return img

    def _draw(self, img1: np.ndarray, img2: np.ndarray, result: MatchResult) -> np.ndarray:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1]      = img1
        canvas[:h2, w1:w1+w2] = img2

        # Inlier set
        inlier_set = set()
        if result.inliers_mask is not None and len(result.inliers_mask):
            for i, k in enumerate(result.inliers_mask.ravel()):
                if k: inlier_set.add(i)

        # Sélection matches
        if self.only_inliers:
            matches = result.inlier_matches
            flags   = [True] * len(matches)
        else:
            matches = result.good_matches
            flags   = [i in inlier_set for i in range(len(matches))]

        # Limite + priorité inliers
        if self.max_matches and len(matches) > self.max_matches:
            in_idx  = [i for i, v in enumerate(flags) if v]
            out_idx = [i for i, v in enumerate(flags) if not v]
            keep    = in_idx + out_idx[:max(0, self.max_matches - len(in_idx))]
            matches = [matches[i] for i in keep]
            flags   = [flags[i]   for i in keep]

        for m, is_in in zip(matches, flags):
            pt1 = tuple(map(int, result.kp1[m.queryIdx].pt))
            pt2 = (int(result.kp2[m.trainIdx].pt[0]) + w1,
                   int(result.kp2[m.trainIdx].pt[1]))
            col = COLOR_INLIER if is_in else COLOR_OUTLIER

            if self.draw_keypoints:
                cv2.circle(canvas, pt1, 4, col, -1)
                cv2.circle(canvas, pt2, 4, col, -1)

            overlay = canvas.copy()
            cv2.line(overlay, pt1, pt2, col, self.thickness)
            alpha = 0.9 if is_in else 0.4
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # Barre stats
        bar = np.zeros((36, canvas.shape[1], 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        cv2.line(bar, (w1, 0), (w1, 36), (80, 80, 80), 1)
        status = "VALID" if result.is_valid else "INVALID"
        col    = COLOR_INLIER if result.is_valid else COLOR_OUTLIER
        txt = (f"{status}  |  good={len(result.good_matches)}  "
               f"inliers={result.best_inliers}  ratio={result.best_inlier_ratio:.2f}  "
               f"model={result.best_model}  score={result.edge_score:.1f}")
        cv2.putText(bar, txt, (12, 24), FONT, 0.50, col, 1, cv2.LINE_AA)
        return np.vstack([bar, canvas])

    def render(self, path1: str, path2: str, output_path: str | None = None) -> dict:
        result = self.matcher.match(path1, path2)
        if result is None:
            raise ValueError(f"Impossible de lire une image : {path1}, {path2}")
        img1 = self._load_color(path1)
        img2 = self._load_color(path2)
        if img1 is None or img2 is None:
            raise ValueError("Impossible de charger les images couleur")
        vis = self._draw(img1, img2, result)
        if output_path is None:
            output_path = f"match_{Path(path1).stem}_vs_{Path(path2).stem}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis)
        logger.info(f"Sauvé : {output_path}")
        return {"output_path": output_path, "result": result, "stats": result.to_dict()}

    def render_to_base64(self, path1: str, path2: str) -> tuple[str, dict]:
        result = self.matcher.match(path1, path2)
        if result is None:
            raise ValueError(f"Impossible de matcher : {path1}, {path2}")
        img1 = self._load_color(path1)
        img2 = self._load_color(path2)
        if img1 is None or img2 is None:
            raise ValueError("Impossible de charger les images couleur")
        vis = self._draw(img1, img2, result)
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return base64.b64encode(buf).decode("utf-8"), result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Visualise les correspondances entre 2 images")
    parser.add_argument("img1")
    parser.add_argument("img2")
    parser.add_argument("--output",      "-o",  default=None)
    parser.add_argument("--max-matches", "-n",  type=int, default=100)
    parser.add_argument("--only-inliers",        action="store_true")
    parser.add_argument("--no-keypoints",        action="store_true")
    parser.add_argument("--thickness",   "-t",  type=int, default=1)
    parser.add_argument("--open",                action="store_true")
    args = parser.parse_args()

    viewer = MatchViewer(
        max_matches=args.max_matches,
        only_inliers=args.only_inliers,
        draw_keypoints=not args.no_keypoints,
        line_thickness=args.thickness,
    )
    out = viewer.render(args.img1, args.img2, args.output)
    print(json.dumps(out["stats"], indent=2))
    print(f"\nSauvé : {out['output_path']}")

    if args.open:
        import subprocess, sys
        cmd = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([cmd, out["output_path"]])


if __name__ == "__main__":
    main()