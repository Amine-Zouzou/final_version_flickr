"""
LightGlue matcher — remplace SIFT pour les photos nettes/modernes.
Utilise SuperPoint + LightGlue (state of the art 2024).

Avantages vs SIFT :
- Meilleur sur les changements d'échelle et de point de vue
- Plus robuste sur les vieilles photos
- Fonctionne sur MPS (Apple Silicon)

Usage :
    from src.matching.lightglue_matcher import LightGlueMatcher
    matcher = LightGlueMatcher()
    result = matcher.match("photo1.jpg", "photo2.jpg")
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import logging
from pathlib import Path

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

from .sift import MatchResult

logger = logging.getLogger(__name__)

DEFAULT = dict(
    resize_max=512,
    max_num_keypoints=1024,
    min_confidence=0.0,
    inliers_min=6,
    inlier_ratio_min=0.20,
    matches_good_min=10,
    f_ransac_thresh=3.0,
    ransac_conf=0.999,
)


class LightGlueMatcher:
    """
    SuperPoint + LightGlue matcher.
    Interface identique à SiftMatcher — interchangeable dans matcher_factory.
    """

    def __init__(self, cfg: dict | None = None):
        c = {**DEFAULT, **(cfg or {})}
        self.resize_max         = c["resize_max"]
        self.max_num_keypoints  = c["max_num_keypoints"]
        self.min_confidence     = c["min_confidence"]
        self.inliers_min      = c["inliers_min"]
        self.inlier_ratio_min = c["inlier_ratio_min"]
        self.matches_good_min = c["matches_good_min"]
        self.f_thresh         = c["f_ransac_thresh"]
        self.ransac_conf      = c["ransac_conf"]

        self.device = (
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        logger.info(f"LightGlue device: {self.device}")

        self._extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        self._matcher   = LightGlue(features="superpoint").eval().to(self.device)

    def _load(self, path: str) -> torch.Tensor | None:
        try:
            img = load_image(path).to(self.device)
            # Resize si nécessaire
            h, w = img.shape[-2:]
            if max(h, w) > self.resize_max:
                s = self.resize_max / max(h, w)
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(int(h*s), int(w*s)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            return img
        except Exception as e:
            logger.warning(f"Impossible de charger {path}: {e}")
            return None

    def _ransac_F(self, pts1: np.ndarray, pts2: np.ndarray):
        if len(pts1) < 9:
            return 0, 0.0, np.array([])
        try:
            _, mask = cv2.findFundamentalMat(
                pts1, pts2, cv2.FM_RANSAC, self.f_thresh, self.ransac_conf
            )
        except cv2.error:
            return 0, 0.0, np.array([])
        if mask is None:
            return 0, 0.0, np.array([])
        n = int(mask.sum())
        return n, n / max(1, len(pts1)), mask

    def match(self, path1: str, path2: str) -> MatchResult | None:
        img1 = self._load(path1)
        img2 = self._load(path2)
        if img1 is None or img2 is None:
            return None

        h1, w1 = img1.shape[-2:]
        h2, w2 = img2.shape[-2:]

        with torch.no_grad():
            feats1 = self._extractor.extract(img1)
            feats2 = self._extractor.extract(img2)
            matches_out = self._matcher({"image0": feats1, "image1": feats2})

        # Extraire les correspondances
        feats1, feats2, matches_out = rbd(feats1), rbd(feats2), rbd(matches_out)
        matches   = matches_out["matches"].cpu().numpy()
        scores    = matches_out["matching_scores0"].cpu().numpy()
        kpts1 = feats1["keypoints"].cpu().numpy()   # (M, 2)
        kpts2 = feats2["keypoints"].cpu().numpy()   # (K, 2)

        if len(matches) < self.matches_good_min:
            logger.info(f"LightGlue {Path(path1).name} ↔ {Path(path2).name} | matches={len(matches)} (trop peu)")
            return self._empty_result((h1, w1), (h2, w2))

        pts1 = kpts1[matches[:, 0]]   # (N, 2)
        pts2 = kpts2[matches[:, 1]]   # (N, 2)

        # RANSAC Fundamental Matrix
        inliers, inlier_ratio, mask = self._ransac_F(
            pts1.astype(np.float32),
            pts2.astype(np.float32),
        )

        score = float(inliers * inlier_ratio * np.log1p(len(matches)))
        valid = (
            len(matches) >= self.matches_good_min
            and inliers  >= self.inliers_min
            and inlier_ratio >= self.inlier_ratio_min
        )

        logger.info(
            f"LightGlue {Path(path1).name} ↔ {Path(path2).name} | "
            f"matches={len(matches)} inliers={inliers} ratio={inlier_ratio:.2f} valid={valid}"
        )

        # Convertir en format MatchResult compatible
        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in kpts1]
        kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in kpts2]
        good_matches = [cv2.DMatch(int(m[0]), int(m[1]), 0.0) for m in matches]

        return MatchResult(
            kp1=kp1, kp2=kp2,
            good_matches=good_matches,
            inliers_mask=mask,
            inliers_F=inliers, inlier_ratio_F=inlier_ratio,
            inliers_H=0, inlier_ratio_H=0.0,
            best_model="F",
            best_inliers=inliers,
            best_inlier_ratio=inlier_ratio,
            edge_score=score,
            is_valid=valid,
            img1_shape=(h1, w1),
            img2_shape=(h2, w2),
        )

    def _empty_result(self, s1, s2) -> MatchResult:
        return MatchResult(
            kp1=[], kp2=[], good_matches=[], inliers_mask=np.array([]),
            inliers_F=0, inlier_ratio_F=0.0,
            inliers_H=0, inlier_ratio_H=0.0,
            best_model="F", best_inliers=0, best_inlier_ratio=0.0,
            edge_score=0.0, is_valid=False,
            img1_shape=s1, img2_shape=s2,
        )