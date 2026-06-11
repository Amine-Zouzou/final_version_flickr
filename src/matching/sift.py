"""
SIFT + RootSIFT matcher — adapté de cluster_same_object.py (flico).
Utilisé pour les photos nettes/modernes.
Pour les vieilles photos dégradées → loftr.py
Pour les peintures → dkm.py (à venir)
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)

DEFAULT = dict(
    sift_nfeatures=4000, resize_max=1400, ratio_test=0.75,
    matches_good_min=12, inliers_min=6, inlier_ratio_min=0.20,
    f_ransac_thresh=3.0, h_ransac_thresh=4.0, ransac_conf=0.999, clahe=True,
    border_fraction=0.07,  # mask this % of each edge → removes postcard borders & watermarks
)


@dataclass
class MatchResult:
    kp1:              list
    kp2:              list
    good_matches:     list
    inliers_mask:     np.ndarray
    inliers_F:        int
    inlier_ratio_F:   float
    inliers_H:        int
    inlier_ratio_H:   float
    best_model:       str
    best_inliers:     int
    best_inlier_ratio: float
    edge_score:       float
    is_valid:         bool
    img1_shape:       tuple
    img2_shape:       tuple

    @property
    def inlier_matches(self) -> list:
        if self.inliers_mask is None or len(self.inliers_mask) == 0:
            return []
        return [m for m, k in zip(self.good_matches, self.inliers_mask.ravel()) if k]

    def to_dict(self) -> dict:
        return dict(
            matches_good=len(self.good_matches),
            inliers_F=self.inliers_F, ratio_F=self.inlier_ratio_F,
            inliers_H=self.inliers_H, ratio_H=self.inlier_ratio_H,
            best_model=self.best_model, best_inliers=self.best_inliers,
            best_inlier_ratio=self.best_inlier_ratio,
            edge_score=self.edge_score, is_valid=int(self.is_valid),
        )


class SiftMatcher:
    def __init__(self, cfg: dict | None = None):
        c = {**DEFAULT, **(cfg or {})}
        self.nfeatures        = c["sift_nfeatures"]
        self.resize_max       = c["resize_max"]
        self.ratio_test       = c["ratio_test"]
        self.matches_good_min = c["matches_good_min"]
        self.inliers_min      = c["inliers_min"]
        self.inlier_ratio_min = c["inlier_ratio_min"]
        self.f_thresh         = c["f_ransac_thresh"]
        self.h_thresh         = c["h_ransac_thresh"]
        self.ransac_conf      = c["ransac_conf"]
        self.clahe            = c["clahe"]
        self.border_fraction  = c["border_fraction"]
        self._det  = cv2.SIFT_create(nfeatures=self.nfeatures) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=self.nfeatures)
        self._type = "float" if "sift" in self._det.__class__.__name__.lower() else "binary"
        self._mat  = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80)) if self._type == "float" else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _load(self, path: str) -> tuple[np.ndarray | None, tuple]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, (0, 0)
        h, w = img.shape[:2]
        if w > 1.7 * h:  # stereocard: crop to left half
            img = img[:, :w // 2]
            w = w // 2
        if max(h, w) > self.resize_max:
            s = self.resize_max / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        if self.clahe:
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        return img, (h, w)

    def _border_mask(self, h: int, w: int) -> np.ndarray:
        """uint8 mask: 255 in the valid centre, 0 on the border margins.
        Prevents SIFT from picking up postcard frames, watermarks, oval vignettes."""
        mask = np.zeros((h, w), dtype=np.uint8)
        bh = max(1, int(h * self.border_fraction))
        bw = max(1, int(w * self.border_fraction))
        mask[bh:h - bh, bw:w - bw] = 255
        return mask

    @staticmethod
    def _rootsift(d: np.ndarray) -> np.ndarray:
        d = d.astype(np.float32)
        d /= (d.sum(axis=1, keepdims=True) + 1e-12)
        return np.sqrt(d)

    def _knn(self, d1, d2):
        if d1 is None or d2 is None: return []
        if self._type == "float":
            d1, d2 = self._rootsift(d1), self._rootsift(d2)
        try:
            return self._mat.knnMatch(d1, d2, k=2)
        except cv2.error:
            return []

    def _ratio(self, knn):
        return [m for p in knn if len(p)==2 for m, n in [p] if m.distance < self.ratio_test * n.distance]

    def _ransac_F(self, kp1, kp2, good):
        if len(good) < 9:   # ← 9 au lieu de 8
            return 0, 0.0, np.array([])
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        try:
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, self.f_thresh, self.ransac_conf)
        except cv2.error:
            return 0, 0.0, np.array([])
        if mask is None:
            return 0, 0.0, np.array([])
        n = int(mask.sum())
        return n, n/max(1, len(good)), mask

    def _ransac_H(self, kp1, kp2, good):
        if len(good) < 8: return 0, 0.0, np.array([])
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, self.h_thresh)
        if mask is None: return 0, 0.0, np.array([])
        n = int(mask.sum()); return n, n/max(1,len(good)), mask

    def match(self, path1: str, path2: str) -> MatchResult | None:
        img1, s1 = self._load(path1)
        img2, s2 = self._load(path2)
        if img1 is None or img2 is None: return None
        kp1, d1 = self._det.detectAndCompute(img1, self._border_mask(*s1))
        kp2, d2 = self._det.detectAndCompute(img2, self._border_mask(*s2))
        good = self._ratio(self._knn(d1, d2))
        inF, rF, mF = self._ransac_F(kp1, kp2, good)
        inH, rH, mH = self._ransac_H(kp1, kp2, good)
        if inF >= inH:
            bi, br, bm, bmask = inF, rF, "F", mF
        else:
            bi, br, bm, bmask = inH, rH, "H", mH
        score = float(bi * br * np.log1p(len(good)))
        valid = len(good) >= self.matches_good_min and bi >= self.inliers_min and br >= self.inlier_ratio_min
        logger.info(f"SIFT {path1.split('/')[-1]} ↔ {path2.split('/')[-1]} | good={len(good)} inliers={bi} ratio={br:.2f} model={bm} valid={valid}")
        return MatchResult(kp1=kp1, kp2=kp2, good_matches=good, inliers_mask=bmask,
                           inliers_F=inF, inlier_ratio_F=rF, inliers_H=inH, inlier_ratio_H=rH,
                           best_model=bm, best_inliers=bi, best_inlier_ratio=br,
                           edge_score=score, is_valid=valid, img1_shape=s1, img2_shape=s2)