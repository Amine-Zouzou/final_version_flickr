"""
Classifier de source — détecte automatiquement le type d'une image.
Utilise CLIP zero-shot pour classer en : photo, painting, scan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

PROMPTS: dict[str, list[str]] = {
    "photo": [
        "a real photograph taken with a camera",
        "a photographic image with realistic details and photographic grain",
        "a black and white photo taken by a photographer",
        "a historical photo with silver halide grain",
        "a documentary photograph of a real scene",
    ],
    "painting": [
        "a hand-drawn illustration or engraving",
        "a printed illustration from a newspaper or book",
        "an artistic drawing with visible brushstrokes or ink lines",
        "a lithograph or etching artwork",
        "a poster or illustrated artwork not taken by a camera",
    ],
    "scan": [
        "a scanned document with text and printed content",
        "a digitized page from a book or archive",
        "a scanned postcard with handwriting or printed text",
        "a document scan with visible paper texture and margins",
        "an archival scan of a printed or typed page",
    ],
}

CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class ClassifyResult:
    source_type: str
    confidence:  float
    scores:      dict
    image_path:  str | None = None

    @property
    def is_photo(self) -> bool:
        return self.source_type == "photo"

    @property
    def is_painting(self) -> bool:
        return self.source_type == "painting"

    @property
    def is_scan(self) -> bool:
        return self.source_type == "scan"

    def recommended_matcher(self) -> str:
        return "sift" if self.is_photo else "loftr"


class SourceClassifier:

    def __init__(self, model_name: str = CLIP_MODEL, threshold: float = 0.40):
        self.threshold = threshold
        self.device = (
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        self._model     = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._text_features, self._prompt_labels = self._encode_prompts()

    def _encode_prompts(self) -> tuple[torch.Tensor, list[str]]:
        all_prompts, all_labels = [], []
        for label, prompts in PROMPTS.items():
            for p in prompts:
                all_prompts.append(p)
                all_labels.append(label)
        inputs = self._processor(text=all_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out        = self._model.text_model(**inputs)
            pooled     = out.pooler_output
            text_feats = self._model.text_projection(pooled)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats, all_labels

    def _encode_image(self, image_path: str) -> torch.Tensor:
        img    = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out       = self._model.vision_model(**inputs)
            pooled    = out.pooler_output
            img_feats = self._model.visual_projection(pooled)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats

    def classify(self, image_path: str) -> ClassifyResult:
        img_feats = self._encode_image(image_path)
        sims = (img_feats @ self._text_features.T).squeeze(0).cpu().float()
        class_scores: dict[str, float] = {}
        for label in PROMPTS:
            indices = [i for i, l in enumerate(self._prompt_labels) if l == label]
            class_scores[label] = float(sims[indices].mean())
        scores_tensor = torch.tensor(list(class_scores.values()))
        probs         = torch.softmax(scores_tensor * 10, dim=0)
        probs_dict    = {k: float(v) for k, v in zip(class_scores.keys(), probs)}
        best_class    = max(probs_dict, key=probs_dict.get)
        confidence    = probs_dict[best_class]
        if confidence < self.threshold:
            best_class = "photo"
        return ClassifyResult(
            source_type=best_class,
            confidence=confidence,
            scores=probs_dict,
            image_path=image_path,
        )

    def classify_batch(self, image_paths: list[str]) -> list[ClassifyResult]:
        results = []
        for path in image_paths:
            try:
                results.append(self.classify(path))
            except Exception as e:
                logger.warning(f"Erreur sur {path} : {e} → fallback photo")
                results.append(ClassifyResult(source_type="photo", confidence=0.0, scores={}, image_path=path))
        return results
