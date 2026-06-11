# Flickr Commons — Image Clustering Pipeline

**Student:** Amine ZOUZOU  
**Supervisor:** Frédéric KAPLAN  
**Academic year:** 2025–2026  
**Lab:** DHLAB, École Polytechnique Fédérale de Lausanne (EPFL)

---

## Project Background

This project develops an automated pipeline to cluster photographs of the same physical object from the [Flickr Commons](https://www.flickr.com/commons) collection. The corpus consists of historical and cultural heritage images contributed by memory institutions worldwide. The goal is to group redundant views of the same monument, artefact, or scene to support downstream 3D reconstruction (photogrammetry) workflows.

## Methodology

The pipeline runs in three stages:

1. **Embedding & candidate retrieval** — Each image is embedded with a vision transformer (DINOv2 or a pre-computed SigLIP vector). Cosine similarity selects the top-k candidate pairs per image, reducing the O(n²) matching problem.

2. **Geometric matching** — Each candidate pair is verified with SIFT + RANSAC (Fundamental/Homography) to count geometric inliers. A `border_fraction` mask suppresses watermarks and postcard frames. Stereocard images are automatically cropped to their left half.

3. **Graph clustering** — Verified pairs become edges in a NetworkX graph. Connected components form the final groups. An optional LightGlue (SuperPoint) symmetric check provides higher-precision verification when enabled.

```
images/
  └── input photos
       │
       ├─ DINOv2 / SigLIP embeddings  →  candidate pairs (cosine sim ≥ threshold)
       │
       ├─ SIFT + RANSAC               →  geometric inliers filter
       │
       └─ NetworkX connected components  →  groups.csv
```

## Repository Structure

```
.
├── notebooks/          # Exploratory Jupyter notebooks
├── report/             # PDF report and LaTeX sources
├── scripts/            # CLI entry points
│   └── run_pipeline.py
├── src/
│   ├── data_collection/   # Flickr API, metadata, download utilities
│   ├── matching/          # Clustering pipeline (cluster, sift, lightglue, enrich)
│   ├── preprocessing/     # Image classifier
│   └── visualization/     # Match viewer
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd flickr-image-clustering

# 2. Create a virtual environment (use /rcp-scratch on EPFL cluster)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials (copy and fill in)
cp .env.example .env
```

## Configuration

Copy `.env.example` to `.env` and fill in your Flickr API credentials:

```
FLICKR_API_KEY=your_key_here
FLICKR_API_SECRET=your_secret_here
```

Never commit `.env` to version control — it is listed in `.gitignore`.

## Usage

```bash
# Run the full clustering pipeline on a folder of images
make run IMAGES=images_test/ OUTPUT=results/ TOPK=10

# Run in dev mode (first 20 images only)
make dev

# See all available commands
make help
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
