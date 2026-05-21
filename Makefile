# ============================================================
#  Flickr Commons — Image Clustering Pipeline
#  Usage : make <target> [VAR=value ...]
# ============================================================

IMAGES  ?= images_test/
OUTPUT  ?= results/
TOPK    ?= 5
PYTHON  ?= python
KV_N    ?= 500
KV_OUT  ?= images_kv/

.PHONY: help run dev aggregate download-eiffel download-paris extract-eiffel extract-paris extract-notredame export download-kv run-kv

# ── Default ─────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Pipeline principale"
	@echo "    make run              Full pipeline  (IMAGES, OUTPUT, TOPK)"
	@echo "    make dev              Full pipeline sur 20 images seulement"
	@echo ""
	@echo "  Données"
	@echo "    make aggregate        Fusionne tous les CSV metadata/"
	@echo "    make extract-eiffel   Extrait le cluster Tour Eiffel"
	@echo "    make extract-paris    Extrait le cluster Paris"
	@echo "    make extract-notredame Extrait le cluster Notre-Dame"
	@echo "    make download-eiffel  Télécharge les images Tour Eiffel"
	@echo "    make download-paris   Télécharge les images Paris"
	@echo ""
	@echo "  Post-traitement"
	@echo "    make export           Copie les images dans dossiers par groupe"
	@echo ""
	@echo "  keyword_then_vision"
	@echo "    make download-kv      Télécharge KV_N images YES (défaut 500)"
	@echo "    make run-kv           Download + pipeline sur images_kv/"
	@echo ""
	@echo "  Variables (ex: make run IMAGES=mon_dossier/ TOPK=10)"
	@echo "    IMAGES  = $(IMAGES)"
	@echo "    OUTPUT  = $(OUTPUT)"
	@echo "    TOPK    = $(TOPK)"
	@echo "    KV_N    = $(KV_N)"
	@echo "    KV_OUT  = $(KV_OUT)"
	@echo ""

# ── Pipeline ────────────────────────────────────────────────
run:
	$(PYTHON) scripts/run_pipeline.py --images $(IMAGES) --output $(OUTPUT) --topk $(TOPK)

dev:
	$(PYTHON) scripts/run_pipeline.py --images $(IMAGES) --output $(OUTPUT) --topk $(TOPK) --dev

# ── Données ─────────────────────────────────────────────────
aggregate:
	$(PYTHON) -c "from src.data_collection.aggregate_metadata import aggregate_flickr_commons; aggregate_flickr_commons()"

extract-eiffel:
	$(PYTHON) src/data_collection/extract_tour_eiffel_cluster_all_csv.py

extract-paris:
	$(PYTHON) src/data_collection/extract_paris_cluster_all_csv.py

extract-notredame:
	$(PYTHON) src/data_collection/extract_notre_dame_cluster_all_csv.py

download-eiffel:
	$(PYTHON) src/data_collection/download_cluster_images.py

download-paris:
	INPUT_CSV=cluster_paris.csv OUTPUT_DIR=images_paris \
	$(PYTHON) src/data_collection/download_cluster_images.py

# ── keyword_then_vision ─────────────────────────────────────
download-kv:
	$(PYTHON) scripts/download_keyword_vision.py --n $(KV_N) --output $(KV_OUT)

run-kv: download-kv
	$(PYTHON) scripts/run_pipeline.py --images $(KV_OUT) --output results_kv/ --topk $(TOPK)

# ── Post-traitement ─────────────────────────────────────────
export:
	$(PYTHON) src/data_collection/export_groups_to_folders.py \
		--groups $(OUTPUT)groups.csv \
		--images $(IMAGES) \
		--output grouped_results/
