# Image Sorter v2.0.0 — Full Documentation

Intelligent image sorting application using state-of-the-art neural networks.
Works completely offline, no internet required.

---

## Project Structure

| File / Directory | Purpose |
|---|---|
| `sort_images_app.py` | Main GUI application (Tkinter) |
| `train_dinov.py` | DINOv2 adapter training script |
| `requirements.txt` | Python dependencies |
| `.cache/` | Feature cache, settings, scan cache |
| `weights/` | Trained adapter weights |

---

## Step 1. Directories

| Setting | Description |
|---|---|
| **Source** | Folder with unsorted images |
| **Target** | Destination folder (subfolders will be created) |
| **Recursive** | Scan Source recursively (including subfolders) |
| **🔍 AI Search** | Semantic search across your image library (works after first run) |
| **Clear cache** | Deletes the feature cache (`.cache/feat_cache.pt`) |

---

## Step 2. Sorting Criteria

Criteria are combinable. Each adds a nesting level to the output folders.

### 1. AI vs Human
Detects whether an image was AI-generated or created by a human (photo/manual artwork).

| Parameter | Description |
|---|---|
| **Dual-vote (SigLIP+SDXL)** | Two-model voting system. Reduces false positives, recommended to keep ON |

**Models:** `ai-vs-human-image-detector` (SigLIP-based) + `sdxl-detector` (Swin Transformer)

### 2. Content & Style (Filters)
Zero-shot image classification by text tags via **SigLIP2**.

| Parameter | Description |
|---|---|
| **Content Tags** | What is depicted: `cat, dog, car, building` |
| **Style Tags** | Art style: `anime, oil painting, realistic photo` |
| **Min Conf** | Minimum confidence to pass filter (0.0–1.0) |
| **Neg Anchor** | Anchor phrase for comparison ("none of the above") |
| **Tag Weights** | `cat:1.5, dog:0.8` — boost/reduce tag influence |
| **Negative Tags** | `-table` — exclude images matching this tag |
| **Multi-tagging** | If an image matches 2+ tags, a combined folder `cat_dog` is created |
| **Tag Preset** | Save/load tag presets |

**Model:** `siglip2-so400m-patch16-512` (Google SigLIP2, 4.5 GB)

### 3. Visual Grouping (Clustering)
Automatically groups visually similar images into folders.

| Parameter | Description |
|---|---|
| **Sens (Sensitivity)** | `0.05`=strict (few large clusters), `0.5`=balanced, `0.9`=loose (many small clusters). Directly controls threshold via distance percentile |
| **Semantic Descriptions** | Comma-separated words to sort into. If empty — automatic clustering |
| **DINOv2 Adapter** | Use trained adapter (improves accuracy when trained on your data) |

**Clustering Pipeline (v2.0):**
1. **DINOv2 multi-scale** — CLS token + spatial pooling = 1536d visual features
2. **SigLIP2** — 1152d semantic image-text features
3. **WD Tagger** — 64d semantic vector (pose, clothing, scene, hair color, expression, characters)
4. **Fusion** — Per-modality L2 normalization × weights: DINOv2 ×1.0, SigLIP ×0.7, WD ×0.3
5. **UMAP** — Dimensionality reduction ~2700d → 20d (preserves local structure)
6. **AgglomerativeClustering** — Ward linkage on Euclidean distances in UMAP space

**Cluster Naming (TF-IDF):**
- Tags common to ALL clusters are penalized (e.g., `1girl`)
- Tags unique to a specific cluster are boosted
- Character names are prioritized in folder names
- Duplicate names resolved with distinguishing subtags instead of `_002` suffixes

**Models:** `dinov2-base` (Meta, 0.35 GB) + `siglip2-so400m-patch16-512` + `wd-swinv2-tagger-v3` (ONNX, 0.5 GB)

### 4. Remove Duplicates
Removes visual duplicates using DINOv2 feature similarity.

| Parameter | Description |
|---|---|
| **Thr** | Similarity threshold (0.0–1.0). At `0.95`, images matching 95%+ are removed |

### 5. Generate Detailed Metadata
Generates `metadata.json` in each sorted folder with WD Tagger tags.

| Parameter | Description |
|---|---|
| **Max per folder** | Maximum images to tag per folder (0 = all) |
| **Tags per image** | Number of tags per image (default 30) |

Includes: rating (general/sensitive/questionable/explicit), general tags, character tags, mature tags.

---

## AI Search (Semantic Search)

After the first run, a cache `feat_cache.pt` is created with SigLIP embeddings of all images.

1. Click **🔍 AI Search**
2. Enter a description in English: `a cute red fox sleeping in the snow`
3. The app finds the 20 most matching images from the cache
4. Results are saved to `AI_Search_Results` inside the Target folder

---

## Settings

| Setting | Description |
|---|---|
| **Copy / Move** | Copy or move files to target |
| **Batch** | Batch size (2–16). Smaller = less VRAM usage |
| **Precision** | `auto` (recommended), `fp16`, `bf16`, or `fp32` |

---

## Code Reference

### Classes

| Class | Description |
|---|---|
| `LoadingBar` | Animated loading bar with timer for model loading progress |
| `ImagePathDataset` | PyTorch Dataset: loads images from paths for DataLoader |
| `DINOv2Adapter` | MLP adapter (768/1536 → 384). Projects DINOv2 vectors into text space |
| `ImageSorterApp` | Main application class — UI, pipeline, all models |

### Core Methods (`ImageSorterApp`)

| Method | Description |
|---|---|
| `_build_ui` | Full GUI construction: inputs, checkboxes, sliders, log |
| `_load_settings` / `_save_settings` | Load/save all settings to `.cache/settings.json` |
| `_load_models` | Loads required models based on selected criteria |
| `_load_wd_tagger` | Loads WD SwinV2 Tagger v3 (ONNX Runtime) |
| `_start` | Validates parameters, launches `_run` in separate thread |
| `_run` | **Main pipeline.** Sequentially: load models → scan → AI/Human → Content/Style filter → Semantic grouping → DINOv2+SigLIP+WD features → Dedup → Clustering → Naming → Metadata → Save |

### Neural Network Methods

| Method | Description |
|---|---|
| `_siglip_embed_image` | Get normalized SigLIP2 image embedding (1, D) |
| `_siglip_embed_texts` | Get normalized SigLIP2 text embeddings (N, D) |
| `_get_tag_embeddings` | Compute tag embeddings with prompt ensembling |
| `_siglip_filter` | Zero-shot filter by positive/negative tags with anchor |
| `_siglip_classify` | Assign image to best matching semantic group |
| `_wd_tagger_infer` | WD Tagger inference: image → dict of {tag: score} |
| `_wd_tagger_refine_clusters` | Refine clusters via tag IoU (splits incoherent groups) |
| `_wd_tagger_name_clusters` | TF-IDF-based cluster naming with distinctive tags |
| `_wd_tagger_generate_metadata` | Generate `metadata.json` per folder |

### Clustering

| Method | Description |
|---|---|
| `_cluster_images` | **UMAP → AgglomerativeClustering (Ward).** Compresses multi-modal features to 20d via UMAP, then clusters with distance-percentile threshold |
| `_deduplicate` | Remove duplicates by cosine similarity of DINOv2 features |

---

## Models — What's Used and What's Not

### ✅ Used in `sort_images_app.py`

| Directory | Model | Purpose | Size |
|---|---|---|---|
| `ai-vs-human-image-detector/` | SigLIP for Classification | AI vs Human classification | ~0.4 GB |
| `sdxl-detector/` | Swin Transformer | Dual-vote: additional AI-generated check | ~0.35 GB |
| `siglip2-so400m-patch16-512/` | Google SigLIP2 so400m | Zero-shot classification, content/style filters, AI Search, image embeddings | ~4.5 GB |
| `dinov2-base/` | Meta DINOv2 ViT-B/14 | Multi-scale visual features for clustering and deduplication | ~0.35 GB |
| `wd-swinv2-tagger-v3/` | WD SwinV2 Tagger v3 (ONNX) | Anime/art tags, semantic vector for clustering, cluster naming, metadata | ~0.5 GB |
| `weights/` | DINOv2 Adapter (PyTorch) | Trained MLP adapter DINOv2 → text space | ~2 MB |

### ❌ NOT Used in `sort_images_app.py`

| Directory | Model | Notes |
|---|---|---|
| `moondream2/` | Moondream2 (VLM) | **Removed in previous versions.** Was for descriptions and cluster naming. Replaced by WD Tagger. Safe to delete (~1.8 GB) |
| `nsfw_image_detection/` | NSFW Classifier | **Not used in current code.** Path is declared but never called. Safe to delete |
| `all-MiniLM-L6-v2/` | SentenceTransformer MiniLM | **Not used in `sort_images_app.py`.** Used only in `train_dinov.py` for adapter training. Only needed if retraining the adapter |

---

## Training Script `train_dinov.py`

Separate script for training the DINOv2 adapter on (image, text description) pairs.

**Models used:**
- `dinov2-base/` — visual feature extraction (frozen backbone)
- `all-MiniLM-L6-v2/` — target text embedding generation

**Output:** `weights/best_adapter.pth` — adapter projecting DINOv2 into text space.

> ⚠️ **Important:** Since v2.0, DINOv2 outputs 1536d (CLS + spatial). Old adapters (trained on 768d) are automatically skipped. To use the adapter, retrain with `train_dinov.py` (uses `input_dim=1536` by default).

---

## Dependencies

```
torch>=2.0.0, torchvision>=0.15.0, transformers==4.40.1,
Pillow>=9.5.0, scikit-learn>=1.3.0, numpy>=1.24.0,
scipy>=1.10.0, einops>=0.7.0, accelerate>=0.29.0,
requests>=2.31.0, umap-learn>=0.5.0
```

Additionally for WD Tagger: `onnxruntime` (or `onnxruntime-gpu`), `pandas`.

Happy sorting! 🎉
