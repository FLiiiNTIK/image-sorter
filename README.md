# Image Sorter v2.0.0

A powerful desktop application for intelligent image sorting, clustering, and classification using state-of-the-art AI models.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **AI vs Human Detection** — Classify images as AI-generated or human-created using SigLIP + SDXL dual-vote system
- **Content & Style Filtering** — Zero-shot classification with custom tags (e.g., `person, landscape, anime`)
- **Visual Clustering** — UMAP + Agglomerative Clustering with multi-modal features (DINOv2 + SigLIP2 + WD Tagger)
- **Smart Folder Naming** — TF-IDF-based tag analysis for distinctive, content-accurate folder names
- **Duplicate Removal** — DINOv2-based perceptual deduplication
- **Detailed Metadata** — Generate `metadata.json` with WD Tagger tags per image
- **AI Search** — Find similar images using SigLIP2 text-to-image search

## Models Required

Download and place these models in the same directory as `sort_images_app.py`:

| Model | Directory Name | Purpose |
|---|---|---|
| [SigLIP2-so400m](https://huggingface.co/google/siglip2-so400m-patch16-512) | `siglip2-so400m-patch16-512/` | Content/style filtering, AI search, embedding |
| [DINOv2-base](https://huggingface.co/facebook/dinov2-base) | `dinov2-base/` | Visual feature extraction for clustering |
| [WD SwinV2 Tagger v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) | `wd-swinv2-tagger-v3/` | Semantic tagging, cluster naming, metadata |
| [AI vs Human Detector](https://huggingface.co/Organika/sdxl-detector) | `ai-vs-human-image-detector/` | AI image classification (SigLIP-based) |
| [SDXL Detector](https://huggingface.co/Organika/sdxl-detector) | `sdxl-detector/` | Secondary AI vote (Swin Transformer) |

### Optional

| Model | Directory Name | Purpose |
|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `all-MiniLM-L6-v2/` | Only for `train_dinov.py` adapter training |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-sorter.git
cd image-sorter

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Main Application

```bash
python sort_images_app.py
```

The GUI allows you to:
1. Select **Source** and **Target** directories
2. Enable desired sorting criteria (AI detection, content/style filtering, clustering)
3. Adjust sensitivity and thresholds
4. Click **Start** to process

### DINOv2 Adapter Training

The optional adapter improves clustering quality by projecting DINOv2 features into a text-aligned space.

```bash
# Prepare training data: place images + JSON descriptions in a folder
python train_dinov.py --data_dir ./training_data --epochs 30

# The adapter will be saved to weights/best_adapter.pth
# Enable "DINOv2 Adapter" checkbox in the app to use it
```

See [README_SETTINGS.md](README_SETTINGS.md) for detailed documentation of all features and parameters.

## Project Structure

```
image-sorter/
├── sort_images_app.py      # Main application
├── train_dinov.py           # DINOv2 adapter training script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── README_SETTINGS.md       # Detailed settings documentation
├── weights/                 # Trained adapter weights
│   └── best_adapter.pth
├── .cache/                  # Feature cache (auto-created)
├── ai-vs-human-image-detector/
├── sdxl-detector/
├── siglip2-so400m-patch16-512/
├── dinov2-base/
├── wd-swinv2-tagger-v3/
└── all-MiniLM-L6-v2/       # Only for training
```

## How It Works

### Clustering Pipeline

1. **Feature Extraction** — Multi-scale DINOv2 (CLS + spatial = 1536d) + SigLIP2 (1152d) + WD Tagger semantic features (64d)
2. **Feature Fusion** — L2-normalized per-modality, weighted (DINOv2: 1.0, SigLIP: 0.7, WD: 0.3), concatenated
3. **Dimensionality Reduction** — UMAP projects to 20 dimensions
4. **Clustering** — Agglomerative Clustering with sensitivity-controlled distance threshold
5. **Refinement** — WD Tagger verifies intra-cluster coherence, splits incoherent groups
6. **Naming** — TF-IDF tag scoring prioritizes distinctive tags over generic ones

### AI Detection

Dual-vote system: SigLIP-based classifier + SDXL Swin Transformer. Both must agree for classification.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (recommended)
- ~4 GB VRAM for full pipeline
- See `requirements.txt` for all dependencies

## License

MIT License
