# Image Sorter v1.2.1

Local desktop app for sorting and organizing image collections with several on-device AI models (AI vs human, content/style filters, clustering, deduplication, tagging, and semantic search).

## Features

- Separate likely AI-generated images from likely human-made ones  
- Filter by content and artistic style  
- Group visually similar images and remove near-duplicates  
- Generate `metadata.json` and optional Florence-2 captions  
- Local semantic search over images you have already scanned  

Everything runs offline after models are downloaded. Internet is only needed for the initial model download.

## Requirements

- Python 3.10+ (recommended)  
- PyTorch with CUDA (optional but recommended)  
- See `requirements.txt` for package versions  

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_models.py
python -m app.main
```

Optional CLI flags:

```bash
python -m app.main --engine camie
python -m app.main --engine wd
python -m app.main --skip-model-check
```

Legacy entry points (wrappers):

- `python sort_images_app.py`  
- `python download_models.py`  

## Documentation

- **[USAGE.md](USAGE.md)** — full user guide: interface, criteria, settings, scripts, models, troubleshooting, and tips  

## Project layout

| Path | Role |
|------|------|
| `app/main.py` | Application entry |
| `app/ui.py` | Tkinter UI |
| `app/pipeline.py` | Sorting pipeline |
| `app/models.py` | Model loading and inference |
| `app/dataset.py` | Image list and filesystem scan |
| `app/lmdb_cache.py` | Shared LMDB image cache |
| `app/utils.py` | Paths, settings, helpers |
| `scripts/` | `download_models.py`, `train_dinov.py`, `extract_tags` wrappers |
| `data/models/` | Downloaded model files |
| `data/cache/` | App cache and `settings.json` |
| `data/weights/` | Checkpoints and adapters |

Older installs may still use `models/`, `.cache/`, and `weights/` at the repo root; the app keeps backward compatibility.

## Scripts

- `python scripts/download_models.py` — verify / download models  
- `python scripts/train_dinov.py` — train the DINOv2 adapter (see USAGE.md)  
- `python scripts/extract_tags.py` — CLI tagging without the GUI  

## License

Specify your license in this repository when you publish it.
