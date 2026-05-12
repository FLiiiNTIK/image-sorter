# Image Sorter — User Guide

This document explains how to install, run, and configure **Image Sorter** (same content as the former Russian README, translated to English).

The app sorts images locally using several AI models: AI vs human separation, content/style filters, tagger-based filters, visual grouping, deduplication, optional detailed metadata, and local semantic search.

---

## 1. Quick start

### Launch the application

Preferred:

```bash
python -m app.main
```

Optional flags:

- `--engine camie` — force **Camie-Tagger-v2**  
- `--engine wd` — force **WD EVA02 v3**  
- `--skip-model-check` — skip automatic model verification / download on startup  

Compatibility launchers:

- `python sort_images_app.py`  
- `python download_models.py`  

### Typical workflow

1. Choose **Source** (folder with images).  
2. Choose **Target** (output folder).  
3. Enable the sorting criteria you need.  
4. Click **Start**.  
5. Wait until processing finishes.  
6. Use **Open** to open the result folder if available.  

---

## 2. Project structure

After the refactor, code lives in modules:

```text
app/
  main.py
  ui.py
  pipeline.py
  models.py
  dataset.py
  lmdb_cache.py
  utils.py

scripts/
  download_models.py
  train_dinov.py
  extract_tags.py

data/
  models/
  cache/
  weights/
```

### Data paths

Preferred layout:

- `data/models/` — model weights  
- `data/cache/` — caches and `settings.json`  
- `data/weights/` — adapters and training artifacts  

Legacy paths still supported: `models/`, `.cache/`, `weights/` at the project root.

---

## 3. How the pipeline works

Sorting runs in **stages**. Each enabled criterion may:

- add a subfolder to the destination path,  
- drop the image at the current stage, or  
- compute features for later stages.  

Typical order:

1. Scan folder  
2. LMDB cache check/build (if enabled)  
3. Load models  
4. AI vs Human  
5. Content filter  
6. Style filter  
7. Tagger filter  
8. Semantic grouping (text descriptions)  
9. DINOv2 / SigLIP / tagger features  
10. Sequence grouping  
11. Deduplication  
12. Visual grouping  
13. Cluster naming  
14. `metadata.json` generation (if enabled)  
15. Copy or move files to **Target**  

Disabled stages are skipped.

---

## 4. UI — Directories

### Source

Input folder. No special folder layout is required; subfolders are fine. With **Recursive**, nested directories are scanned.

### Target

Output root. Subfolders are created automatically from enabled criteria. **Move** removes files from **Source** under that path; **Copy** leaves originals.

### Recursive

Scan subfolders. Turn off if you only want the top-level directory.

### Blocked

Semicolon-, comma-, or newline-separated list of folder names or paths to **exclude** entirely (no scan, no LMDB, no sort). Examples: `temp`, `old/skip_this`, or an absolute path.

### Clear cache

Clears scan cache, feature cache (`feat_cache.pt`), and LMDB if present. Use after large folder changes or suspected stale cache.

### AI Search

Opens local semantic search over **SigLIP** embeddings already saved from a previous run. Requires `feat_cache.pt` from a prior scan/sort with features computed. Results go under `Target/AI_Search_Results/...`.

### Start / Cancel

**Start** runs work in a background thread. **Cancel** is cooperative (current batch may finish); already copied/moved files are not rolled back.

---

## 5. UI — Criteria

### AI vs Human

Splits images into `ai` vs `hum`. Optional **Dual-vote (SigLIP+SDXL)** requires **both** models to agree for the AI class (stricter, slower, more VRAM).

### Content / Style

SigLIP-based similarity to comma-separated **positive** tags, a negative anchor, and optional negative tags. **Min** is the confidence threshold (e.g. soft `0.05–0.15`, medium `0.2–0.4`, strict `0.5+`).

### Tagger Filter

Uses WD or Camie tagger scores. Supports positive tags (`blue_hair, school_uniform`) and negative tags (`-text, -weapon`). **Negative tags** are also matched against the **filename** (e.g. `-table` excludes `girl_on_table_001.png` even if the tagger missed `table`).

### Visual Grouping

Clusters by visual / semantic similarity. Empty **Semantic** → auto groups and names; filled **Semantic** → comma-separated categories you define. **Sens** controls cluster tightness (lower → more, smaller groups; higher → fewer, broader groups). Typical **Sens** range: `0.3–0.5`.

### Remove Duplicates

Near-duplicate removal. **Thr** e.g. `0.95+` for strict duplicates; lower values are more aggressive.

### DINOv2 Adapter

Uses a trained adapter on top of frozen DINOv2 for better semantic clustering when `adapter.pth` (or manga variant) is available.

### Generate Detailed Metadata

Writes `metadata.json` in result folders (ratings, tags, scores, optional `caption_florence2`). **Max/folder** limits images per folder (`0` = all). **Tags/img** caps tags per image (often `20–40` is enough).

### Generate Source Metadata

Writes into **Source** without sorting:

- `metadata.json` — flat `{"relative/path.png": "tag1, tag2"}` (compatible with `train_dinov.py`)  
- `metadata_detailed.json` — rich format with scores and optional Florence caption  

Respects Recursive, Blocked, LMDB, tagger settings, **Tags/img**, **Camie min**, and **Florence-2** when enabled. Large-dataset options: **Skip existing captions**, **Save every N**, **Florence/folder** cap (`0` = no cap).

### Training tab

Train the DINO adapter from the GUI: generate **Source** metadata first, then **Train DINO Adapter** with **Metadata: auto** unless you need a specific mode.

**Metadata** modes:

- `auto` — prefer `metadata_detailed.json` next to `metadata.json` to avoid duplicate training text  
- `simple` — flat `metadata.json` only  
- `detailed` — tags + scores from `metadata_detailed.json`  
- `detailed_florence` — also inject `caption_florence2` into training text  

---

## 6. Settings

### Group Manga / Sequences

Groups sequential pages (manga, bursts) using filename similarity, timestamps, and DINO features when available.

### LMDB

- `auto` — use LMDB if the `lmdb` package is installed  
- `on` — require LMDB  
- `off` — read images directly from disk  

Shared LMDB speeds repeat runs and aligns GUI, `extract_tags.py`, and `train_dinov.py` when **Cache dir** is the same. Default LMDB dir is often `data/cache/dinov2_train_lmdb`.

### Rebuild

Force LMDB rebuild before a run (after mass image replacement or corruption suspicion).

### Copy / Move

**Copy** is safer and uses more disk space. **Move** is faster on one disk but removes files from **Source** — use only when you intend to reorganize originals.

### torch.compile

May speed long runs after warmup; first launch can be slower; benefit varies by GPU/driver.

### Batch / Auto VRAM batch / DINO batch / Tagger batch / Florence Batch

Tune batch sizes for VRAM. On **8 GB** cards, keep **Auto VRAM batch** on, VRAM profile **Auto** or **8 GB**, and lower **Florence Batch** (e.g. `2`, or `1` if OOM).

### Unload inactive models

Moves idle torch models to CPU before heavy Florence work — recommended on 8 GB.

### VRAM / Precision / Tagger

**VRAM** profile caps memory for auto-batching. **Precision**: `auto` (default), `bf16`, `fp16`, or `fp32`. **Tagger** chooses Camie vs WD; **Thr** is Camie / tagger-related confidence.

---

## 7. Presets

Presets store content tags, style tags, and filter combinations for quick reuse.

---

## 8. Local AI Search

Uses SigLIP text ↔ image similarity over cached embeddings. Example queries: `red car in snow`, `1girl in school uniform`. This is **not** web search or OCR.

---

## 9. Cache and settings file

Persisted settings (paths, toggles, thresholds, LMDB, presets, precision, Florence, tagger, etc.) are stored primarily in:

- `data/cache/settings.json`  

Legacy: `.cache/settings.json` if it already exists.

Caches include scan results, feature tensors, LMDB bytes, and model-related caches.

### Shared LMDB keys

- `path:relative/path.png` — GUI / `extract_tags.py`  
- `img:0`, `img:1`, … — `train_dinov.py` index  

Same **Cache dir** lets you tag, train, and sort without rebuilding bytes.

---

## 10. Models used in the app

| Model | Role |
|--------|------|
| `ai-vs-human-image-detector` | Primary AI vs human |
| `sdxl-detector` | Dual-vote second opinion |
| `siglip2-so400m-patch16-512` | Content/style, semantic grouping, AI search |
| `dinov2-base` | Grouping, dedup, sequences |
| `wd-eva02-large-tagger-v3` | WD tagger path |
| `camie-tagger-v2` | Camie tagger path |
| `florence-2-large-promptgen-v2` | Optional captions |
| `all-MiniLM-L6-v2` | Text embeddings for `train_dinov.py` |

---

## 11. Scripts

### `scripts/download_models.py`

Checks and downloads required models.

### `scripts/train_dinov.py`

Trains the DINOv2 → text adapter. Useful flags:

- `--lmdb auto|on|off`, `--lmdb_dir PATH`, `--rebuild_lmdb`  
- `--metadata_source auto|simple|detailed|detailed_florence`  
- `--min_tag_score 0.1`  
- `--no_auto_optimize`  

Example:

```bash
python scripts/train_dinov.py --data_dir "D:\dataset" --lmdb auto --lmdb_dir "D:\cache\shared_lmdb"
```

Log hints (English):

- `LMDB found and opened read-only without rebuild` — cache reused  
- `Building LMDB` — cache is being written or rebuilt  
- `DataLoader: batch=..., workers=..., persistent=True, prefetch=4` — fast data path  

If you see `DecompressionBombWarning` or low GPU utilization with huge sources, lower **Max side** in training and prefer **Augment: light** until speed is acceptable.

### `scripts/extract_tags.py`

CLI tagging without the GUI; mirrors options of **Generate Source Metadata** for batch workflows.

---

## 12. `metadata.json` contents

When detailed metadata is enabled, expect folder-level stats plus per-image fields such as:

- `rating`, `general_tags`, `character_tags`, `mature_tags`, `all_scores`, `caption_florence2`  

Useful for dataset prep and export to other tools.

---

## 13. Practical presets (workflows)

- **Fast AI/human only:** enable **AI vs Human** (optional dual-vote), disable other criteria.  
- **Unsorted “soup” by topic:** **Visual Grouping** + **Group Manga/Sequences**; optional metadata.  
- **Anime tag precision:** **Tagger Filter** with explicit +/- tags and threshold tuning.  
- **Duplicate cleanup:** **Remove Duplicates** starting at **Thr = 0.95**.  
- **Maximum speed:** disable Florence, dual-vote if not needed, skip detailed metadata, tune batch/precision, avoid unnecessary cache clears.  

---

## 14. Performance and memory

Slowdowns usually come from Florence-2, SigLIP2, large batches, `torch.compile` warmup, metadata generation, and heavy clustering on huge sets. Reduce batch, disable Florence / dual-vote, try `fp16`/`auto`, and clear cache only when needed.

---

## 15. Refactor notes (for upgraders)

- Entry point: `python -m app.main`  
- **Tagger Filter** negative tags also check filenames  
- Florence path avoids per-image CPU↔GPU thrash for faster captions  
- Centralized paths under `app/utils.py` / `get_paths()`  

---

## 16. FAQ

**Nothing was found** — check **Source**, **Recursive**, filter thresholds, and **Blocked** paths.  

**Too many output folders** — raise **Sens** in visual grouping, simplify semantic labels, or relax dedup/grouping.  

**Almost everything in one folder** — **Sens** may be too high; filters too loose; semantics too generic.  

**Florence is slow** — expected; reduce images (**Max/folder**), use a lighter Florence mode, or disable Florence.  

**Tagger filter drops files “without” the tag** — negative rules intentionally scan **filenames** as well as tagger output.  

---

## 17. Dependencies (main)

```text
torch>=2.0.0
torchvision>=0.15.0
transformers==4.40.1
Pillow>=9.5.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0
einops>=0.7.0
accelerate>=0.29.0
requests>=2.31.0
umap-learn>=0.5.0
```

Taggers additionally need `onnxruntime` or `onnxruntime-gpu` and `pandas`.

---

## Short cheat sheet

| Control | Meaning |
|---------|---------|
| AI vs Human | Split AI vs human-looking |
| Content | “What” is in the image (SigLIP) |
| Style | Artistic style (SigLIP) |
| Tagger Filter | Fine-grained WD/Camie tags (+ filename negatives) |
| Visual Grouping | Similar-image clusters |
| Remove Duplicates | Near-duplicate removal |
| Generate Detailed Metadata | Rich per-folder `metadata.json` |
| Florence-2 | Longer captions, much slower |

Good luck organizing your library.
