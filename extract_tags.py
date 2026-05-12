"""
extract_tags.py — CLI utility to build `metadata.json` with the WD EVA02 or Camie tagger.
Runs outside the main GUI.

Example:
python extract_tags.py --data_dir "path/to/images" --threshold 0.35 --exclude_folder "temp"
"""

import os
import sys
import json
import argparse
import time
import io
import hashlib
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm
from app.utils import format_size, get_paths
from app.lmdb_cache import configured_lmdb_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import lmdb
except Exception:
    lmdb = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = get_paths()
WD_TAGGER_PATH = str(PATHS.wd_tagger_path)
CAMIE_TAGGER_PATH = str(PATHS.camie_tagger_path)
DEFAULT_LMDB_DIR = str(configured_lmdb_dir())

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def safe_filename(name: str) -> str:
    """Sanitize folder name, preserving Unicode alphanumeric."""
    if not name: return "other"
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip() or "other"

def prepare_image(image: Image.Image, engine: str = "wd", target_size: int = 448) -> np.ndarray:
    """Prepare PIL image for Tagger ONNX inference."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    
    if engine == "camie":
        import torchvision.transforms as transforms
        aspect_ratio = w / h
        if aspect_ratio > 1:
            nw = 512
            nh = int(512 / aspect_ratio)
        else:
            nh = 512
            nw = int(512 * aspect_ratio)
        
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        pad_color = (124, 116, 104)
        padded = Image.new('RGB', (512, 512), pad_color)
        padded.paste(image, ((512 - nw) // 2, (512 - nh) // 2))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(padded).unsqueeze(0).numpy()
    else:
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        padded.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))

        # Convert to float32 numpy, RGB → BGR
        # ONNX EVA02 expects NHWC layout (1, 448, 448, 3)
        arr = np.array(padded, dtype=np.float32) / 255.0
        arr = arr[:, :, ::-1].copy()  # RGB → BGR
        return np.expand_dims(arr, axis=0).copy()  # add batch dim


def clean_rel_path(path: str) -> str:
    return path.replace("\\", "/")


def _path_key(rel_path: str) -> bytes:
    return f"path:{clean_rel_path(rel_path)}".encode("utf-8")


def _train_key(index: int) -> bytes:
    return f"img:{index}".encode("ascii")


def _extract_lmdb_hash(image_paths: list[str], data_dir: str) -> str:
    records = []
    for path in image_paths:
        try:
            stat = os.stat(path)
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        records.append((clean_rel_path(os.path.relpath(path, start=data_dir)), size, mtime))
    return hashlib.md5(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _train_lmdb_hash(samples: list[dict[str, str]]) -> str:
    records = []
    for sample in samples:
        path = sample["image_path"]
        try:
            stat = os.stat(path)
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        records.append((path, size, mtime, sample["description"]))
    return hashlib.md5(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _lmdb_map_size(image_paths: list[str]) -> int:
    total = 0
    for path in image_paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return max(128 << 20, int(total * 2.2) + (64 << 20))


def prepare_extract_lmdb(image_paths: list[str], data_dir: str, lmdb_dir: str, mode: str, rebuild: bool = False):
    if mode == "off":
        return None
    if lmdb is None:
        message = "LMDB unavailable: install the `lmdb` package."
        if mode == "on":
            raise RuntimeError(message)
        print(f"[LMDB] {message} Continuing without LMDB.")
        return None

    lmdb_path = Path(lmdb_dir)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    expected_hash = _extract_lmdb_hash(image_paths, data_dir)
    meta_path = lmdb_path / "extract_meta.json"
    if not rebuild and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("hash") == expected_hash and int(meta.get("count", 0)) == len(image_paths):
                print(f"[LMDB] Image cache is up to date: {lmdb_path}")
                return lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True, meminit=False, max_readers=256)
        except Exception as exc:
            print(f"[LMDB] extract_meta.json damaged, rebuilding: {exc}")

    map_size = _lmdb_map_size(image_paths)
    print(f"[LMDB] Building image cache: {lmdb_path} | map_size={format_size(map_size)}")
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False, max_readers=256)
    txn = env.begin(write=True)
    written = skipped = 0
    try:
        for index, img_path in enumerate(tqdm(image_paths, desc="LMDB images")):
            rel_path = clean_rel_path(os.path.relpath(img_path, start=data_dir))
            try:
                with open(img_path, "rb") as handle:
                    payload = handle.read()
                txn.put(_path_key(rel_path), payload)
                written += 1
            except OSError:
                skipped += 1
            if index > 0 and index % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        env.close()
        raise
    env.sync()
    env.close()
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hash": expected_hash,
                "count": len(image_paths),
                "written": written,
                "skipped": skipped,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[LMDB] Done: written={written}, skipped={skipped}")
    return lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True, meminit=False, max_readers=256)


def read_lmdb_image(env, rel_path: str) -> bytes | None:
    if env is None:
        return None
    with env.begin(write=False) as txn:
        value = txn.get(_path_key(rel_path))
    return bytes(value) if value is not None else None


def write_train_lmdb(samples: list[dict[str, str]], data_dir: str, lmdb_dir: str, mode: str) -> None:
    if mode == "off" or not samples:
        return
    if lmdb is None:
        return
    lmdb_path = Path(lmdb_dir)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    train_hash = _train_lmdb_hash(samples)
    meta_path = lmdb_path / "meta.json"
    image_total = 0
    for sample in samples:
        try:
            image_total += os.path.getsize(sample["image_path"])
        except OSError:
            pass
    data_file = lmdb_path / "data.mdb"
    existing_size = data_file.stat().st_size if data_file.exists() else 0
    map_size = max(
        256 << 20,
        existing_size + int(image_total * 1.4) + (128 << 20),
        int(image_total * 3.4) + (128 << 20),
    )
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False, max_readers=256)
    txn = env.begin(write=True)
    written = skipped = 0
    try:
        for index, sample in enumerate(tqdm(samples, desc="LMDB train index")):
            rel_path = clean_rel_path(os.path.relpath(sample["image_path"], start=data_dir))
            payload = txn.get(_path_key(rel_path))
            if payload is None:
                try:
                    with open(sample["image_path"], "rb") as handle:
                        payload = handle.read()
                except OSError:
                    skipped += 1
                    continue
            txn.put(_train_key(index), payload)
            written += 1
            if index > 0 and index % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        env.close()
        raise
    env.sync()
    env.close()
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hash": train_hash,
                "count": len(samples),
                "written": written,
                "skipped": skipped,
                "source_data_dir": data_dir,
                "metadata_file": os.path.join(data_dir, "metadata.json"),
                "created_by": "extract_tags.py",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[LMDB] DINOv2 train LMDB index ready: {lmdb_path} | written={written}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata.json with WD or Camie tagger")
    parser.add_argument("--data_dir", required=True, help="Folder containing your image collection")
    parser.add_argument("--engine", type=str, choices=["wd", "camie"], default="camie", help="Tagger engine: wd or camie")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum tag confidence (0.0–1.0)")
    parser.add_argument("--exclude_folder", type=str, action="append", help="Folder name to exclude (repeatable)")
    parser.add_argument("--save_txt", action="store_true", help="Also write a .txt tag file next to each image")
    parser.add_argument("--lmdb", choices=["auto", "on", "off"], default="auto",
                        help="LMDB image cache: auto if lmdb installed, on requires lmdb, off disables.")
    parser.add_argument("--lmdb_dir", default=DEFAULT_LMDB_DIR,
                        help="LMDB directory (default shared with train_dinov.py).")
    parser.add_argument("--rebuild_lmdb", action="store_true",
                        help="Force LMDB rebuild before tagging.")
    parser.add_argument("--no_train_lmdb", action="store_true",
                        help="Do not write train_dinov-compatible img:N keys after metadata.json.")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: folder not found: {data_dir}")
        sys.exit(1)

    import download_models
    download_models.check_and_download_all()

    print(f"Loading tag list ({args.engine})...")
    if args.engine == "camie":
        os.makedirs(CAMIE_TAGGER_PATH, exist_ok=True)
        model_path = os.path.join(CAMIE_TAGGER_PATH, "camie-tagger-v2.onnx")
        tags_path = os.path.join(CAMIE_TAGGER_PATH, "camie-tagger-v2-metadata.json")
        if not os.path.exists(model_path) or not os.path.exists(tags_path):
            print("Camie-Tagger-v2 model files are missing!")
            sys.exit(1)
        
        with open(tags_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idx_to_tag = meta["dataset_info"]["tag_mapping"]["idx_to_tag"]
        tag_to_cat = meta["dataset_info"]["tag_mapping"]["tag_to_category"]
        max_idx = max([int(k) for k in idx_to_tag.keys()])
        wd_tags = [idx_to_tag.get(str(i), f"unknown_{i}") for i in range(max_idx + 1)]
        wd_tag_categories = tag_to_cat
    else:
        model_path = os.path.join(WD_TAGGER_PATH, "model.onnx")
        tags_path = os.path.join(WD_TAGGER_PATH, "selected_tags.csv")

        if not os.path.exists(model_path) or not os.path.exists(tags_path):
            print(f"Error: WD Tagger model not found under {WD_TAGGER_PATH}")
            sys.exit(1)

        df = pd.read_csv(tags_path)
        wd_tags = df["name"].tolist()
        wd_tag_categories = dict(zip(df["name"], df["category"]))

    print(f"Loading ONNX tagger from {model_path}...")
    
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        print("  [GPU] CUDAExecutionProvider available (onnxruntime-gpu). Using GPU...")
        # onnxruntime tries CUDA first, then falls back through the provider list
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("  [CPU] onnxruntime-gpu not installed or CUDA unavailable. Using CPU...")
        providers = ['CPUExecutionProvider']
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        active_provider = session.get_providers()[0]
        if active_provider == 'CPUExecutionProvider' and 'CUDAExecutionProvider' in providers:
            print("  [Warning] GPU init failed (driver/CUDA). Falling back to CPU...")
    except Exception as e:
        print(f"  [Warning] Session init error ({e}). Forcing CPU...")
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        
    input_name = session.get_inputs()[0].name
    print(f"  Active provider: {session.get_providers()[0]}")

    print(f"Scanning folder {data_dir}...")
    image_paths = []
    
    exclude_folders = args.exclude_folder or []
    
    for root, dirs, files in os.walk(data_dir):
        if exclude_folders:
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTS:
                image_paths.append(os.path.join(root, f))
    image_paths.sort(key=lambda p: clean_rel_path(os.path.relpath(p, start=data_dir)).lower())

    if not image_paths:
        print("No supported images found in that folder.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Starting tagging...")
    lmdb_env = prepare_extract_lmdb(
        image_paths=image_paths,
        data_dir=data_dir,
        lmdb_dir=args.lmdb_dir,
        mode=args.lmdb,
        rebuild=args.rebuild_lmdb,
    )

    # Simple flat dictionary mapping file path to tag string
    metadata = {}
    tagged_samples = []

    errors = 0
    t0 = time.time()
    try:
        for img_path in tqdm(image_paths, desc="Tagging"):
            rel_path = clean_rel_path(os.path.relpath(img_path, start=data_dir))
            try:
                image_bytes = read_lmdb_image(lmdb_env, rel_path)
                if image_bytes is not None:
                    with Image.open(io.BytesIO(image_bytes)) as pil_img:
                        img_tensor = prepare_image(pil_img, engine=args.engine)
                else:
                    with Image.open(img_path) as pil_img:
                        img_tensor = prepare_image(pil_img, engine=args.engine)

                outputs = session.run(None, {input_name: img_tensor})
                if args.engine == "camie":
                    logits = outputs[1][0] if len(outputs) >= 2 else outputs[0][0]
                    preds = 1.0 / (1.0 + np.exp(-logits))
                else:
                    preds = outputs[0][0]

                general_tags = []
                character_tags = []
                all_scores = {}

                for score, tag_name in zip(preds, wd_tags):
                    if score > args.threshold:
                        if tag_name in {"no_humans", "text_focus"} or tag_name.startswith("year_"):
                            continue

                        category = wd_tag_categories.get(tag_name, -1)

                        if args.engine == "camie":
                            is_char = (category == "character")
                            is_rating = (category in ["rating", "meta", "object"])
                        else:
                            is_char = (category == 4)
                            is_rating = (category == 9)

                        all_scores[tag_name] = round(float(score), 4)

                        if is_rating:  # Ratings — skip for tag lists
                            continue
                        elif is_char:  # Characters
                            character_tags.append(tag_name)
                        else:  # General
                            general_tags.append(tag_name)

                # Combine tags for standard comma-separated format
                final_tags = general_tags + character_tags
                tag_string = ", ".join(final_tags)

                # Save to JSON flat structure
                metadata[rel_path] = tag_string
                tagged_samples.append({"image_path": os.path.abspath(img_path), "description": tag_string})

                # Also save standard .txt file next to image if requested
                if args.save_txt:
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(tag_string)

            except Exception as e:
                errors += 1
                print(f"\nError processing {img_path}: {e}")
    finally:
        if lmdb_env is not None:
            lmdb_env.close()

    out_file = os.path.join(data_dir, "metadata.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if not args.no_train_lmdb:
        try:
            write_train_lmdb(tagged_samples, data_dir, args.lmdb_dir, args.lmdb)
        except Exception as exc:
            if args.lmdb == "on":
                raise
            print(f"[LMDB] Could not prepare train cache; train_dinov.py may rebuild it later: {exc}")

    elapsed = time.time() - t0
    tagged = len(metadata)
    print(f"\nDone in {elapsed:.1f}s. Tagged: {tagged}, errors: {errors}")
    print(f"Wrote combined metadata to: {out_file}")
    if not args.no_train_lmdb and args.lmdb != "off":
        print(f"LMDB for train_dinov.py: {args.lmdb_dir}")
    if args.save_txt:
        print("Per-image .txt tag files were also written.")
    print("\nYou can now train the DINOv2 adapter or a LoRA on this dataset.")

if __name__ == "__main__":
    main()
