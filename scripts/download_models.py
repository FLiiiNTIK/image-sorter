from __future__ import annotations

import sys

from app.utils import get_paths

PATHS = get_paths()
MODELS_DIR = PATHS.data_dir / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ai-vs-human-image-detector": {"repo": "Ateeqq/ai-vs-human-image-detector", "type": "transformers", "expected_file": "config.json"},
    "sdxl-detector": {"repo": "umm-maybe/AI-image-detector", "type": "transformers", "expected_file": "config.json"},
    "siglip2-so400m-patch16-512": {"repo": "google/siglip2-so400m-patch16-512", "type": "transformers", "expected_file": "config.json"},
    "dinov2-base": {"repo": "facebook/dinov2-base", "type": "transformers", "expected_file": "config.json"},
    "wd-eva02-large-tagger-v3": {"repo": "SmilingWolf/wd-eva02-large-tagger-v3", "type": "hf_hub", "files": ["model.onnx", "selected_tags.csv"]},
    "camie-tagger-v2": {"repo": "Camais03/camie-tagger-v2", "type": "hf_hub", "files": ["camie-tagger-v2.onnx", "camie-tagger-v2-metadata.json"]},
    "florence-2-large-promptgen-v2": {"repo": "MiaoshouAI/Florence-2-large-PromptGen-v2.0", "type": "transformers", "expected_file": "config.json"},
}


def migrate_existing_models() -> None:
    import shutil

    for folder_name in MODELS:
        legacy_path = PATHS.legacy_models_dir / folder_name
        target_path = MODELS_DIR / folder_name
        if legacy_path.exists() and not target_path.exists():
            print(f"Moving {folder_name} into data/models/ ...")
            try:
                shutil.move(str(legacy_path), str(target_path))
            except Exception as exc:
                print(f"Warning: failed to move {folder_name}: {exc}")


def check_and_download_all() -> None:
    migrate_existing_models()
    missing: list[tuple[str, dict[str, object]]] = []
    for folder_name, info in MODELS.items():
        folder_path = MODELS_DIR / folder_name
        if info["type"] == "transformers":
            if not (folder_path / str(info["expected_file"])).exists():
                missing.append((folder_name, info))
        else:
            if any(not (folder_path / str(filename)).exists() for filename in info["files"]):
                missing.append((folder_name, info))

    if not missing:
        return

    print("\n" + "=" * 50)
    print("Missing models detected. Starting download...")
    print("=" * 50 + "\n")

    from huggingface_hub import hf_hub_download, snapshot_download

    for folder_name, info in missing:
        folder_path = MODELS_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"[{folder_name}] Downloading from {info['repo']}...")
        try:
            if info["type"] == "transformers":
                snapshot_download(repo_id=str(info["repo"]), local_dir=str(folder_path), ignore_patterns=["*.msgpack", "*.h5"])
            else:
                for filename in info["files"]:
                    hf_hub_download(repo_id=str(info["repo"]), filename=str(filename), local_dir=str(folder_path))
            print(f"[{folder_name}] Download complete.\n")
        except Exception as exc:
            print(f"Download failed for {folder_name}: {exc}")
            sys.exit(1)

    print("=" * 50)
    print("All required models are available.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    check_and_download_all()
