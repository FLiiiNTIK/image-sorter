from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .lmdb_cache import ImageLmdbReader
from .utils import VALID_EXTS, emit_log, format_size, get_paths, read_json, safe_filename, write_json


@dataclass
class DatasetRecord:
    path: Path
    image: Image.Image | None
    error: str | None = None


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], lmdb_dir: str | Path | None = None, lmdb_source: str | Path | None = None):
        self.paths = [Path(path) for path in paths]
        self.lmdb_dir = Path(lmdb_dir) if lmdb_dir else None
        self.lmdb_source = Path(lmdb_source).expanduser().resolve() if lmdb_source else None
        self._lmdb_reader: ImageLmdbReader | None = None

    def _reader(self) -> ImageLmdbReader | None:
        if self.lmdb_dir is None or self.lmdb_source is None:
            return None
        if self._lmdb_reader is None:
            self._lmdb_reader = ImageLmdbReader(self.lmdb_dir, self.lmdb_source)
        return self._lmdb_reader

    def close(self) -> None:
        if self._lmdb_reader is not None:
            self._lmdb_reader.close()
            self._lmdb_reader = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> DatasetRecord:
        path = self.paths[idx]
        try:
            reader = self._reader()
            if reader is None:
                with Image.open(path) as raw:
                    image = raw.convert("RGB")
            else:
                image = reader.open_image(path)
            image.thumbnail((512, 512), Image.Resampling.BILINEAR)
            return DatasetRecord(path=path, image=image)
        except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as exc:
            return DatasetRecord(path=path, image=None, error=str(exc))


def pil_collate(batch: list[DatasetRecord]) -> list[DatasetRecord]:
    return batch


def parse_blocked_subfolders(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[\n;,]+", str(value))
    blocked: list[str] = []
    for item in raw_items:
        cleaned = item.strip().strip('"').strip("'")
        if cleaned:
            blocked.append(cleaned)
    return blocked


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _blocked_matchers(source_path: Path, blocked_subfolders: str | list[str] | tuple[str, ...] | None) -> tuple[list[Path], list[str]]:
    absolute_or_relative_paths: list[Path] = []
    names: list[str] = []
    for item in parse_blocked_subfolders(blocked_subfolders):
        normalized = item.replace("\\", "/").strip("/")
        path_item = Path(item).expanduser()
        if path_item.is_absolute():
            absolute_or_relative_paths.append(path_item.resolve())
        elif "/" in normalized:
            absolute_or_relative_paths.append((source_path / normalized).resolve())
        else:
            names.append(normalized.casefold())
    return absolute_or_relative_paths, names


def _is_blocked_path(path: Path, blocked_paths: list[Path], blocked_names: list[str]) -> bool:
    resolved = path.resolve()
    if any(_is_relative_to(resolved, blocked_path) for blocked_path in blocked_paths):
        return True
    if blocked_names and any(part.casefold() in blocked_names for part in resolved.parts):
        return True
    return False


def scan_images(
    source: str | Path,
    recursive: bool,
    blocked_subfolders: str | list[str] | tuple[str, ...] | None = None,
    cache_status_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> list[Path]:
    source_path = Path(source).expanduser().resolve()
    paths_cfg = get_paths()
    cache_file = paths_cfg.scan_cache_file
    try:
        mtime = str(source_path.stat().st_mtime)
    except OSError:
        mtime = "0"
    blocked_paths, blocked_names = _blocked_matchers(source_path, blocked_subfolders)
    blocked_key = "|".join([str(path) for path in blocked_paths] + blocked_names)
    cache_key = f"{source_path}|{recursive}|{mtime}|blocked={blocked_key}"

    cache = read_json(cache_file, {}, log_callback)
    if cache.get("key") == cache_key:
        cached_paths = [Path(item) for item in cache.get("paths", []) if Path(item).is_file()]
        if cache_status_callback is not None:
            cache_status_callback(f"cached ({len(cached_paths)})")
        return cached_paths

    if cache_status_callback is not None:
        cache_status_callback("scanning...")
    if blocked_paths or blocked_names:
        emit_log(log_callback, f"Blocked subfolders: {len(blocked_paths) + len(blocked_names)}")

    if recursive:
        found_paths: list[Path] = []
        for root, dirs, files in os.walk(source_path):
            root_path = Path(root)
            dirs[:] = [
                dirname
                for dirname in dirs
                if not _is_blocked_path(root_path / dirname, blocked_paths, blocked_names)
            ]
            for filename in files:
                path = root_path / filename
                if path.suffix.lower() in VALID_EXTS and not _is_blocked_path(path, blocked_paths, blocked_names):
                    found_paths.append(path)
    else:
        found_paths = [
            path
            for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTS and not _is_blocked_path(path, blocked_paths, blocked_names)
        ]

    payload = {"key": cache_key, "paths": [str(path) for path in found_paths]}
    write_json(paths_cfg.cache_dir / "scan_cache.json", payload, log_callback)
    if cache_status_callback is not None:
        cache_status_callback(f"scanned ({len(found_paths)})")
    return found_paths


def clear_cache(log_callback: Callable[[str], None] | None = None, lmdb_dir: str | Path | None = None) -> tuple[list[str], str | None]:
    paths = get_paths()
    removed: list[str] = []
    error: str | None = None
    for label, path in (
        ("scan", paths.resolve_cache_file("scan_cache.json")),
        ("features", paths.resolve_cache_file("feat_cache.pt")),
    ):
        try:
            if path.exists():
                size = path.stat().st_size if label == "features" else 0
                path.unlink()
                removed.append(f"{label} ({format_size(size)})" if label == "features" else label)
        except Exception as exc:
            error = str(exc)
            emit_log(log_callback, f"Warning: failed to clear cache file {path.name}: {exc}")
    lmdb_paths = [paths.cache_dir / "dinov2_train_lmdb"]
    if lmdb_dir:
        custom = Path(lmdb_dir).expanduser()
        if custom not in lmdb_paths:
            lmdb_paths.append(custom)
    for path in lmdb_paths:
        try:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                removed.append("lmdb")
        except Exception as exc:
            error = str(exc)
            emit_log(log_callback, f"Warning: failed to clear LMDB cache {path}: {exc}")
    return removed, error
