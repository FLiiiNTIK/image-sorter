from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .utils import emit_log, format_size, get_paths, read_json

try:
    import lmdb
except Exception:
    lmdb = None


DEFAULT_LMDB_NAME = "dinov2_train_lmdb"


def default_lmdb_dir() -> Path:
    return get_paths().cache_dir / DEFAULT_LMDB_NAME


def configured_lmdb_dir(log_callback: Callable[[str], None] | None = None) -> Path:
    paths = get_paths()
    settings = read_json(paths.settings_file, {}, log_callback)
    if isinstance(settings, dict):
        value = str(settings.get("lmdb_cache_dir", "")).strip()
        if value:
            return resolve_lmdb_dir(value)
    return default_lmdb_dir()


def resolve_lmdb_dir(value: str | Path | None) -> Path:
    text = str(value or "").strip()
    return Path(text).expanduser().resolve() if text else default_lmdb_dir()


def clean_rel_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def path_key(rel_path: str | Path) -> bytes:
    return f"path:{clean_rel_path(rel_path)}".encode("utf-8")


def image_lmdb_hash(image_paths: list[Path], source_dir: Path) -> str:
    records = []
    for path in sorted((Path(item) for item in image_paths), key=lambda p: clean_rel_path(p)):
        try:
            rel_path = clean_rel_path(path.resolve().relative_to(source_dir.resolve()))
        except Exception:
            rel_path = clean_rel_path(path.name)
        try:
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        records.append((rel_path, size, mtime))
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _lmdb_map_size(image_paths: list[Path], lmdb_dir: Path) -> int:
    total = 0
    for path in image_paths:
        try:
            total += path.stat().st_size
        except OSError:
            pass
    data_file = lmdb_dir / "data.mdb"
    existing_size = data_file.stat().st_size if data_file.exists() else 0
    return max(
        256 << 20,
        existing_size + int(total * 1.15) + (512 << 20),
        int(total * 1.25) + (512 << 20),
    )


def _disk_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def _lmdb_unavailable(message: str, mode: str, log_callback: Callable[[str], None] | None = None) -> None:
    if mode == "on":
        raise RuntimeError(message)
    emit_log(log_callback, message + " Continuing without LMDB.")


class ImageLmdbReader:
    def __init__(self, lmdb_dir: str | Path, source_dir: str | Path):
        self.lmdb_dir = Path(lmdb_dir)
        self.source_dir = Path(source_dir).expanduser().resolve()
        self._env = None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open(self):
        if lmdb is None:
            return None
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_dir),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
        return self._env

    def rel_path(self, path: str | Path) -> str:
        item = Path(path).expanduser().resolve()
        try:
            return clean_rel_path(item.relative_to(self.source_dir))
        except Exception:
            return clean_rel_path(item.name)

    def read_bytes(self, path: str | Path) -> bytes | None:
        env = self._open()
        if env is None:
            return None
        with env.begin(write=False) as txn:
            value = txn.get(path_key(self.rel_path(path)))
        return bytes(value) if value is not None else None

    def open_image(self, path: str | Path) -> Image.Image:
        payload = self.read_bytes(path)
        if payload is None:
            with Image.open(path) as raw:
                return raw.convert("RGB")
        with Image.open(io.BytesIO(payload)) as raw:
            return raw.convert("RGB")


def prepare_image_lmdb(
    image_paths: list[Path],
    source_dir: str | Path,
    lmdb_dir: str | Path | None,
    mode: str = "auto",
    rebuild: bool = False,
    log_callback: Callable[[str], None] | None = None,
) -> ImageLmdbReader | None:
    mode = str(mode or "auto").strip().lower()
    if mode == "off":
        return None
    if lmdb is None:
        message = "LMDB is not installed. Install lmdb or switch LMDB cache to Off."
        if mode == "on":
            raise RuntimeError(message)
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None

    source = Path(source_dir).expanduser().resolve()
    cache_dir = resolve_lmdb_dir(lmdb_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sorted_paths = sorted([Path(path).expanduser().resolve() for path in image_paths], key=lambda p: clean_rel_path(p))
    expected_hash = image_lmdb_hash(sorted_paths, source)
    meta_path = cache_dir / "extract_meta.json"

    if not rebuild and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("hash") == expected_hash and int(meta.get("count", 0)) == len(sorted_paths):
                emit_log(log_callback, f"LMDB image cache ready: {cache_dir}")
                return ImageLmdbReader(cache_dir, source)
        except Exception as exc:
            emit_log(log_callback, f"Warning: LMDB metadata is invalid, rebuilding: {exc}")

    map_size = _lmdb_map_size(sorted_paths, cache_dir)
    free_bytes = _disk_free_bytes(cache_dir)
    if free_bytes and map_size > int(free_bytes * 0.9):
        message = (
            f"LMDB cache directory does not have enough free space: {cache_dir}. "
            f"Need up to {format_size(map_size)}, free {format_size(free_bytes)}. "
            "Choose a larger Cache dir or switch LMDB to Off."
        )
        if mode == "on":
            raise RuntimeError(message)
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None

    emit_log(log_callback, f"Building LMDB image cache: {cache_dir} ({len(sorted_paths)} images, map={format_size(map_size)})")
    try:
        env = lmdb.open(
            str(cache_dir),
            map_size=map_size,
            subdir=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
    except Exception as exc:
        message = f"LMDB could not be opened at {cache_dir}: {exc}"
        if mode == "on":
            raise RuntimeError(message) from exc
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None
    txn = env.begin(write=True)
    written = skipped = 0
    try:
        for index, path in enumerate(sorted_paths):
            try:
                rel_path = clean_rel_path(path.relative_to(source))
            except Exception:
                rel_path = clean_rel_path(path.name)
            try:
                txn.put(path_key(rel_path), path.read_bytes())
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
        message = f"LMDB build failed at {cache_dir}. Choose a larger Cache dir or switch LMDB to Off."
        if mode == "on":
            raise RuntimeError(message)
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None
    env.sync()
    env.close()

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hash": expected_hash,
                "count": len(sorted_paths),
                "written": written,
                "skipped": skipped,
                "source_data_dir": str(source),
                "created_by": "main_app",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    emit_log(log_callback, f"LMDB image cache built: written={written}, skipped={skipped}")
    return ImageLmdbReader(cache_dir, source)
