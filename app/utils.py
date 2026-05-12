from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NEW_MODELS_DIR = DATA_DIR / "models"
NEW_WEIGHTS_DIR = DATA_DIR / "weights"
NEW_CACHE_DIR = DATA_DIR / "cache"
LEGACY_MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_WEIGHTS_DIR = PROJECT_ROOT / "weights"
LEGACY_CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
CONTENT_NEG_ANCHOR = "abstract pattern, random noise, blank image, nothing recognizable"
STYLE_NEG_ANCHOR = "unknown style, unrecognizable technique, random visual"
MODEL_SIZES_GB = {
    "ai": 0.4,
    "sdxl": 0.35,
    "siglip": 4.5,
    "dino": 0.35,
    "tagger": 0.5,
    "florence": 1.6,
}

SETTINGS_BOOL_KEYS = [
    "sort_ai_human",
    "use_sdxl_vote",
    "sort_content",
    "sort_style",
    "sort_grouping",
    "sort_dedup",
    "move_files",
    "recursive_scan",
    "gen_metadata",
    "use_dino_adapter",
    "optimize_models",
    "group_sequences",
    "use_florence",
    "sort_tagger_filter",
    "route_uncertain",
    "dry_run",
    "auto_vram_batch",
    "unload_inactive_models",
    "remember_cluster_names",
    "canonical_folder_names",
    "hierarchical_folder_names",
    "global_name_optimization",
    "character_aware_recursive",
    "character_create_multiple",
    "rebuild_lmdb",
    "metadata_skip_existing_captions",
    "train_manga",
    "train_finetune",
    "train_fresh",
    "train_no_auto_optimize",
    "train_cpu",
]
SETTINGS_STRING_KEYS = [
    "source_dir",
    "target_dir",
    "content_tags",
    "style_tags",
    "group_desc_var",
    "content_neg_anchor",
    "style_neg_anchor",
    "tag_preset_var",
    "tagger_engine_var",
    "florence_mode",
    "florence_char",
    "florence_profile",
    "tagger_filter_tags",
    "quick_start_var",
    "config_preset_var",
    "name_uniqueness_level",
    "lmdb_mode",
    "lmdb_cache_dir",
    "blocked_subfolders",
    "vram_profile",
    "train_metadata_source",
    "train_augment_mode",
]
SETTINGS_NUMBER_KEYS = [
    "content_min_conf",
    "style_min_conf",
    "group_threshold",
    "dedup_threshold",
    "batch_size_var",
    "dino_batch_size",
    "tagger_batch_size",
    "vram_limit_gb",
    "meta_max_per_folder",
    "meta_max_tokens",
    "meta_tags_per_image",
    "camie_threshold",
    "florence_batch_size",
    "florence_max_side",
    "uncertainty_threshold",
    "max_folders_created",
    "character_min_score",
    "character_margin",
    "character_max_multi",
    "metadata_save_every",
    "metadata_florence_per_folder",
    "train_epochs",
    "train_batch_size",
    "train_grad_accum",
    "train_lr",
    "train_min_tag_score",
    "train_max_side",
]

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def emit_log(log_callback: Callable[[str], None] | None, message: str) -> None:
    if log_callback is not None:
        log_callback(message)
    else:
        logger.info(message)


def emit_progress(progress_callback: Callable[[float], None] | None, value: float) -> None:
    if progress_callback is not None:
        progress_callback(value)


def format_size(num_bytes: float | int, precision: int = 2) -> str:
    """Format bytes using binary steps, matching how Windows reports drive sizes."""
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.{precision}f} {unit}"
        value /= 1024.0


def log_exception(
    log_callback: Callable[[str], None] | None,
    prefix: str,
    exc: BaseException,
    include_traceback: bool = True,
) -> None:
    emit_log(log_callback, f"{prefix}: {exc}")
    if include_traceback:
        emit_log(log_callback, traceback.format_exc())


def safe_filename(value: str) -> str:
    """Strip characters unsafe for filesystem folder/file names."""
    return _UNSAFE_CHARS.sub("_", value).strip(" .")[:200] or "unnamed"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any, log_callback: Callable[[str], None] | None = None) -> Any:
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read JSON from %s", path, exc_info=exc)
        emit_log(log_callback, f"Warning: could not read {path.name}: {exc}")
        return default


def write_json(path: Path, payload: Any, log_callback: Callable[[str], None] | None = None) -> bool:
    try:
        ensure_directory(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return True
    except Exception as exc:
        logger.warning("Failed to write JSON to %s", path, exc_info=exc)
        emit_log(log_callback, f"Warning: could not write {path.name}: {exc}")
        return False


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    data_dir: Path
    models_dir: Path
    weights_dir: Path
    cache_dir: Path
    logs_dir: Path
    legacy_models_dir: Path
    legacy_weights_dir: Path
    legacy_cache_dir: Path

    @property
    def ai_human_model_path(self) -> Path:
        return self.models_dir / "ai-vs-human-image-detector"

    @property
    def sdxl_detector_path(self) -> Path:
        return self.models_dir / "sdxl-detector"

    @property
    def siglip2_model_path(self) -> Path:
        return self.models_dir / "siglip2-so400m-patch16-512"

    @property
    def dinov2_model_path(self) -> Path:
        return self.models_dir / "dinov2-base"

    @property
    def wd_tagger_path(self) -> Path:
        return self.models_dir / "wd-eva02-large-tagger-v3"

    @property
    def camie_tagger_path(self) -> Path:
        return self.models_dir / "camie-tagger-v2"

    @property
    def florence_model_path(self) -> Path:
        return self.models_dir / "florence-2-large-promptgen-v2"

    @property
    def adapter_path(self) -> Path:
        return self.weights_dir / "best_adapter.pth"

    @property
    def manga_adapter_path(self) -> Path:
        return self.weights_dir / "manga_adapter.pth"

    @property
    def all_minilm_path(self) -> Path:
        legacy = self.project_root / "all-MiniLM-L6-v2"
        in_models = self.models_dir / "all-MiniLM-L6-v2"
        return in_models if in_models.exists() else legacy

    def resolve_cache_file(self, filename: str) -> Path:
        current = self.cache_dir / filename
        legacy = self.legacy_cache_dir / filename
        if current.exists():
            return current
        if legacy.exists():
            return legacy
        return current

    @property
    def settings_file(self) -> Path:
        return self.resolve_cache_file("settings.json")

    @property
    def scan_cache_file(self) -> Path:
        return self.resolve_cache_file("scan_cache.json")

    @property
    def feat_cache_file(self) -> Path:
        return self.resolve_cache_file("feat_cache.pt")


def _pick_storage_dir(primary: Path, legacy: Path) -> Path:
    if primary.exists() and any(primary.iterdir()):
        return primary
    if legacy.exists():
        return legacy
    ensure_directory(primary)
    return primary


def get_paths() -> AppPaths:
    ensure_directory(DATA_DIR)
    ensure_directory(NEW_MODELS_DIR)
    ensure_directory(NEW_WEIGHTS_DIR)
    ensure_directory(NEW_CACHE_DIR)
    ensure_directory(LOGS_DIR)
    return AppPaths(
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        models_dir=_pick_storage_dir(NEW_MODELS_DIR, LEGACY_MODELS_DIR),
        weights_dir=_pick_storage_dir(NEW_WEIGHTS_DIR, LEGACY_WEIGHTS_DIR),
        cache_dir=NEW_CACHE_DIR,
        logs_dir=LOGS_DIR,
        legacy_models_dir=LEGACY_MODELS_DIR,
        legacy_weights_dir=LEGACY_WEIGHTS_DIR,
        legacy_cache_dir=LEGACY_CACHE_DIR,
    )


def install_windows_dpi_awareness() -> None:
    if str(getattr(__import__("sys"), "platform", "")) != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception as exc:
        logger.debug("Unable to set DPI awareness", exc_info=exc)


def parse_tags(tags_str: str, log_callback: Callable[[str], None] | None = None) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    pos: list[tuple[str, float]] = []
    neg: list[tuple[str, float]] = []
    for raw_tag in tags_str.split(","):
        tag = raw_tag.strip()
        if not tag:
            continue

        weight = 1.0
        if ":" in tag:
            tag_name, raw_weight = tag.rsplit(":", 1)
            try:
                weight = float(raw_weight)
                tag = tag_name.strip()
            except ValueError as exc:
                emit_log(log_callback, f"Warning: invalid tag weight '{raw_weight}' for '{tag_name.strip()}': {exc}")
                tag = tag_name.strip()

        if not tag:
            continue
        if tag.startswith("-"):
            neg.append((tag[1:].strip(), weight))
        else:
            pos.append((tag, weight))
    return pos, neg


def make_ensembles(tag: str) -> list[str]:
    return [f"a photo of {tag}", f"an image of {tag}", f"a picture of {tag}", tag]


def detect_runtime(precision_choice: str) -> dict[str, Any]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_vram = 0
    if device == "cuda":
        try:
            total_vram = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            total_vram = 0
    if device == "cpu":
        use_amp = False
        amp_dtype = torch.float32
    elif precision_choice == "fp32":
        use_amp = False
        amp_dtype = torch.float32
    elif precision_choice == "fp16":
        use_amp = True
        amp_dtype = torch.float16
    elif precision_choice == "bf16":
        use_amp = True
        amp_dtype = torch.bfloat16
    else:
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dtype_name = {
        torch.bfloat16: "BF16",
        torch.float16: "FP16",
        torch.float32: "FP32",
    }.get(amp_dtype, "FP32")
    return {
        "device": device,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "dtype_name": dtype_name,
        "total_vram": total_vram,
        "status_text": f"Ready - {'CUDA' if device == 'cuda' else 'CPU'} ({dtype_name})",
    }


def resolve_vram_limit_bytes(profile: str, custom_gb: float, total_vram: int) -> int:
    """Return a conservative CUDA working limit in bytes. 0 means disabled."""
    if total_vram <= 0:
        return 0
    profile_key = str(profile or "Auto").strip().lower()
    if profile_key == "custom":
        requested = max(0.0, float(custom_gb or 0.0))
        if requested <= 0:
            return int(total_vram * 0.90)
        return int(min(requested * 1024**3, total_vram * 0.96))
    match = re.search(r"(\d+(?:\.\d+)?)", profile_key)
    if match:
        requested = float(match.group(1)) * 1024**3
        return int(min(requested, total_vram * 0.96))
    return int(total_vram * 0.90)


def load_settings(log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    paths = get_paths()
    current = read_json(paths.settings_file, {}, log_callback)
    if current:
        return current
    legacy = paths.legacy_cache_dir / "settings.json"
    if legacy == paths.settings_file:
        return current
    return read_json(legacy, {}, log_callback)


def save_settings(payload: dict[str, Any], log_callback: Callable[[str], None] | None = None) -> bool:
    paths = get_paths()
    return write_json(paths.cache_dir / "settings.json", payload, log_callback)


class LoadingBar:
    def __init__(self, set_fn: Callable[[float], None], start_pct: float, end_pct: float, est_seconds: float):
        self._set = set_fn
        self._start = start_pct
        self._end = end_pct
        self._estimate = max(est_seconds, 1.0)
        self._done = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        start_time = time.time()
        while not self._done.is_set():
            frac = 1.0 - math.exp(-2.5 * (time.time() - start_time) / self._estimate)
            self._set(self._start + frac * (self._end - self._start) * 0.95)
            time.sleep(0.08)

    def complete(self) -> None:
        self._done.set()
        self._set(self._end)
