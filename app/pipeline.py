from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import ImagePathDataset, pil_collate, scan_images
from .lmdb_cache import ImageLmdbReader, prepare_image_lmdb, resolve_lmdb_dir
from .models import ModelManager
from .utils import (
    CONTENT_NEG_ANCHOR,
    SETTINGS_BOOL_KEYS,
    SETTINGS_NUMBER_KEYS,
    SETTINGS_STRING_KEYS,
    STYLE_NEG_ANCHOR,
    emit_log,
    emit_progress,
    get_paths,
    parse_tags,
    read_json,
    safe_filename,
    write_json,
)

WD_NUDE_TAGS = {"nude", "unskirt", "nipples", "pussy", "penis", "breasts", "cleavage", "underwear", "bikini", "swimsuit", "naked"}
WD_P_STAND_TAGS = {"standing", "on_one_leg"}
WD_P_SIT_TAGS = {"sitting", "kneeling", "squatting", "seiza", "crossed_legs"}
WD_P_LIE_TAGS = {"lying", "on_stomach", "on_back", "on_side"}
WD_C_SHIRT_TAGS = {"shirt", "t-shirt", "collared_shirt", "blouse", "tank_top", "sweater", "hoodie"}
WD_C_JACKET_TAGS = {"jacket", "coat", "cardigan", "suit_jacket", "cloak"}
WD_C_DRESS_TAGS = {"dress", "sundress", "wedding_dress"}
WD_C_SKIRT_TAGS = {"skirt", "pleated_skirt", "miniskirt", "pencil_skirt"}
WD_C_PANTS_TAGS = {"pants", "jeans", "shorts", "sweatpants", "trousers", "leggings"}
WD_SC_INDOOR = {"indoors", "bedroom", "classroom", "office", "kitchen", "bathroom", "living_room"}
WD_SC_OUTDOOR = {"outdoors", "sky", "cloud", "street", "road", "bridge"}
WD_SC_NATURE = {"nature", "forest", "mountain", "ocean", "lake", "river", "field", "tree", "flower"}
WD_SC_URBAN = {"city", "building", "skyscraper", "cityscape", "town", "alley"}
WD_H_BLACK = {"black_hair"}
WD_H_BLONDE = {"blonde_hair", "light_brown_hair"}
WD_H_BROWN = {"brown_hair"}
WD_H_RED = {"red_hair", "pink_hair"}
WD_H_BLUE = {"blue_hair", "aqua_hair"}
WD_H_WHITE = {"white_hair", "grey_hair", "silver_hair"}
WD_H_GREEN = {"green_hair"}
WD_H_PURPLE = {"purple_hair"}
WD_EX_HAPPY = {"smile", "grin", "laughing", "open_mouth", ":d"}
WD_EX_SAD = {"crying", "tears", "frown", "sad"}
WD_EX_ANGRY = {"angry", "furrowed_brow", "clenched_teeth"}
WD_EX_NEUTRAL = {"expressionless", "closed_mouth", "serious"}

_SHARED_MANAGER: ModelManager | None = None


@dataclass
class FlorenceQueueRecord:
    path: Path
    prompt_tags: str
    image: Image.Image | None
    error: str | None = None


class FlorenceQueueDataset(Dataset):
    def __init__(
        self,
        queued: list[tuple[Path, str]],
        max_side: int = 768,
        lmdb_dir: str | Path | None = None,
        lmdb_source: str | Path | None = None,
    ):
        self.queued = [(Path(path), str(prompt_tags)) for path, prompt_tags in queued]
        self.max_side = max(256, int(max_side))
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
        return len(self.queued)

    def __getitem__(self, idx: int) -> FlorenceQueueRecord:
        path, prompt_tags = self.queued[idx]
        try:
            reader = self._reader()
            if reader is None:
                with Image.open(path) as raw:
                    image = raw.convert("RGB")
            else:
                image = reader.open_image(path)
            image.thumbnail((self.max_side, self.max_side), Image.Resampling.BILINEAR)
            return FlorenceQueueRecord(path=path, prompt_tags=prompt_tags, image=image)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return FlorenceQueueRecord(path=path, prompt_tags=prompt_tags, image=None, error=str(exc))


def florence_collate(batch: list[FlorenceQueueRecord]) -> list[FlorenceQueueRecord]:
    return batch


def get_model_manager() -> ModelManager:
    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = ModelManager()
    return _SHARED_MANAGER


def get_runtime_status(precision_choice: str = "auto") -> str:
    return get_model_manager().get_status_text(precision_choice)


def run_pipeline(config: dict[str, Any], progress_callback: Callable[[float], None] | None = None, log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    runner = PipelineRunner(config, progress_callback=progress_callback, log_callback=log_callback)
    return runner.run()


def run_metadata_export(config: dict[str, Any], progress_callback: Callable[[float], None] | None = None, log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    runner = PipelineRunner(config, progress_callback=progress_callback, log_callback=log_callback)
    return runner.run_metadata_export()


def run_ai_search(config: dict[str, Any], progress_callback: Callable[[float], None] | None = None, log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    manager = get_model_manager()
    manager.set_callbacks(progress_callback, log_callback)
    manager.update_runtime(config)
    search_query = str(config["query"]).strip()
    target_dir = Path(config["target_dir"]).expanduser().resolve()
    top_k = int(config.get("top_k", 20))
    copy_results = bool(config.get("copy_results", True))
    selected_paths_raw = config.get("selected_paths")
    selected_paths = {
        str(Path(path).expanduser().resolve())
        for path in selected_paths_raw
    } if isinstance(selected_paths_raw, list) else None
    if not manager.ensure_siglip_loaded():
        return {"ok": False, "message": "SigLIP2 failed to load."}

    feat_cache_file = get_paths().feat_cache_file
    if not feat_cache_file.exists():
        return {"ok": False, "message": f"No feature cache found at {feat_cache_file}"}

    try:
        feat_cache = torch.load(str(feat_cache_file), map_location="cpu", weights_only=True)
        siglip_cache = {key: value for key, value in feat_cache.items() if str(key).startswith("siglip_")}
        if not siglip_cache:
            return {"ok": False, "message": "No SigLIP embeddings found in cache."}

        with torch.inference_mode():
            query_feat = manager.siglip_embed_texts([search_query])[0].cpu()

        paths: list[Path] = []
        feats: list[torch.Tensor] = []
        for key, value in siglip_cache.items():
            without_prefix = str(key)[len("siglip_") :]
            last_us = without_prefix.rfind("_")
            path_text = without_prefix[:last_us] if last_us > 0 else without_prefix
            paths.append(Path(path_text))
            feats.append(value.view(1, -1))

        all_feats = F.normalize(torch.cat(feats, dim=0), p=2, dim=1)
        sims = torch.mm(all_feats, query_feat.unsqueeze(1)).squeeze(1)
        top_k = min(top_k, len(paths))
        scores, indices = torch.topk(sims, top_k)

        result_items: list[dict[str, Any]] = []
        for index, item in enumerate(indices):
            source_path = paths[item.item()]
            score = float(scores[index].item())
            result_items.append(
                {
                    "rank": index + 1,
                    "path": str(source_path),
                    "score": score,
                    "exists": source_path.exists(),
                }
            )

        copied = 0
        result_dir: Path | None = None
        if copy_results:
            result_dir = target_dir / "AI_Search_Results" / safe_filename(search_query)[:50]
            result_dir.mkdir(parents=True, exist_ok=True)
            for item in result_items:
                source_path = Path(str(item["path"]))
                if not source_path.exists():
                    continue
                if selected_paths is not None and str(source_path.resolve()) not in selected_paths:
                    continue
                destination = result_dir / f"{int(item['rank']):03d}_{float(item['score']):.2f}{source_path.suffix}"
                shutil.copy2(source_path, destination)
                copied += 1

        return {
            "ok": True,
            "message": f"Copied {copied} files to {result_dir}" if copy_results else f"Found {len(result_items)} matches",
            "result_dir": str(result_dir) if result_dir is not None else "",
            "copied": copied,
            "results": result_items,
        }
    except Exception as exc:
        emit_log(log_callback, f"AI search failed: {exc}")
        emit_log(log_callback, traceback.format_exc())
        return {"ok": False, "message": str(exc)}


def get_run_history(limit: int = 30) -> list[dict[str, Any]]:
    history_file = get_paths().logs_dir / "run_history.json"
    if not history_file.exists():
        return []
    try:
        with history_file.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except Exception:
        return []
    if not isinstance(entries, list):
        return []
    cleaned = [entry for entry in entries if isinstance(entry, dict)]
    cleaned.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return cleaned[: max(1, int(limit))]


def undo_last_move_run(target_dir: str | Path, log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    target = Path(target_dir).expanduser().resolve()
    journal_file = target / ".undo_last_move.json"
    if not journal_file.exists():
        return {"ok": False, "message": f"Undo journal not found: {journal_file}"}

    payload = read_json(journal_file, {}, log_callback)
    operations = payload.get("operations", [])
    if not isinstance(operations, list) or not operations:
        return {"ok": False, "message": "Undo journal is empty."}

    restored = skipped = errors = 0
    remaining: list[dict[str, str]] = []
    for operation in reversed(operations):
        src = Path(str(operation.get("src", ""))).expanduser()
        dst = Path(str(operation.get("dst", ""))).expanduser()
        if not dst.exists():
            skipped += 1
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            destination = src
            suffix_counter = 1
            while destination.exists():
                destination = src.with_name(f"{src.stem}_restored_{suffix_counter}{src.suffix}")
                suffix_counter += 1
            shutil.move(str(dst), str(destination))
            restored += 1
        except Exception as exc:
            errors += 1
            remaining.append({"src": str(src), "dst": str(dst)})
            emit_log(log_callback, f"Undo warning for {dst.name}: {exc}")

    if remaining:
        payload["operations"] = list(reversed(remaining))
        write_json(journal_file, payload, log_callback)
    else:
        archived = journal_file.with_name(f".undo_last_move_applied_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            shutil.move(str(journal_file), str(archived))
        except Exception:
            pass

    return {
        "ok": errors == 0,
        "message": f"Undo finished. Restored: {restored}, Skipped: {skipped}, Errors: {errors}",
        "restored": restored,
        "skipped": skipped,
        "errors": errors,
    }


class PipelineRunner:
    def __init__(
        self,
        config: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = dict(config)
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.manager = get_model_manager()
        self.manager.set_callbacks(progress_callback, log_callback)
        self.paths = get_paths()
        self.cancel_event: Event = self.config.get("cancel_event") or Event()
        self.stop_after_batch_event: Event = self.config.get("stop_after_batch_event") or Event()
        self.scan_status_callback = self.config.get("scan_status_callback")
        self.eta_callback: Callable[[dict[str, Any]], None] | None = self.config.get("eta_callback")
        self._timings: dict[str, float] = {}
        self._eta_speed_ema: float | None = None
        self._eta_stage = ""
        self._last_eta_emit = 0.0
        self._eta_last_completed = 0
        self._eta_last_ts = 0.0
        self._eta_update_count = 0
        self._stop_after_batch_triggered = False
        self._folder_uncertainty: dict[str, dict[str, Any]] = {}
        self._move_journal: list[dict[str, str]] = []
        self._move_journal_target: Path | None = None
        self._run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._image_lmdb: ImageLmdbReader | None = None
        self._image_lmdb_dir: Path | None = None
        self._image_lmdb_source: Path | None = None

    def log(self, message: str) -> None:
        emit_log(self.log_callback, message)

    def prog(self, value: float) -> None:
        emit_progress(self.progress_callback, value)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def _is_soft_stop_requested(self) -> bool:
        return self.stop_after_batch_event.is_set()

    def _mark_soft_stop(self, reason: str | None = None) -> None:
        if not self._stop_after_batch_triggered:
            self._stop_after_batch_triggered = True
            if reason:
                self.log(reason)

    def _batch_size(self, kind: str, key: str, fallback: int) -> int:
        raw = int(self.config.get(key, 0) or 0)
        requested = raw if raw > 0 else fallback
        return self.manager.resolve_batch_size(kind, requested)

    @staticmethod
    def _chunks(items: list[Any], size: int) -> list[list[Any]]:
        size = max(1, int(size))
        return [items[index : index + size] for index in range(0, len(items), size)]

    def _close_image_lmdb(self) -> None:
        if self._image_lmdb is not None:
            self._image_lmdb.close()
            self._image_lmdb = None

    def _open_image_for_path(self, path: Path) -> Image.Image:
        if self._image_lmdb is not None:
            try:
                return self._image_lmdb.open_image(path)
            except Exception as exc:
                self.log(f"Warning: LMDB image read failed for {path.name}, falling back to file: {exc}")
        with Image.open(path) as raw:
            return raw.convert("RGB")

    def _format_eta(self, eta_seconds: float) -> str:
        if eta_seconds <= 0:
            return "0s"
        mins, secs = divmod(int(eta_seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def _emit_eta(
        self,
        *,
        stage: str,
        completed: int,
        total: int,
        start_ts: float,
        force: bool = False,
    ) -> None:
        if stage != self._eta_stage:
            self._eta_stage = stage
            self._eta_speed_ema = None
            self._last_eta_emit = 0.0
            self._eta_last_completed = 0
            self._eta_last_ts = 0.0
            self._eta_update_count = 0
        now = time.time()
        elapsed = max(now - start_ts, 1e-6)
        delta_completed = max(completed - self._eta_last_completed, 0)
        delta_time = max(now - self._eta_last_ts, 1e-6) if self._eta_last_ts > 0 else 0.0
        if delta_completed > 0 and delta_time > 0:
            inst_speed = delta_completed / delta_time
        else:
            inst_speed = completed / elapsed
        alpha = 0.2
        if self._eta_speed_ema is None:
            self._eta_speed_ema = inst_speed
        else:
            self._eta_speed_ema = alpha * inst_speed + (1.0 - alpha) * self._eta_speed_ema
        speed = max(self._eta_speed_ema, 1e-6)
        remaining = max(total - completed, 0)
        eta_seconds = remaining / speed

        should_emit = force or (now - self._last_eta_emit >= 1.0)
        if not should_emit:
            return
        self._last_eta_emit = now
        self._eta_last_completed = completed
        self._eta_last_ts = now
        self._eta_update_count += 1

        warmup_mode = self._eta_update_count <= 1 and completed < total
        if warmup_mode:
            eta_text = "warming up..."
            speed_for_ui = 0.0
        else:
            eta_text = self._format_eta(eta_seconds)
            speed_for_ui = speed
        if self.eta_callback is None:
            self.log(f"  [{stage}] {completed}/{total} | {speed_for_ui:.1f} i/s | ETA: {eta_text}")
        if self.eta_callback is not None:
            try:
                self.eta_callback(
                    {
                        "stage": stage,
                        "completed": completed,
                        "total": total,
                        "speed": speed_for_ui,
                        "eta_seconds": eta_seconds,
                        "eta_text": eta_text,
                        "gpu_text": self.manager.gpu_status_text(),
                    }
                )
            except Exception:
                # ETA callback is optional UI sugar; failures should not break processing.
                pass

    def run(self) -> dict[str, Any]:
        result: dict[str, Any]
        try:
            result = self._run_inner()
        except Exception as exc:
            self.log(f"Critical error in processing thread: {exc}")
            self.log(traceback.format_exc())
            result = {"ok": False, "message": "Failed due to an unexpected error."}
        finally:
            self._close_image_lmdb()
        result = self._flush_move_journal(result)
        self._append_run_history(result)
        return result

    def run_metadata_export(self) -> dict[str, Any]:
        try:
            return self._run_metadata_export_inner()
        except Exception as exc:
            self.log(f"Critical error in metadata export: {exc}")
            self.log(traceback.format_exc())
            return {"ok": False, "message": "Metadata export failed due to an unexpected error."}
        finally:
            self._close_image_lmdb()

    def _run_metadata_export_inner(self) -> dict[str, Any]:
        source = Path(self.config["source_dir"]).expanduser().resolve()
        if not source.is_dir():
            return {"ok": False, "message": f"Source not found: {source}"}

        load_config = dict(self.config)
        load_config.update(
            {
                "sort_ai_human": False,
                "sort_content": False,
                "sort_style": False,
                "sort_grouping": False,
                "sort_dedup": False,
                "sort_tagger_filter": False,
                "character_aware_recursive": False,
                "gen_metadata": True,
            }
        )
        if not self.manager.load_for_config(load_config, self.progress_callback, self.log_callback):
            return {"ok": False, "message": "Model loading failed for metadata export."}

        self.log("Scanning images for metadata export...")
        self.prog(0)
        scan_start = time.time()
        img_paths = scan_images(
            source,
            recursive=bool(self.config.get("recursive_scan", True)),
            blocked_subfolders=self.config.get("blocked_subfolders", ""),
            cache_status_callback=self.scan_status_callback,
            log_callback=self.log_callback,
        )
        total = len(img_paths)
        if not total:
            return {"ok": False, "message": "No images found."}

        self._image_lmdb = prepare_image_lmdb(
            img_paths,
            source,
            self.config.get("lmdb_cache_dir") or resolve_lmdb_dir(None),
            mode=str(self.config.get("lmdb_mode", "auto")),
            rebuild=bool(self.config.get("rebuild_lmdb", False)),
            log_callback=self.log_callback,
        )
        if self._image_lmdb is not None:
            self._image_lmdb_dir = resolve_lmdb_dir(self.config.get("lmdb_cache_dir"))
            self._image_lmdb_source = source

        batch_size = self._batch_size("main", "batch_size_var", 8)
        tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", batch_size)
        florence_batch_size = self._batch_size("florence", "florence_batch_size", self.manager.florence_batch_size)
        tags_per_img = max(1, int(self.config.get("meta_tags_per_image", 30)))
        include_florence = bool(self.config.get("use_florence")) and self.manager.florence_model is not None
        max_side = max(256, int(self.config.get("florence_max_side", 768)))
        skip_existing_captions = bool(self.config.get("metadata_skip_existing_captions", True))
        save_every = max(0, int(self.config.get("metadata_save_every", 500)))
        florence_per_folder = max(0, int(self.config.get("metadata_florence_per_folder", 0)))
        flat_path = source / "metadata.json"
        detailed_path = source / "metadata_detailed.json"
        flat_metadata: dict[str, str] = {}
        detailed_metadata: dict[str, Any] = {
            "folder": str(source),
            "total_images": total,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lmdb_dir": str(self._image_lmdb_dir or resolve_lmdb_dir(self.config.get("lmdb_cache_dir"))),
            "images": {},
        }
        if detailed_path.exists():
            try:
                existing_detailed = read_json(detailed_path, {}, self.log_callback)
                if isinstance(existing_detailed, dict) and isinstance(existing_detailed.get("images"), dict):
                    detailed_metadata["images"].update(existing_detailed["images"])
                    self.log(f"Loaded existing detailed metadata: {len(existing_detailed['images'])} images")
            except Exception as exc:
                self.log(f"Warning: could not load existing detailed metadata: {exc}")
        if flat_path.exists():
            existing_flat = read_json(flat_path, {}, self.log_callback)
            if isinstance(existing_flat, dict):
                flat_metadata.update({str(key): str(value) for key, value in existing_flat.items() if isinstance(value, str)})

        planned_florence: set[str] = set()
        if include_florence:
            planned_counts: dict[str, int] = {}
            for path in img_paths:
                try:
                    rel_path = path.resolve().relative_to(source).as_posix()
                except Exception:
                    rel_path = path.name
                existing_item = detailed_metadata["images"].get(rel_path, {})
                if (
                    skip_existing_captions
                    and isinstance(existing_item, dict)
                    and str(existing_item.get("caption_florence2", "")).strip()
                ):
                    continue
                folder_key = str(Path(rel_path).parent).replace("\\", "/")
                if folder_key == ".":
                    folder_key = ""
                if florence_per_folder > 0 and planned_counts.get(folder_key, 0) >= florence_per_folder:
                    continue
                planned_florence.add(rel_path)
                planned_counts[folder_key] = planned_counts.get(folder_key, 0) + 1

        def flush_metadata(reason: str = "") -> None:
            detailed_metadata["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            detailed_metadata["total_images"] = total
            with flat_path.open("w", encoding="utf-8") as handle:
                json.dump(flat_metadata, handle, ensure_ascii=False, indent=2)
            with detailed_path.open("w", encoding="utf-8") as handle:
                json.dump(detailed_metadata, handle, ensure_ascii=False, indent=2)
            if reason:
                self.log(f"Metadata checkpoint saved ({reason}): {len(flat_metadata)} images")

        processed = errors = captions_generated = captions_skipped = 0
        folder_caption_counts: dict[str, int] = {}
        export_start = time.time()
        self.log(f"Generating source metadata for {total} images...")
        self.log(f"Metadata batches: main={batch_size}, tagger={tagger_batch_size}, Florence={florence_batch_size}")
        self.log(f"Camie tag min confidence: {float(self.config.get('camie_threshold', 0.05)):.3f}")
        if include_florence:
            self.log(
                "Florence metadata mode: "
                f"skip_existing={skip_existing_captions}, per_folder={florence_per_folder or 'all'}, save_every={save_every or 'end'}"
            )
            self.log(f"Florence captions planned: {len(planned_florence)} / {total} images")
        for batch_start in range(0, total, batch_size):
            if self.is_cancelled():
                flush_metadata("cancelled")
                return {"ok": False, "message": "Metadata export cancelled."}
            batch_paths = img_paths[batch_start : batch_start + batch_size]
            batch_images: list[Image.Image] = []
            batch_valid_paths: list[Path] = []
            batch_rel_paths: list[str] = []
            for path in batch_paths:
                try:
                    batch_images.append(self._open_image_for_path(path))
                    batch_valid_paths.append(path)
                    batch_rel_paths.append(path.resolve().relative_to(source).as_posix())
                except Exception as exc:
                    errors += 1
                    self.log(f"Warning: cannot open {path.name}: {exc}")
                    self.log(traceback.format_exc())

            if batch_images:
                preds_batch: list[dict[str, float]] = []
                for image_chunk in self._chunks(batch_images, tagger_batch_size):
                    preds_batch.extend(self.manager.wd_tagger_infer_batch(image_chunk))
                captions: list[str] = [""] * len(batch_images)
                if include_florence:
                    try:
                        florence_images: list[Image.Image] = []
                        prompts: list[str] = []
                        florence_positions: list[int] = []
                        for pos, (image, preds, rel_path) in enumerate(zip(batch_images, preds_batch, batch_rel_paths)):
                            existing_item = detailed_metadata["images"].get(rel_path, {})
                            if (
                                skip_existing_captions
                                and isinstance(existing_item, dict)
                                and str(existing_item.get("caption_florence2", "")).strip()
                            ):
                                captions[pos] = str(existing_item.get("caption_florence2", ""))
                                captions_skipped += 1
                                continue
                            if rel_path not in planned_florence:
                                captions_skipped += 1
                                continue
                            resized = image.copy()
                            resized.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
                            florence_images.append(resized)
                            prompts.append(", ".join(self._metadata_tag_lists(preds, tags_per_img)[0][:10]))
                            florence_positions.append(pos)
                            folder_key = str(Path(rel_path).parent).replace("\\", "/")
                            if folder_key == ".":
                                folder_key = ""
                            folder_caption_counts[folder_key] = folder_caption_counts.get(folder_key, 0) + 1
                        if florence_images:
                            for start in range(0, len(florence_images), florence_batch_size):
                                generated = self.manager.florence_infer_batch(
                                    florence_images[start : start + florence_batch_size],
                                    prompts[start : start + florence_batch_size],
                                )
                                for pos, caption in zip(florence_positions[start : start + florence_batch_size], generated):
                                    captions[pos] = caption
                                    if caption:
                                        captions_generated += 1
                    except Exception as exc:
                        self.log(f"Warning: Florence metadata batch failed: {exc}")
                        self.log(traceback.format_exc())

                for rel_path, preds, caption in zip(batch_rel_paths, preds_batch, captions):
                    general_tags, character_tags, rating, all_scores = self._metadata_tag_lists(preds, tags_per_img)
                    tag_string = ", ".join(general_tags + character_tags)
                    flat_metadata[rel_path] = tag_string
                    existing_item = detailed_metadata["images"].get(rel_path, {})
                    item: dict[str, Any] = dict(existing_item) if isinstance(existing_item, dict) else {}
                    item.update({
                        "general_tags": general_tags,
                        "character_tags": character_tags,
                        "rating": rating,
                        "all_scores": all_scores,
                    })
                    if caption:
                        item["caption_florence2"] = caption
                    detailed_metadata["images"][rel_path] = item

            processed += len(batch_paths)
            if save_every > 0 and processed % save_every < len(batch_paths):
                flush_metadata(f"{processed}/{total}")
            self.prog(min(99.0, processed / max(total, 1) * 100.0))
            self._emit_eta(
                stage="Metadata export",
                completed=processed,
                total=total,
                start_ts=export_start,
                force=(processed >= total),
            )

        flush_metadata("final")
        self.prog(100)
        elapsed = time.time() - scan_start
        return {
            "ok": True,
            "message": (
                f"Metadata exported: {len(flat_metadata)} images, errors: {errors}, "
                f"Florence generated: {captions_generated}, skipped: {captions_skipped}, "
                f"time: {elapsed:.1f}s | {flat_path}"
            ),
            "metadata_file": str(flat_path),
            "detailed_metadata_file": str(detailed_path),
            "processed": len(flat_metadata),
            "errors": errors,
        }

    def _metadata_tag_lists(
        self,
        preds: dict[str, float],
        tags_per_img: int,
    ) -> tuple[list[str], list[str], dict[str, float], dict[str, float]]:
        general_tags: list[str] = []
        character_tags: list[str] = []
        rating: dict[str, float] = {}
        all_scores: dict[str, float] = {}
        content = sorted(preds.items(), key=lambda item: item[1], reverse=True)
        for tag, score in content:
            category = self.manager.wd_tag_categories.get(tag, -1)
            clean_tag = str(tag).strip()
            if category in (9, "rating", "meta", "object"):
                rating[clean_tag] = round(float(score), 4)
                continue
            if tag in {"no_humans", "text_focus"} or str(tag).startswith("year_"):
                continue
            all_scores[clean_tag] = round(float(score), 4)
            if category in (4, "character"):
                if len(character_tags) < tags_per_img and clean_tag not in character_tags:
                    character_tags.append(clean_tag)
            elif len(general_tags) < tags_per_img and clean_tag not in general_tags:
                general_tags.append(clean_tag)
        return general_tags[:tags_per_img], character_tags[:tags_per_img], rating, all_scores

    def _run_inner(self) -> dict[str, Any]:
        source = Path(self.config["source_dir"]).expanduser().resolve()
        target = Path(self.config["target_dir"]).expanduser().resolve()
        if not self.config.get("dry_run"):
            target.mkdir(parents=True, exist_ok=True)
        self._move_journal_target = target

        initial_load_config = dict(self.config)
        deferred_florence = bool(initial_load_config.get("use_florence"))
        if deferred_florence:
            initial_load_config["use_florence"] = False
            self.log("Deferring Florence-2 load until metadata stage.")

        if not self.manager.load_for_config(initial_load_config, self.progress_callback, self.log_callback):
            return {"ok": False, "message": "Model loading failed."}

        self._timings = {"start_total": time.time()}
        self.log("Scanning images...")
        scan_start = time.time()
        self.prog(0)
        img_paths = scan_images(
            source,
            recursive=bool(self.config.get("recursive_scan", True)),
            blocked_subfolders=self.config.get("blocked_subfolders", ""),
            cache_status_callback=self.scan_status_callback,
            log_callback=self.log_callback,
        )
        self._timings["scan"] = time.time() - scan_start

        total = len(img_paths)
        if not total:
            return {"ok": False, "message": "No images found."}
        requested_batch = max(1, int(self.config.get("batch_size_var", 8)))
        batch_size = self._batch_size("main", "batch_size_var", requested_batch)
        self.log(f"Found {total} images. Batch: {batch_size}")
        self._image_lmdb = prepare_image_lmdb(
            img_paths,
            source,
            self.config.get("lmdb_cache_dir") or resolve_lmdb_dir(None),
            mode=str(self.config.get("lmdb_mode", "auto")),
            rebuild=bool(self.config.get("rebuild_lmdb", False)),
            log_callback=self.log_callback,
        )
        if self._image_lmdb is not None:
            self._image_lmdb_dir = resolve_lmdb_dir(self.config.get("lmdb_cache_dir"))
            self._image_lmdb_source = source

        do_ai = bool(self.config.get("sort_ai_human"))
        do_sdxl = do_ai and bool(self.config.get("use_sdxl_vote"))
        do_cont = bool(self.config.get("sort_content"))
        do_style = bool(self.config.get("sort_style"))
        do_group = bool(self.config.get("sort_grouping"))
        do_dedup = bool(self.config.get("sort_dedup"))

        group_sens = max(0.01, min(0.99, float(self.config.get("group_threshold", 0.35))))
        dedup_thr = max(0.5, min(0.999, float(self.config.get("dedup_threshold", 0.88))))

        group_descs = [tag.strip() for tag in str(self.config.get("group_desc_var", "")).split(",") if tag.strip()]
        use_sem_group = do_group and bool(group_descs)
        use_dino_group = do_group and not group_descs
        need_dino_feats = do_dedup or use_dino_group

        cont_pos, cont_neg = parse_tags(self.config.get("content_tags", ""), self.log_callback) if do_cont else ([], [])
        sty_pos, sty_neg = parse_tags(self.config.get("style_tags", ""), self.log_callback) if do_style else ([], [])

        uncertainty_threshold = max(0.5, min(0.99, float(self.config.get("uncertainty_threshold", 0.7))))
        route_uncertain = bool(self.config.get("route_uncertain", True))

        with torch.inference_mode():
            self.log("Initializing models and feature cache...")
            init_start = time.time()
            cont_min_conf = float(self.config.get("content_min_conf", 0.05))
            sty_min_conf = float(self.config.get("style_min_conf", 0.05))

            if do_cont:
                cont_pos_embeds = self.manager.get_tag_embeddings(cont_pos)
                cont_neg_embeds = self.manager.get_tag_embeddings(cont_neg)
                cont_anc_embed = self.manager.get_tag_embeddings(
                    [self.config.get("content_neg_anchor", CONTENT_NEG_ANCHOR)],
                    is_anchor=True,
                )
            else:
                cont_pos_embeds = cont_neg_embeds = cont_anc_embed = torch.empty(0, device=self.manager.device)

            if do_style:
                sty_pos_embeds = self.manager.get_tag_embeddings(sty_pos)
                sty_neg_embeds = self.manager.get_tag_embeddings(sty_neg)
                sty_anc_embed = self.manager.get_tag_embeddings(
                    [self.config.get("style_neg_anchor", STYLE_NEG_ANCHOR)],
                    is_anchor=True,
                )
            else:
                sty_pos_embeds = sty_neg_embeds = sty_anc_embed = torch.empty(0, device=self.manager.device)

            if use_sem_group:
                group_descs_embeds = self.manager.get_tag_embeddings(group_descs)
                group_anc_embed = self.manager.get_tag_embeddings(
                    ["none of the above, other, miscellaneous"],
                    is_anchor=True,
                )
                all_group_embeds = torch.cat([group_descs_embeds, group_anc_embed], dim=0)
            else:
                all_group_embeds = None

            self._timings["init"] = time.time() - init_start

            feat_cache_file = self.paths.feat_cache_file
            try:
                feat_cache = (
                    torch.load(str(feat_cache_file), weights_only=True)
                    if feat_cache_file.exists()
                    else {}
                )
            except Exception as exc:
                self.log(f"Warning: failed to load feature cache: {exc}")
                feat_cache = {}
            cache_dirty = False

        dataset = ImagePathDataset(img_paths, lmdb_dir=self._image_lmdb_dir, lmdb_source=self._image_lmdb_source)
        cpu_count = os.cpu_count() or 1
        num_workers = 0 if sys.platform == "win32" else min(4, max(cpu_count - 1, 1))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.manager.device == "cuda"),
            collate_fn=pil_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

        accepted: list[tuple[Path, str]] = []
        dino_paths: list[Path] = []
        dino_feats: list[torch.Tensor] = []
        dino_subs: list[str] = []
        wd_preds_cache: dict[Path, dict[str, float]] = {}
        errors = skipped = processed = 0
        proc_end = 80

        need_tagger = bool(
            (self.config.get("sort_grouping") and not str(self.config.get("group_desc_var", "")).strip())
            or self.config.get("gen_metadata")
            or self.config.get("sort_tagger_filter")
            or (self.config.get("character_aware_recursive") and self.config.get("recursive_scan"))
        )
        can_stream = (
            not need_dino_feats
            and not need_tagger
            and not self.config.get("gen_metadata")
            and not self.config.get("group_sequences")
        )
        do_tagger_filter = bool(self.config.get("sort_tagger_filter"))
        tagger_pos, tagger_neg = parse_tags(self.config.get("tagger_filter_tags", ""), self.log_callback) if do_tagger_filter else ([], [])
        tagger_threshold = float(self.config.get("camie_threshold", 0.05))
        tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", batch_size)

        process_start = time.time()
        with torch.inference_mode():
            for batch_records in loader:
                if self.is_cancelled():
                    return {"ok": False, "message": "Cancelled."}

                valid = [(record.path, record.image) for record in batch_records if record.image is not None]
                bad_records = [record for record in batch_records if record.image is None]
                if bad_records:
                    errors += len(bad_records)
                    for record in bad_records:
                        if record.error:
                            self.log(f"Warning: cannot open {record.path.name}: {record.error}")
                        else:
                            self.log(f"Warning: cannot open {record.path.name}")
                if not valid:
                    processed += len(batch_records)
                    self.prog(processed / total * proc_end)
                    continue

                paths_v = [item[0] for item in valid]
                imgs_v = [item[1] for item in valid]
                count = len(imgs_v)
                ok = [True] * count
                subs = [""] * count

                if self.is_cancelled():
                    return {"ok": False, "message": "Cancelled."}
                if do_ai:
                    self._run_ai_human_batch(
                        imgs_v,
                        paths_v,
                        subs,
                        route_uncertain=route_uncertain,
                        uncertainty_threshold=uncertainty_threshold,
                    )

                siglip_embs: list[torch.Tensor | None] = [None] * count
                if do_cont or do_style or use_sem_group or use_dino_group:
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    missing_siglip_indices: list[int] = []
                    for index in range(count):
                        if not ok[index]:
                            continue
                        try:
                            mtime = str(paths_v[index].stat().st_mtime)
                        except OSError:
                            mtime = "0"
                        cache_key = f"siglip_{paths_v[index]}_{mtime}"
                        if cache_key in feat_cache:
                            siglip_embs[index] = feat_cache[cache_key].to(self.manager.device, non_blocking=True)
                        else:
                            missing_siglip_indices.append(index)

                    if missing_siglip_indices:
                        try:
                            missing_imgs = [imgs_v[index] for index in missing_siglip_indices]
                            batch_embs = self.manager.siglip_embed_images_batch(missing_imgs)
                            for cache_index, embedding in zip(missing_siglip_indices, batch_embs):
                                siglip_embs[cache_index] = embedding.unsqueeze(0)
                                try:
                                    mtime = str(paths_v[cache_index].stat().st_mtime)
                                except OSError:
                                    mtime = "0"
                                feat_cache[f"siglip_{paths_v[cache_index]}_{mtime}"] = siglip_embs[cache_index].cpu()
                                cache_dirty = True
                        except torch.cuda.OutOfMemoryError:
                            self.log("Warning: OOM during SigLIP batch embedding - retrying one by one.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            for cache_index in missing_siglip_indices:
                                try:
                                    embedding = self.manager.siglip_embed_images_batch([imgs_v[cache_index]])[0]
                                    siglip_embs[cache_index] = embedding.unsqueeze(0)
                                    try:
                                        mtime = str(paths_v[cache_index].stat().st_mtime)
                                    except OSError:
                                        mtime = "0"
                                    feat_cache[f"siglip_{paths_v[cache_index]}_{mtime}"] = siglip_embs[cache_index].cpu()
                                    cache_dirty = True
                                except Exception as exc:
                                    ok[cache_index] = False
                                    self.log(f"Warning: SigLIP retry failed for {paths_v[cache_index].name}: {exc}")
                                    self.log(traceback.format_exc())
                        except Exception as exc:
                            self.log(f"Warning: SigLIP batch embedding error: {exc}")
                            self.log(traceback.format_exc())
                            for cache_index in missing_siglip_indices:
                                ok[cache_index] = False

                if do_cont and (cont_pos or cont_neg):
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    alive = [index for index in range(count) if ok[index] and siglip_embs[index] is not None]
                    if alive:
                        try:
                            results = self.manager.siglip_filter(
                                [siglip_embs[index] for index in alive if siglip_embs[index] is not None],
                                cont_pos,
                                cont_pos_embeds,
                                cont_neg,
                                cont_neg_embeds,
                                cont_anc_embed,
                                cont_min_conf,
                            )
                            for result_index, item_index in enumerate(alive):
                                passed, tag, _margin = results[result_index]
                                if not passed:
                                    ok[item_index] = False
                                    skipped += 1
                                else:
                                    subs[item_index] = str(Path(subs[item_index]) / tag) if subs[item_index] else tag
                        except Exception as exc:
                            self.log(f"Warning: content filter error: {exc}")
                            self.log(traceback.format_exc())

                if do_style and (sty_pos or sty_neg):
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    alive = [index for index in range(count) if ok[index] and siglip_embs[index] is not None]
                    if alive:
                        try:
                            results = self.manager.siglip_filter(
                                [siglip_embs[index] for index in alive if siglip_embs[index] is not None],
                                sty_pos,
                                sty_pos_embeds,
                                sty_neg,
                                sty_neg_embeds,
                                sty_anc_embed,
                                sty_min_conf,
                            )
                            for result_index, item_index in enumerate(alive):
                                passed, tag, _margin = results[result_index]
                                if not passed:
                                    ok[item_index] = False
                                    skipped += 1
                                else:
                                    subs[item_index] = str(Path(subs[item_index]) / tag) if subs[item_index] else tag
                        except Exception as exc:
                            self.log(f"Warning: style filter error: {exc}")
                            self.log(traceback.format_exc())

                if do_tagger_filter and self.manager.wd_tagger is not None and (tagger_pos or tagger_neg):
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    alive = [index for index in range(count) if ok[index]]
                    if alive:
                        missing_wd = [index for index in alive if paths_v[index] not in wd_preds_cache]
                        if missing_wd:
                            try:
                                for chunk in self._chunks(missing_wd, tagger_batch_size):
                                    batch_preds = self.manager.wd_tagger_infer_batch([imgs_v[index] for index in chunk])
                                    for item_index, preds in zip(chunk, batch_preds):
                                        if preds:
                                            wd_preds_cache[paths_v[item_index]] = preds
                            except Exception as exc:
                                self.log(f"Warning: tagger batch inference error: {exc}")
                                self.log(traceback.format_exc())

                        for item_index in alive:
                            preds = wd_preds_cache.get(paths_v[item_index], {})
                            passed, _fail_tag = self.manager.tagger_filter(
                                preds,
                                tagger_pos,
                                tagger_neg,
                                tagger_threshold,
                                image_name=paths_v[item_index],
                            )
                            if not passed:
                                ok[item_index] = False
                                skipped += 1

                if self.config.get("character_aware_recursive") and self.manager.wd_tagger is not None:
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    alive = [index for index in range(count) if ok[index]]
                    missing_wd = [index for index in alive if paths_v[index] not in wd_preds_cache]
                    if missing_wd:
                        try:
                            for chunk in self._chunks(missing_wd, tagger_batch_size):
                                batch_preds = self.manager.wd_tagger_infer_batch([imgs_v[index] for index in chunk])
                                for item_index, preds in zip(chunk, batch_preds):
                                    if preds:
                                        wd_preds_cache[paths_v[item_index]] = preds
                        except Exception as exc:
                            self.log(f"Warning: character-aware tagger error: {exc}")
                            self.log(traceback.format_exc())
                    for item_index in alive:
                        subs[item_index] = self._prepend_character_folder(
                            paths_v[item_index],
                            subs[item_index],
                            wd_preds_cache.get(paths_v[item_index], {}),
                        )

                if use_sem_group and all_group_embeds is not None:
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    alive = [index for index in range(count) if ok[index] and siglip_embs[index] is not None]
                    for item_index in alive:
                        try:
                            best_index, _score = self.manager.siglip_classify(siglip_embs[item_index], all_group_embeds)
                            if best_index != -1:
                                group_name = safe_filename(group_descs[best_index])
                                subs[item_index] = str(Path(subs[item_index]) / group_name) if subs[item_index] else group_name
                        except Exception as exc:
                            self.log(f"Warning: semantic group error for {paths_v[item_index].name}: {exc}")
                            self.log(traceback.format_exc())

                if need_dino_feats:
                    if self.is_cancelled():
                        return {"ok": False, "message": "Cancelled."}
                    cache_dirty = self._collect_dino_features(
                        paths_v,
                        imgs_v,
                        ok,
                        subs,
                        siglip_embs,
                        feat_cache,
                        wd_preds_cache,
                        dino_paths,
                        dino_feats,
                        dino_subs,
                        need_tagger=need_tagger,
                        use_dino_group=use_dino_group,
                    ) or cache_dirty
                else:
                    for item_index in range(count):
                        if ok[item_index]:
                            if can_stream:
                                sanitized_sub = self._sanitize_subfolder_path(subs[item_index])
                                destination = target / sanitized_sub if sanitized_sub else target
                                if not self._place(paths_v[item_index], destination):
                                    errors += 1
                                    continue
                            accepted.append((paths_v[item_index], subs[item_index]))

                processed += len(batch_records)
                self.prog(processed / total * proc_end)
                self._emit_eta(
                    stage="Main pipeline",
                    completed=processed,
                    total=total,
                    start_ts=process_start,
                    force=(processed >= total),
                )
                if self._is_soft_stop_requested():
                    self._mark_soft_stop("Stop-after-batch requested. Finalizing processed images...")
                    break

        return self._finalize_results(
            accepted=accepted,
            dino_paths=dino_paths,
            dino_feats=dino_feats,
            dino_subs=dino_subs,
            wd_preds_cache=wd_preds_cache,
            total=total,
            processed=processed,
            skipped=skipped,
            errors=errors,
            target=target,
            group_sens=group_sens,
            dedup_thr=dedup_thr,
            need_tagger=need_tagger,
            use_dino_group=use_dino_group,
            cache_dirty=cache_dirty,
            feat_cache=feat_cache,
            can_stream=can_stream,
        )

    def _run_ai_human_batch(
        self,
        imgs_v: list[Any],
        paths_v: list[Path],
        subs: list[str],
        *,
        route_uncertain: bool,
        uncertainty_threshold: float,
    ) -> None:
        try:
            inputs = self.manager.ai_proc(images=imgs_v, return_tensors="pt")
            inputs = {
                key: value.to(self.manager.device, non_blocking=True)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.amp.autocast(
                device_type=self.manager.device,
                dtype=self.manager.amp_dtype,
                enabled=(self.manager.use_amp and self.manager.device != "cpu"),
            ):
                logits = self.manager.ai_model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)
            indices = logits.argmax(-1)
            probs_np = probs.cpu().numpy()
            indices_np = indices.cpu().numpy()

            if self.config.get("use_sdxl_vote") and self.manager.sdxl_model is not None:
                sdxl_inputs = self.manager.sdxl_proc(images=imgs_v, return_tensors="pt")
                sdxl_inputs = {
                    key: value.to(self.manager.device, non_blocking=True)
                    for key, value in sdxl_inputs.items()
                    if isinstance(value, torch.Tensor)
                }
                with torch.amp.autocast(
                    device_type=self.manager.device,
                    dtype=self.manager.amp_dtype,
                    enabled=(self.manager.use_amp and self.manager.device != "cpu"),
                ):
                    sdxl_logits = self.manager.sdxl_model(**sdxl_inputs).logits
                sdxl_probs = torch.softmax(sdxl_logits.float(), dim=-1)
                sdxl_indices = sdxl_logits.argmax(-1)
                sdxl_probs_np = sdxl_probs.cpu().numpy()
                sdxl_indices_np = sdxl_indices.cpu().numpy()
                for index in range(len(imgs_v)):
                    label_1 = str(self.manager.ai_model.config.id2label[indices_np[index]]).lower()
                    label_2_raw = str(self.manager.sdxl_model.config.id2label[sdxl_indices_np[index]]).lower()
                    label_2 = "ai" if label_2_raw in {"artificial", "ai"} else "hum"
                    conf_1 = float(probs_np[index][indices_np[index]])
                    conf_2 = float(sdxl_probs_np[index][sdxl_indices_np[index]])
                    disagree = label_1 != label_2
                    low_conf = conf_1 < uncertainty_threshold or conf_2 < uncertainty_threshold
                    if route_uncertain and (disagree or low_conf):
                        subs[index] = "Needs_Review/AI_Confidence"
                    else:
                        if disagree:
                            subs[index] = label_1
                        else:
                            subs[index] = label_1
            else:
                for index in range(len(imgs_v)):
                    label = str(self.manager.ai_model.config.id2label[indices_np[index]]).lower()
                    confidence = float(probs_np[index][indices_np[index]])
                    if route_uncertain and confidence < uncertainty_threshold:
                        subs[index] = "Needs_Review/AI_Confidence"
                    else:
                        subs[index] = label
        except torch.cuda.OutOfMemoryError:
            self.log("Warning: OOM during AI/Human batch inference - retrying one by one.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for index in range(len(imgs_v)):
                try:
                    inputs = self.manager.ai_proc(images=imgs_v[index], return_tensors="pt")
                    inputs = {
                        key: value.to(self.manager.device, non_blocking=True)
                        for key, value in inputs.items()
                        if isinstance(value, torch.Tensor)
                    }
                    with torch.amp.autocast(
                        device_type=self.manager.device,
                        dtype=self.manager.amp_dtype,
                        enabled=(self.manager.use_amp and self.manager.device != "cpu"),
                    ):
                        logits = self.manager.ai_model(**inputs).logits
                    subs[index] = self.manager.ai_model.config.id2label[logits.argmax(-1).item()]
                except Exception as exc:
                    self.log(f"Warning: {paths_v[index].name}: {exc}")
                    self.log(traceback.format_exc())
        except Exception as exc:
            self.log(f"Warning: AI/Human error: {exc}")
            self.log(traceback.format_exc())

    def _collect_dino_features(
        self,
        paths_v: list[Path],
        imgs_v: list[Any],
        ok: list[bool],
        subs: list[str],
        siglip_embs: list[torch.Tensor | None],
        feat_cache: dict[str, torch.Tensor],
        wd_preds_cache: dict[Path, dict[str, float]],
        dino_paths: list[Path],
        dino_feats: list[torch.Tensor],
        dino_subs: list[str],
        *,
        need_tagger: bool,
        use_dino_group: bool,
    ) -> bool:
        if self.is_cancelled():
            return False
        cache_dirty = False
        count = len(imgs_v)
        missing_dino_indices: list[int] = []
        dino_vecs: list[torch.Tensor | None] = [None] * count
        for index in range(count):
            if not ok[index]:
                continue
            try:
                mtime = str(paths_v[index].stat().st_mtime)
            except OSError:
                mtime = "0"
            cache_key = f"dinov2ms_{paths_v[index]}_{mtime}_{bool(self.config.get('use_dino_adapter', True))}"
            if cache_key in feat_cache:
                dino_vecs[index] = feat_cache[cache_key]
            else:
                missing_dino_indices.append(index)

        if missing_dino_indices:
            if self.is_cancelled():
                return cache_dirty
            dino_batch_size = self._batch_size("dino", "dino_batch_size", len(missing_dino_indices))
            try:
                for chunk in self._chunks(missing_dino_indices, dino_batch_size):
                    dino_inputs = self.manager.dino_proc(images=[imgs_v[index] for index in chunk], return_tensors="pt")
                    dino_inputs = {
                        key: value.to(self.manager.device, non_blocking=True)
                        for key, value in dino_inputs.items()
                        if isinstance(value, torch.Tensor)
                    }
                    with torch.amp.autocast(
                        device_type=self.manager.device,
                        dtype=self.manager.amp_dtype,
                        enabled=(self.manager.use_amp and self.manager.device != "cpu"),
                    ):
                        hidden_state = self.manager.dino_model(**dino_inputs).last_hidden_state
                        cls_tok = hidden_state[:, 0, :]
                        spatial = hidden_state[:, 1:, :].mean(dim=1)
                        batch_feat = torch.cat([cls_tok, spatial], dim=-1)
                        if self.manager.dino_adapter is not None:
                            batch_feat = self.manager.dino_adapter(batch_feat)
                    batch_feat_cpu = batch_feat.float().cpu()
                    for batch_index, item_index in enumerate(chunk):
                        value = batch_feat_cpu[batch_index].unsqueeze(0)
                        dino_vecs[item_index] = value
                        try:
                            mtime = str(paths_v[item_index].stat().st_mtime)
                        except OSError:
                            mtime = "0"
                        cache_key = f"dinov2ms_{paths_v[item_index]}_{mtime}_{bool(self.config.get('use_dino_adapter', True))}"
                        feat_cache[cache_key] = value
                        cache_dirty = True
            except torch.cuda.OutOfMemoryError:
                self.log("Warning: OOM during DINOv2 batch feature extraction - retrying one by one.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for item_index in missing_dino_indices:
                    if self.is_cancelled():
                        return cache_dirty
                    try:
                        dino_inputs = self.manager.dino_proc(images=[imgs_v[item_index]], return_tensors="pt")
                        dino_inputs = {
                            key: value.to(self.manager.device, non_blocking=True)
                            for key, value in dino_inputs.items()
                            if isinstance(value, torch.Tensor)
                        }
                        with torch.amp.autocast(
                            device_type=self.manager.device,
                            dtype=self.manager.amp_dtype,
                            enabled=(self.manager.use_amp and self.manager.device != "cpu"),
                        ):
                            hidden_state = self.manager.dino_model(**dino_inputs).last_hidden_state
                            cls_tok = hidden_state[:, 0, :]
                            spatial = hidden_state[:, 1:, :].mean(dim=1)
                            item_feat = torch.cat([cls_tok, spatial], dim=-1)
                            if self.manager.dino_adapter is not None:
                                item_feat = self.manager.dino_adapter(item_feat)
                        value = item_feat.float().cpu()
                        dino_vecs[item_index] = value
                        try:
                            mtime = str(paths_v[item_index].stat().st_mtime)
                        except OSError:
                            mtime = "0"
                        cache_key = f"dinov2ms_{paths_v[item_index]}_{mtime}_{bool(self.config.get('use_dino_adapter', True))}"
                        feat_cache[cache_key] = value
                        cache_dirty = True
                    except Exception as exc:
                        self.log(f"Warning: DINOv2 retry failed for {paths_v[item_index].name}: {exc}")
                        self.log(traceback.format_exc())
                        ok[item_index] = False
            except Exception as exc:
                self.log(f"Warning: DINOv2 batch feature error: {exc}")
                self.log(traceback.format_exc())
                for item_index in missing_dino_indices:
                    ok[item_index] = False

        alive_tagger = [index for index in range(count) if ok[index]]
        if need_tagger and self.manager.wd_tagger is not None and alive_tagger:
            if self.is_cancelled():
                return cache_dirty
            missing_wd = [index for index in alive_tagger if paths_v[index] not in wd_preds_cache]
            if missing_wd:
                try:
                    tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", len(missing_wd))
                    for chunk in self._chunks(missing_wd, tagger_batch_size):
                        batch_preds = self.manager.wd_tagger_infer_batch([imgs_v[index] for index in chunk])
                        for item_index, preds in zip(chunk, batch_preds):
                            if preds:
                                wd_preds_cache[paths_v[item_index]] = preds
                except Exception as exc:
                    self.log(f"Warning: tagger batch inference error: {exc}")
                    self.log(traceback.format_exc())

        for index in range(count):
            if self.is_cancelled():
                return cache_dirty
            if not ok[index] or dino_vecs[index] is None:
                continue
            try:
                dino_vec = dino_vecs[index]
                sem_vec = None
                if need_tagger and self.manager.wd_tagger is not None:
                    preds = wd_preds_cache.get(paths_v[index])
                    if preds:
                        sem_vec = self._build_semantic_vector(preds)

                if use_dino_group:
                    sig_vec = siglip_embs[index].cpu() if siglip_embs[index] is not None else torch.zeros_like(dino_vec)
                    dino_n = F.normalize(dino_vec, p=2, dim=-1) * 1.0
                    sig_n = F.normalize(sig_vec, p=2, dim=-1) * 0.7
                    parts = [dino_n, sig_n]
                    if sem_vec is not None:
                        sem_n = F.normalize(sem_vec, p=2, dim=-1) * 0.3
                        parts.append(sem_n)
                    combined = F.normalize(torch.cat(parts, dim=-1), p=2, dim=-1)
                    dino_feats.append(combined)
                else:
                    dino_feats.append(dino_vec)

                dino_paths.append(paths_v[index])
                dino_subs.append(subs[index])
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.log(f"Warning: OOM on {paths_v[index].name}, skipping")
                ok[index] = False
            except Exception as exc:
                self.log(f"Warning: feature extract {paths_v[index].name}: {exc}")
                self.log(traceback.format_exc())
                ok[index] = False

        return cache_dirty

    def _build_semantic_vector(self, preds: dict[str, float]) -> torch.Tensor | None:
        try:
            def get_max_score(tag_set: set[str]) -> float:
                return max([score for tag, score in preds.items() if tag in tag_set] + [0.0])

            nude_score = get_max_score(WD_NUDE_TAGS)
            state_vec = torch.tensor(
                [
                    nude_score,
                    max(0.0, 1.0 - nude_score),
                    get_max_score(WD_P_STAND_TAGS),
                    get_max_score(WD_P_SIT_TAGS),
                    get_max_score(WD_P_LIE_TAGS),
                    get_max_score(WD_C_SHIRT_TAGS),
                    get_max_score(WD_C_JACKET_TAGS),
                    get_max_score(WD_C_DRESS_TAGS),
                    get_max_score(WD_C_SKIRT_TAGS),
                    get_max_score(WD_C_PANTS_TAGS),
                    get_max_score(WD_SC_INDOOR),
                    get_max_score(WD_SC_OUTDOOR),
                    get_max_score(WD_SC_NATURE),
                    get_max_score(WD_SC_URBAN),
                    get_max_score(WD_H_BLACK),
                    get_max_score(WD_H_BLONDE),
                    get_max_score(WD_H_BROWN),
                    get_max_score(WD_H_RED),
                    get_max_score(WD_H_BLUE),
                    get_max_score(WD_H_WHITE),
                    get_max_score(WD_H_GREEN),
                    get_max_score(WD_H_PURPLE),
                    get_max_score(WD_EX_HAPPY),
                    get_max_score(WD_EX_SAD),
                    get_max_score(WD_EX_ANGRY),
                    get_max_score(WD_EX_NEUTRAL),
                ],
                dtype=torch.float32,
                device="cpu",
            )

            char_preds = [
                (tag, score)
                for tag, score in preds.items()
                if self.manager.wd_tag_categories.get(tag, -1) in (4, "character")
            ]
            top_chars = sorted(char_preds, key=lambda item: item[1], reverse=True)[:8]
            char_dim = 38
            char_vec = torch.zeros(char_dim, device="cpu")
            for tag, score in top_chars:
                if score > 0.35:
                    index = hash(tag) % char_dim
                    char_vec[index] = max(char_vec[index].item(), score)
            return torch.cat([state_vec, char_vec], dim=-1).unsqueeze(0)
        except Exception as exc:
            self.log(f"Warning: failed to build semantic vector: {exc}")
            self.log(traceback.format_exc())
            return None

    def _finalize_results(
        self,
        *,
        accepted: list[tuple[Path, str]],
        dino_paths: list[Path],
        dino_feats: list[torch.Tensor],
        dino_subs: list[str],
        wd_preds_cache: dict[Path, dict[str, float]],
        total: int,
        processed: int,
        skipped: int,
        errors: int,
        target: Path,
        group_sens: float,
        dedup_thr: float,
        need_tagger: bool,
        use_dino_group: bool,
        cache_dirty: bool,
        feat_cache: dict[str, torch.Tensor],
        can_stream: bool,
    ) -> dict[str, Any]:
        if self.is_cancelled():
            return {"ok": False, "message": "Cancelled."}
        if self.config.get("group_sequences") and not self.is_cancelled():
            self._group_sequences(accepted, dino_paths, dino_feats, dino_subs)

        if self.config.get("sort_dedup") and dino_feats and not self.is_cancelled():
            before = len(dino_feats)
            self.log(f"Deduplicating {before} images...")
            dino_paths, dino_feats, dino_subs = self._deduplicate(dino_paths, dino_feats, dino_subs, dedup_thr)
            self.log(f"  Removed {before - len(dino_feats)} duplicates")
            self.prog(85)

        clustered_groups: dict[int, list[tuple[Path, str]]] = {}
        if use_dino_group and dino_feats and not self.is_cancelled():
            self.log(f"Clustering {len(dino_feats)} images...")
            try:
                if self.config.get("character_aware_recursive"):
                    clustered_groups = {}
                    buckets: dict[str, list[int]] = {}
                    for index, sub in enumerate(dino_subs):
                        parts = Path(self._sanitize_subfolder_path(sub)).parts if sub else ()
                        character_key = parts[0] if parts else ""
                        buckets.setdefault(character_key, []).append(index)
                    next_group_id = 0
                    for character_key, indices in buckets.items():
                        if not indices:
                            continue
                        self.log(f"  Character bucket '{character_key or 'root'}': {len(indices)} images")
                        bucket_groups = self._cluster_images(
                            [dino_paths[index] for index in indices],
                            [dino_feats[index] for index in indices],
                            [dino_subs[index] for index in indices],
                            group_sens,
                            max_clusters=int(self.config.get("max_folders_created", 0) or 0),
                        )
                        for _bucket_id, members in bucket_groups.items():
                            clustered_groups[next_group_id] = members
                            next_group_id += 1
                else:
                    clustered_groups = self._cluster_images(
                        dino_paths,
                        dino_feats,
                        dino_subs,
                        group_sens,
                        max_clusters=int(self.config.get("max_folders_created", 0) or 0),
                    )
            except Exception as exc:
                self.log(f"Warning: clustering error: {exc}")
                self.log(traceback.format_exc())
                accepted.extend(list(zip(dino_paths, dino_subs)))
        elif dino_feats:
            accepted.extend(list(zip(dino_paths, dino_subs)))
        self.prog(85)

        if self.is_cancelled():
            return {"ok": False, "message": "Cancelled."}

        uncat_indices = [index for index, (_path, sub) in enumerate(accepted) if not sub]
        if need_tagger and uncat_indices and not self.is_cancelled():
            self.log(f"Categorizing {len(uncat_indices)} 'Other' images using cached tags...")
            generic_tags = {
                "monochrome",
                "greyscale",
                "no_humans",
                "negative_space",
                "simple_background",
                "white_background",
                "black_background",
                "transparent_background",
                "sketch",
                "lineart",
                "comic",
                "manga",
                "gradient_background",
                "pattern_background",
            }
            for item_index in uncat_indices:
                path, _sub = accepted[item_index]
                preds = wd_preds_cache.get(path)
                if preds:
                    filtered_tags = []
                    for tag, score in preds.items():
                        if self._contains_year_token(tag):
                            continue
                        if self.manager.wd_tag_categories.get(tag, -1) not in (9, "rating") and score > 0.35:
                            adjusted = score * (0.1 if tag in generic_tags else 1.0)
                            filtered_tags.append((tag, adjusted))
                    top_tags = sorted(filtered_tags, key=lambda item: item[1], reverse=True)[:2]
                    folder_name = (
                        safe_filename(
                            self._strip_year_tokens(" ".join(tag.replace("_", " ").title() for tag, _ in top_tags))
                        )[:40]
                        if top_tags
                        else "Other"
                    )
                else:
                    folder_name = "Other"
                accepted[item_index] = (path, folder_name)
                self.log(f"  {path.name} -> {folder_name}")

        if need_tagger and clustered_groups and not self.is_cancelled():
            self.log(f"Naming {len(clustered_groups)} clusters from tags...")
            accepted.extend(self._wd_tagger_name_clusters(clustered_groups, wd_preds_cache))
            self.prog(92)

        accepted = self._apply_folder_limit(accepted)

        do_meta = (
            need_tagger
            and self.config.get("gen_metadata")
            and not self.is_cancelled()
            and not self._stop_after_batch_triggered
            and not self.config.get("dry_run")
        )
        if self.config.get("dry_run") and self.config.get("gen_metadata"):
            self.log("Dry run: metadata file writing skipped.")
        if do_meta:
            if self.config.get("use_florence"):
                self.log("Loading Florence-2 on demand for metadata...")
                if self.manager.device == "cuda" and bool(self.config.get("unload_inactive_models", True)):
                    self.manager.offload_non_florence_to_cpu()
                if not self.manager.load_for_config(self.config, self.progress_callback, self.log_callback):
                    return {"ok": False, "message": "Florence-2 failed to load for metadata stage."}
            self.log(f"Generating rich metadata for {len(accepted)} images...")
            self._wd_tagger_generate_metadata(accepted, target, wd_preds_cache)

        if need_tagger:
            self.manager.unload_tagger()
            wd_preds_cache.clear()

        if cache_dirty and not self.is_cancelled():
            try:
                self.log("Saving feature cache...")
                feat_cache_path = self.paths.cache_dir / "feat_cache.pt"
                feat_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat_cache, str(feat_cache_path))
            except Exception as exc:
                self.log(f"Warning: could not save cache: {exc}")
                self.log(traceback.format_exc())

        self.prog(95)
        if self.is_cancelled():
            return {"ok": False, "message": "Cancelled."}

        placement_errors = 0
        if not can_stream:
            self.log(f"Saving {len(accepted)} images...")
            for index, (path, sub) in enumerate(accepted):
                sanitized_sub = self._sanitize_subfolder_path(sub)
                destination = target / sanitized_sub if sanitized_sub else target
                if not self._place(path, destination):
                    placement_errors += 1
                self.prog(95 + (index + 1) / max(len(accepted), 1) * 5)
        else:
            self.log("Files were streamed successfully.")

        uncertain_saved = sum(1 for _path, sub in accepted if str(sub).startswith("Needs_Review"))
        if uncertain_saved:
            self.log(f"Uncertain queue: moved {uncertain_saved} files to Needs_Review")

        total_time = time.time() - self._timings.get("start_total", time.time())
        self._timings["total"] = total_time
        self._save_timing_summary(processed)
        self.manager.cleanup()
        done_message = (
            f"Stopped after current batch in {total_time:.1f}s. "
            if self._stop_after_batch_triggered
            else f"Done in {total_time:.1f}s! "
        )
        action_label = "Planned" if self.config.get("dry_run") else "Saved"
        if self.config.get("dry_run"):
            done_message = f"Dry run done in {total_time:.1f}s. "
        return {
            "ok": True,
            "message": f"{done_message}Total: {total} | Processed: {processed} | {action_label}: {len(accepted) - placement_errors} | Needs review: {uncertain_saved} | Filtered: {skipped} | Errors: {errors + placement_errors}",
            "processed": processed,
            "saved": len(accepted) - placement_errors,
            "uncertain_saved": uncertain_saved,
            "filtered": skipped,
            "errors": errors + placement_errors,
            "elapsed": total_time,
            "stopped_after_batch": self._stop_after_batch_triggered,
        }

    def _group_sequences(
        self,
        accepted: list[tuple[Path, str]],
        dino_paths: list[Path],
        dino_feats: list[torch.Tensor],
        dino_subs: list[str],
    ) -> None:
        self.log("Grouping sequential images (Manga mode with heuristics)...")

        def get_clean_base(name: str) -> str:
            base = Path(name).stem
            return re.sub(r"[\s_\-]*\d+$", "", base).lower()

        def get_prefix(name: str, length: int = 6) -> str:
            return Path(name).stem[:length].lower()

        def natural_sort_key(path: Path) -> tuple[str, list[Any]]:
            parts = re.split(r"(\d+)", path.name.lower())
            normalized: list[Any] = []
            for part in parts:
                normalized.append(int(part) if part.isdigit() else part)
            return (str(path.parent), normalized)

        if dino_feats:
            cpu_feats = [value.cpu().float() for value in dino_feats]
            sorted_idx = list(range(len(dino_paths)))
            sorted_idx.sort(key=lambda item: natural_sort_key(dino_paths[item]))

            dir_seq_index: dict[str, list[int]] = {}
            sequences: list[dict[str, Any]] = []

            with torch.inference_mode():
                for sorted_pos, item_index in enumerate(sorted_idx):
                    if sorted_pos % 50 == 0:
                        self.prog(80 + (sorted_pos / max(1, len(sorted_idx))) * 5)

                    path = dino_paths[item_index]
                    dirname = str(path.parent)
                    filename = path.name
                    try:
                        mtime = path.stat().st_mtime
                    except OSError:
                        mtime = 0

                    base = get_clean_base(filename)
                    prefix = get_prefix(filename)
                    feat = cpu_feats[item_index]
                    best_seq = None
                    best_score = -1.0

                    for seq_idx in dir_seq_index.get(dirname, [])[-20:]:
                        seq = sequences[seq_idx]
                        score = 0.0
                        if abs(mtime - seq["first_mtime"]) <= 40:
                            score += 0.6
                        if base and base == seq["base"]:
                            score += 0.4
                        if prefix and prefix == seq["prefix"]:
                            score += 0.3
                        sim = F.cosine_similarity(feat, seq["last_feat"], dim=-1).item()
                        if sim > 0.85:
                            score += 1.0
                        if score >= 0.7 and score > best_score:
                            best_score = score
                            best_seq = seq

                    if best_seq is not None:
                        best_seq["indices"].append(item_index)
                        best_seq["last_feat"] = feat
                    else:
                        seq_list_idx = len(sequences)
                        sequences.append(
                            {
                                "dir": dirname,
                                "indices": [item_index],
                                "first_mtime": mtime,
                                "base": base,
                                "prefix": prefix,
                                "last_feat": feat,
                            }
                        )
                        dir_seq_index.setdefault(dirname, []).append(seq_list_idx)

            unified_count = 0
            for seq in sequences:
                indices = seq["indices"]
                if len(indices) > 1:
                    unified_count += len(indices)
                    avg_feat = torch.stack([cpu_feats[index] for index in indices]).mean(dim=0)
                    avg_feat = F.normalize(avg_feat.unsqueeze(0), p=2, dim=-1).squeeze(0)
                    sub_counts: dict[str, int] = {}
                    for index in indices:
                        sub = dino_subs[index]
                        if sub:
                            sub_counts[sub] = sub_counts.get(sub, 0) + 1
                    best_sub = max(sub_counts, key=sub_counts.get) if sub_counts else ""
                    for index in indices:
                        dino_feats[index] = avg_feat
                        dino_subs[index] = best_sub
            num_seqs = sum(1 for item in sequences if len(item["indices"]) > 1)
            self.log(f"  Unified features for {num_seqs} sequences ({unified_count} images)")
            return

        sorted_idx = list(range(len(accepted)))
        sorted_idx.sort(key=lambda item: natural_sort_key(accepted[item][0]))
        dir_seq_index: dict[str, list[int]] = {}
        sequences: list[dict[str, Any]] = []
        for item_index in sorted_idx:
            path = accepted[item_index][0]
            dirname = str(path.parent)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0
            base = get_clean_base(path.name)
            prefix = get_prefix(path.name)

            best_seq = None
            best_score = -1.0
            for seq_idx in dir_seq_index.get(dirname, [])[-20:]:
                seq = sequences[seq_idx]
                score = 0.0
                if abs(mtime - seq["first_mtime"]) <= 40:
                    score += 0.6
                if base and base == seq["base"]:
                    score += 0.4
                if prefix and prefix == seq["prefix"]:
                    score += 0.3
                if score >= 0.7 and score > best_score:
                    best_score = score
                    best_seq = seq

            if best_seq is not None:
                best_seq["indices"].append(item_index)
            else:
                seq_list_idx = len(sequences)
                sequences.append(
                    {
                        "dir": dirname,
                        "indices": [item_index],
                        "first_mtime": mtime,
                        "base": base,
                        "prefix": prefix,
                    }
                )
                dir_seq_index.setdefault(dirname, []).append(seq_list_idx)

        unified_count = 0
        for seq in sequences:
            indices = seq["indices"]
            if len(indices) > 1:
                unified_count += len(indices)
                sub_counts: dict[str, int] = {}
                for index in indices:
                    sub = accepted[index][1]
                    if sub:
                        sub_counts[sub] = sub_counts.get(sub, 0) + 1
                best_sub = max(sub_counts, key=sub_counts.get) if sub_counts else ""
                for index in indices:
                    accepted[index] = (accepted[index][0], best_sub)
        num_seqs = sum(1 for item in sequences if len(item["indices"]) > 1)
        self.log(f"  Unified folders for {num_seqs} sequences ({unified_count} images)")

    def _save_timing_summary(self, processed_count: int) -> None:
        try:
            self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = self.paths.logs_dir / f"sort_run_{timestamp}.log"
            with log_file.open("w", encoding="utf-8") as handle:
                handle.write("=== Image Sorting Performance Log ===\n")
                handle.write(f"Timestamp: {time.ctime()}\n")
                handle.write(f"Total Images Processed: {processed_count}\n")
                handle.write("-" * 40 + "\n")
                handle.write("\n[Phase 2: Preparations]\n")
                handle.write(f"  Scanning: {self._timings.get('scan', 0):.2f}s\n")
                handle.write(f"  Initialization (Tags): {self._timings.get('init', 0):.2f}s\n")
                total_time = self._timings.get("total", 0)
                handle.write("\n[Phase 3: Processing Loop]\n")
                handle.write(f"  Total Run Time: {total_time:.2f}s\n")
                if processed_count > 0 and total_time > 0:
                    handle.write(f"  Average Speed: {processed_count / total_time:.2f} images/sec\n")
                handle.write("\n" + "=" * 40 + "\n")
            self.log(f"Performance log saved: {log_file}")
        except Exception as exc:
            self.log(f"Warning: failed to save timing log: {exc}")
            self.log(traceback.format_exc())

    def _deduplicate(
        self,
        paths: list[Path],
        feat_list: list[torch.Tensor],
        subs: list[str],
        threshold: float,
    ) -> tuple[list[Path], list[torch.Tensor], list[str]]:
        feats = F.normalize(torch.cat(feat_list, dim=0), p=2, dim=1).float()
        removed: set[int] = set()
        chunk_size = 512
        for index in range(len(paths)):
            if index in removed:
                continue
            remaining = feats[index + 1 :]
            if remaining.shape[0] == 0:
                break
            for chunk_start in range(0, remaining.shape[0], chunk_size):
                chunk_end = min(chunk_start + chunk_size, remaining.shape[0])
                chunk = remaining[chunk_start:chunk_end]
                sims = torch.mm(feats[index : index + 1], chunk.t())[0]
                dup_indices = (sims >= threshold).nonzero(as_tuple=True)[0].tolist()
                for dup_index in dup_indices:
                    removed.add(index + 1 + chunk_start + dup_index)
        keep = [index for index in range(len(paths)) if index not in removed]
        return [paths[index] for index in keep], [feat_list[index] for index in keep], [subs[index] for index in keep]

    def _cluster_images(
        self,
        paths: list[Path],
        feat_list: list[torch.Tensor],
        subs: list[str],
        sensitivity: float,
        max_clusters: int = 0,
    ) -> dict[int, list[tuple[Path, str]]]:
        if not feat_list:
            return {}
        if len(feat_list) == 1:
            self.log("  Found 1 group (only 1 image)")
            return {0: [(paths[0], subs[0])]}

        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import euclidean_distances

        full_feats_tensor = F.normalize(torch.cat(feat_list, dim=0), p=2, dim=1).float().cpu()
        feats = full_feats_tensor.detach().numpy()
        n_samples = feats.shape[0]
        feat_dim = feats.shape[1]
        self.log(f"  Features: {n_samples} samples x {feat_dim} dims")

        umap_dim = min(20, feat_dim - 1, max(2, n_samples - 2))
        if feat_dim > 32 and n_samples > umap_dim + 2:
            try:
                import umap

                n_neighbors = max(2, min(15, n_samples - 1))
                self.log(f"  UMAP: {feat_dim}d -> {umap_dim}d (neighbors={n_neighbors})")
                reducer = umap.UMAP(
                    n_components=umap_dim,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
                feats = reducer.fit_transform(feats)
                self.log(f"  UMAP done -> {feats.shape[1]}d")
            except Exception as exc:
                self.log(f"  Warning: UMAP failed ({exc}), using raw features")
                self.log(traceback.format_exc())

        dists = euclidean_distances(feats)
        tri_idx = np.triu_indices_from(dists, k=1)
        flat_dists = dists[tri_idx]
        threshold = float(np.percentile(flat_dists, sensitivity * 100))
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="euclidean",
            linkage="ward",
        ).fit_predict(feats)

        label_to_indices: dict[int, list[int]] = {}
        groups: dict[int, list[tuple[Path, str]]] = {}
        for index, (path, sub, label) in enumerate(zip(paths, subs, labels)):
            label_id = int(label)
            label_to_indices.setdefault(label_id, []).append(index)
            groups.setdefault(label_id, []).append((path, sub))
        self.log(f"  Found {len(groups)} clusters")
        if max_clusters > 0 and len(groups) > max_clusters:
            groups = self._merge_visual_clusters_by_similarity(
                groups=groups,
                label_to_indices=label_to_indices,
                feats=full_feats_tensor,
                max_clusters=max_clusters,
            )
        return groups

    def _merge_visual_clusters_by_similarity(
        self,
        *,
        groups: dict[int, list[tuple[Path, str]]],
        label_to_indices: dict[int, list[int]],
        feats: torch.Tensor,
        max_clusters: int,
    ) -> dict[int, list[tuple[Path, str]]]:
        if max_clusters <= 0 or len(groups) <= max_clusters:
            return groups

        max_clusters = max(1, int(max_clusters))
        sorted_labels = sorted(groups, key=lambda label: len(groups[label]), reverse=True)
        anchor_labels = sorted_labels[:max_clusters]
        overflow_labels = sorted_labels[max_clusters:]

        def centroid_for(label: int) -> torch.Tensor:
            indices = label_to_indices.get(label, [])
            if not indices:
                return torch.zeros(feats.shape[1], dtype=feats.dtype)
            centroid = feats[indices].mean(dim=0)
            return F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

        anchor_centroids = {label: centroid_for(label) for label in anchor_labels}
        merged = {label: list(groups[label]) for label in anchor_labels}
        merged_indices = {label: list(label_to_indices.get(label, [])) for label in anchor_labels}
        reassigned_images = 0

        for label in overflow_labels:
            source_centroid = centroid_for(label)
            best_label = max(
                anchor_labels,
                key=lambda anchor: F.cosine_similarity(source_centroid, anchor_centroids[anchor], dim=0).item(),
            )
            merged[best_label].extend(groups[label])
            merged_indices[best_label].extend(label_to_indices.get(label, []))
            reassigned_images += len(groups[label])

            updated_indices = merged_indices[best_label]
            if updated_indices:
                anchor_centroids[best_label] = F.normalize(feats[updated_indices].mean(dim=0).unsqueeze(0), p=2, dim=1).squeeze(0)

        self.log(
            f"  Folder limit for visual grouping: {len(groups)} -> {len(merged)} clusters. "
            f"Reassigned {reassigned_images} images to nearest clusters."
        )
        return merged

    def _apply_folder_limit(self, accepted: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
        max_folders = int(self.config.get("max_folders_created", 0) or 0)
        if max_folders <= 0:
            return accepted

        counts: dict[str, int] = {}
        for _path, sub in accepted:
            sanitized = self._sanitize_subfolder_path(sub)
            if sanitized:
                counts[sanitized] = counts.get(sanitized, 0) + 1
        unique_folders = len(counts)
        if unique_folders <= max_folders:
            return accepted

        # Keep anchor folders, then remap overflow images to the most visually similar anchor folder.
        sorted_folders = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        keep_set = {name for name, _cnt in sorted_folders[:max_folders]}
        fallback_folder = sorted_folders[0][0]

        feat_cache_file = self.paths.feat_cache_file
        try:
            feat_cache = (
                torch.load(str(feat_cache_file), map_location="cpu", weights_only=True)
                if feat_cache_file.exists()
                else {}
            )
        except Exception:
            feat_cache = {}

        def _load_feat(path: Path) -> torch.Tensor | None:
            try:
                mtime = str(path.stat().st_mtime)
            except OSError:
                mtime = "0"

            dino_key = f"dinov2ms_{path}_{mtime}_{bool(self.config.get('use_dino_adapter', True))}"
            dino_val = feat_cache.get(dino_key)
            if isinstance(dino_val, torch.Tensor):
                vec = dino_val.view(-1).cpu().float()
                if vec.numel() > 0:
                    return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)

            siglip_key = f"siglip_{path}_{mtime}"
            siglip_val = feat_cache.get(siglip_key)
            if isinstance(siglip_val, torch.Tensor):
                vec = siglip_val.view(-1).cpu().float()
                if vec.numel() > 0:
                    return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
            return None

        folder_vectors: dict[str, list[torch.Tensor]] = {folder: [] for folder in keep_set}
        for path, sub in accepted:
            sanitized = self._sanitize_subfolder_path(sub)
            if sanitized in keep_set:
                vec = _load_feat(path)
                if vec is not None:
                    folder_vectors[sanitized].append(vec)

        folder_centroids: dict[str, torch.Tensor] = {}
        for folder, vectors in folder_vectors.items():
            if not vectors:
                continue
            centroid = torch.stack(vectors, dim=0).mean(dim=0)
            folder_centroids[folder] = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

        remapped: list[tuple[Path, str]] = []
        merged_count = 0
        similarity_routed = 0
        for path, sub in accepted:
            sanitized = self._sanitize_subfolder_path(sub)
            if not sanitized:
                remapped.append((path, sub))
                continue
            if sanitized in keep_set:
                remapped.append((path, sanitized))
            else:
                vec = _load_feat(path)
                best_folder = None
                best_score = -1.0
                if vec is not None and folder_centroids:
                    for folder, centroid in folder_centroids.items():
                        score = F.cosine_similarity(vec, centroid, dim=0).item()
                        if score > best_score:
                            best_score = score
                            best_folder = folder
                if best_folder:
                    remapped.append((path, best_folder))
                    similarity_routed += 1
                else:
                    remapped.append((path, fallback_folder))
                merged_count += 1

        self.log(
            f"Folder limit applied: {unique_folders} -> {max_folders}. "
            f"Reassigned by similarity: {similarity_routed}. "
            f"Fallback merged: {merged_count - similarity_routed} -> '{fallback_folder}'."
        )
        return remapped

    @staticmethod
    def _normalized_tag(tag: str) -> str:
        cleaned = re.sub(r"[\s\-]+", "_", str(tag).strip().lower())
        aliases = {
            "1_girl": "1girl",
            "solo_female": "1girl",
            "solo_male": "1boy",
            "multiple_girl": "multiple_girls",
            "multiple_boy": "multiple_boys",
            "kitchenknife": "kitchen_knife",
            "pinkapron": "pink_apron",
            "sea": "ocean",
            "seaside": "beach",
            "indoors_scene": "indoors",
            "outdoors_scene": "outdoors",
        }
        return aliases.get(cleaned, cleaned)

    @staticmethod
    def _display_tag(tag: str) -> str:
        token = str(tag).strip().lower()
        special = {
            "1girl": "1Girl",
            "1boy": "1Boy",
            "2girls": "2Girls",
            "2boys": "2Boys",
            "multiple_girls": "Multiple Girls",
            "multiple_boys": "Multiple Boys",
        }
        if token in special:
            return special[token]
        return token.replace("_", " ").title()

    @staticmethod
    def _folder_name_tokens(value: str) -> set[str]:
        text = re.sub(r"[_\\/]+", " ", str(value).lower())
        return {token for token in re.findall(r"[a-zа-яё0-9]+", text) if token}

    @classmethod
    def _folder_name_similarity(cls, left: str, right: str) -> float:
        left_tokens = cls._folder_name_tokens(left)
        right_tokens = cls._folder_name_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        jaccard = intersection / max(union, 1)
        containment = max(
            intersection / max(len(left_tokens), 1),
            intersection / max(len(right_tokens), 1),
        )
        left_text = " ".join(sorted(left_tokens))
        right_text = " ".join(sorted(right_tokens))
        substring = 1.0 if left_text in right_text or right_text in left_text else 0.0
        return max(jaccard, containment * 0.9, substring)

    @staticmethod
    def _generic_folder_name_tokens() -> set[str]:
        return {
            "1girl",
            "1boy",
            "2girls",
            "2boys",
            "girl",
            "girls",
            "boy",
            "boys",
            "female",
            "male",
            "character",
            "solo",
            "portrait",
            "close",
            "up",
            "smile",
            "smiling",
            "looking",
            "viewer",
            "standing",
            "sitting",
            "long",
            "short",
            "hair",
            "shirt",
            "white",
            "black",
            "background",
            "simple",
            "indoors",
            "outdoors",
            "person",
            "people",
        }

    @classmethod
    def _canonical_folder_name(cls, value: str) -> str:
        tokens = cls._folder_name_tokens(value)
        parts: list[str] = []
        if {"1girl", "girl", "girls", "woman", "female"} & tokens:
            parts.append("Female Character")
        elif {"1boy", "boy", "boys", "man", "male"} & tokens:
            parts.append("Male Character")
        cleaned = str(value)
        cleaned = re.sub(r"\b(?:1girl|girl|girls|woman|female|1boy|boy|boys|man|male)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            parts.append(cleaned)
        return safe_filename(" ".join(parts) if parts else value)

    @classmethod
    def _hierarchical_folder_name(cls, value: str) -> str:
        tokens = cls._folder_name_tokens(value)
        if {"female", "male", "character", "1girl", "1boy", "girl", "boy"} & tokens:
            return str(Path("Character") / safe_filename(value))
        if {"beach", "forest", "city", "street", "classroom", "kitchen", "bedroom", "outdoors", "indoors"} & tokens:
            return str(Path("Scene") / safe_filename(value))
        if {"sketch", "watercolor", "pixel", "anime", "photo", "render"} & tokens:
            return str(Path("Style") / safe_filename(value))
        return value

    @classmethod
    def _display_character_name(cls, tag: str) -> str:
        token = cls._normalized_tag(tag)
        token = re.sub(r"_\([^)]*\)$", "", token)
        token = re.sub(r"\([^)]*\)$", "", token)
        token = token.replace("_", " ").strip()
        if not token:
            return ""
        return safe_filename(token.title())

    def _source_character_hint(self, path: Path) -> str:
        if not self.config.get("recursive_scan"):
            return ""
        try:
            source = Path(self.config.get("source_dir", "")).expanduser().resolve()
            rel_parent = path.resolve().parent.relative_to(source)
        except Exception:
            return ""
        parts = [part for part in rel_parent.parts if part and part not in {".", ".."}]
        if not parts:
            return ""
        ignored = {
            "images",
            "image",
            "img",
            "pics",
            "pictures",
            "photos",
            "new",
            "old",
            "misc",
            "other",
            "sorted",
            "unsorted",
            "unknown",
            "download",
            "downloads",
        }
        for part in parts:
            cleaned = safe_filename(str(part).replace("_", " ").replace("-", " "))
            if cleaned and cleaned.lower() not in ignored and not self._contains_year_token(cleaned):
                return cleaned.title()
        return ""

    def _character_folder_for_image(self, path: Path, preds: dict[str, float] | None) -> str:
        if not (self.config.get("character_aware_recursive") and self.config.get("recursive_scan")):
            return ""

        min_score = max(0.01, float(self.config.get("character_min_score", 0.35)))
        margin = max(0.0, float(self.config.get("character_margin", 0.08)))
        max_multi = max(2, int(self.config.get("character_max_multi", 4)))
        characters: list[tuple[str, float]] = []
        if preds and self.manager.wd_tag_categories:
            for raw_tag, score in preds.items():
                category = self.manager.wd_tag_categories.get(raw_tag, -1)
                if category in (4, "character") and score >= min_score:
                    display = self._display_character_name(raw_tag)
                    if display:
                        characters.append((display, float(score)))

        deduped: dict[str, float] = {}
        for name, score in characters:
            deduped[name] = max(deduped.get(name, 0.0), score)
        characters = sorted(deduped.items(), key=lambda item: item[1], reverse=True)

        if len(characters) >= 2 and self.config.get("character_create_multiple"):
            selected = characters[:max_multi]
            return safe_filename(" ".join(name for name, _score in selected))

        if characters:
            top_name, top_score = characters[0]
            second_score = characters[1][1] if len(characters) > 1 else 0.0
            if len(characters) == 1 or (top_score - second_score) >= margin:
                return top_name

        return self._source_character_hint(path)

    def _prepend_character_folder(self, path: Path, sub: str, preds: dict[str, float] | None = None) -> str:
        character_folder = self._character_folder_for_image(path, preds)
        if not character_folder:
            return sub
        sanitized_character = self._sanitize_subfolder_path(character_folder)
        sanitized_sub = self._sanitize_subfolder_path(sub)
        if not sanitized_character:
            return sub
        if sanitized_sub:
            first_part = Path(sanitized_sub).parts[0]
            if first_part.lower() == sanitized_character.lower():
                return sanitized_sub
            return str(Path(sanitized_character) / sanitized_sub)
        return sanitized_character

    def _semantic_name_similarities(self, names: list[str]) -> dict[tuple[int, int], float]:
        if len(names) < 2 or self.manager.siglip_model is None or self.manager.siglip_proc is None:
            return {}
        try:
            with torch.inference_mode():
                embeddings = self.manager.siglip_embed_texts(names).cpu()
            sims: dict[tuple[int, int], float] = {}
            for left in range(len(names)):
                for right in range(left + 1, len(names)):
                    sims[(left, right)] = F.cosine_similarity(embeddings[left], embeddings[right], dim=0).item()
            return sims
        except Exception as exc:
            self.log(f"Warning: semantic folder-name comparison unavailable: {exc}")
            return {}

    def _optimize_cluster_folder_names(
        self,
        final_names: dict[int, str],
        cluster_candidate_tags: dict[int, list[tuple[str, float, float]]],
    ) -> dict[int, str]:
        if not final_names:
            return final_names

        generic_tokens = self._generic_folder_name_tokens()
        uniqueness = str(self.config.get("name_uniqueness_level", "Medium")).strip().lower()
        lexical_thresholds = {"low": 0.88, "medium": 0.74, "high": 0.62}
        semantic_thresholds = {"low": 0.94, "medium": 0.90, "high": 0.86}
        lexical_threshold = lexical_thresholds.get(uniqueness, lexical_thresholds["medium"])
        semantic_threshold = semantic_thresholds.get(uniqueness, semantic_thresholds["medium"])

        tag_cluster_counts: dict[str, int] = {}
        for candidates in cluster_candidate_tags.values():
            seen = {tag for tag, _coverage, _avg_score in candidates}
            for tag in seen:
                tag_cluster_counts[tag] = tag_cluster_counts.get(tag, 0) + 1
        total_clusters = max(len(final_names), 1)

        def is_generic_tag(tag: str) -> bool:
            tokens = self._folder_name_tokens(tag)
            return tag in generic_tokens or (bool(tokens) and tokens <= generic_tokens)

        def distinctive_tags(cluster_id: int) -> list[str]:
            scored: list[tuple[float, str]] = []
            for tag, coverage, avg_score in cluster_candidate_tags.get(cluster_id, []):
                if self._contains_year_token(tag) or is_generic_tag(tag):
                    continue
                rarity = math.log((total_clusters + 1) / (1 + tag_cluster_counts.get(tag, 0))) + 1.0
                scored.append((coverage * avg_score * rarity, tag))
            scored.sort(reverse=True)
            return [tag for _score, tag in scored]

        optimized: dict[int, str] = {}
        for cluster_id, raw_name in final_names.items():
            name = safe_filename(raw_name)
            if self.config.get("canonical_folder_names"):
                name = self._canonical_folder_name(name)

            name_tokens = self._folder_name_tokens(name)
            non_generic = name_tokens - generic_tokens
            if not non_generic:
                distinct = distinctive_tags(cluster_id)
                if distinct:
                    name = safe_filename(self._display_tag(distinct[0]))
            optimized[cluster_id] = name

        if not self.config.get("global_name_optimization", True):
            if self.config.get("hierarchical_folder_names"):
                optimized = {cluster_id: self._hierarchical_folder_name(name) for cluster_id, name in optimized.items()}
            return optimized

        cluster_ids = list(optimized.keys())
        names = [optimized[cluster_id] for cluster_id in cluster_ids]
        semantic_sims = self._semantic_name_similarities(names)
        used_names: dict[str, int] = {}
        changed = 0

        for position, cluster_id in enumerate(cluster_ids):
            name = optimized[cluster_id]
            distinct = distinctive_tags(cluster_id)

            if name in used_names:
                for tag in distinct:
                    display = self._display_tag(tag)
                    candidate = safe_filename(display)[:60]
                    if candidate not in used_names:
                        name = candidate
                        changed += 1
                        break
                    candidate = safe_filename(f"{name} {display}")[:60]
                    if candidate not in used_names:
                        name = candidate
                        changed += 1
                        break

            for previous_position in range(position):
                previous_id = cluster_ids[previous_position]
                previous_name = optimized[previous_id]
                lexical_sim = self._folder_name_similarity(name, previous_name)
                semantic_sim = semantic_sims.get((previous_position, position), semantic_sims.get((position, previous_position), 0.0))
                if lexical_sim < lexical_threshold and semantic_sim < semantic_threshold:
                    continue

                clarified = False
                for tag in distinct:
                    display = self._display_tag(tag)
                    for candidate in (
                        safe_filename(display)[:60],
                        safe_filename(f"{name} {display}")[:60],
                    ):
                        if (
                            candidate
                            and candidate not in used_names
                            and self._folder_name_similarity(candidate, previous_name) < lexical_threshold
                        ):
                            name = candidate
                            clarified = True
                            changed += 1
                            break
                    if clarified:
                        break
                if not clarified:
                    name = previous_name
                    changed += 1
                break

            optimized[cluster_id] = name
            used_names[name] = cluster_id

        if self.config.get("hierarchical_folder_names"):
            optimized = {cluster_id: self._hierarchical_folder_name(name) for cluster_id, name in optimized.items()}

        if changed:
            self.log(f"Optimized folder names: adjusted/merged {changed} cluster names (uniqueness={uniqueness.title()}).")
        return optimized

    @staticmethod
    def _contains_year_token(value: str) -> bool:
        text = str(value).lower()
        return bool(
            re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|2100)(?!\d)", text)
            or re.search(r"\byears?\b", text)
            or re.search(r"\byear[_\-\s]*(?:19\d{2}|20\d{2}|2100)\b", text)
        )

    @classmethod
    def _strip_year_tokens(cls, value: str) -> str:
        cleaned = str(value)
        cleaned = re.sub(r"\byear[_\-\s]*(?:19\d{2}|20\d{2}|2100)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:19\d{2}|20\d{2}|2100)[_\-\s]*years?\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?<!\d)(?:19\d{2}|20\d{2}|2100)(?!\d)", " ", cleaned)
        cleaned = re.sub(r"\byears?\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-()[]{}")
        return cleaned

    @classmethod
    def _sanitize_subfolder_path(cls, sub: str) -> str:
        if not sub:
            return ""
        parts = [cls._strip_year_tokens(part) for part in re.split(r"[\\/]+", sub) if part]
        parts = [safe_filename(part) for part in parts if part]
        return str(Path(*parts)) if parts else ""

    def _wd_tagger_name_clusters(
        self,
        clustered_groups: dict[int, list[tuple[Path, str]]],
        wd_preds_cache: dict[Path, dict[str, float]],
    ) -> list[tuple[Path, str]]:
        safe_naming_mode = bool(self.config.get("safe_naming_mode", True))
        consensus_ratio = float(self.config.get("name_consensus_ratio", 0.65))
        min_tag_score = float(self.config.get("name_min_tag_score", 0.18))
        char_min_score = float(self.config.get("name_char_min_score", 0.55))
        char_margin = float(self.config.get("name_char_margin", 0.12))
        max_parts = 3 if safe_naming_mode else 5

        exclude_tags = {
            "monochrome",
            "greyscale",
            "no_humans",
            "negative_space",
            "simple_background",
            "white_background",
            "black_background",
            "transparent_background",
            "sketch",
            "lineart",
            "comic",
            "manga",
            "gradient_background",
            "pattern_background",
            "traditional_media",
            "watercolor_(medium)",
            "pixel_art",
            "oikakeko",
            "highres",
            "absurdres",
            "commentary_request",
            "commentary",
            "translated",
            "bad_id",
            "bad_pixiv_id",
            "revision",
            "character_request",
        }
        weak_name_tags = {"1girl", "1boy", "multiple_girls", "multiple_boys", "solo", "portrait", "close_up"}

        self._folder_uncertainty = {}
        cluster_names: dict[int, str] = {}
        cluster_candidate_tags: dict[int, list[tuple[str, float, float]]] = {}
        cluster_uncertain: dict[int, bool] = {}
        cluster_uncertain_reasons: dict[int, list[str]] = {}

        for cluster_id, members in clustered_groups.items():
            missing_paths = [path for path, _sub in members if path not in wd_preds_cache]
            if missing_paths and self.manager.wd_tagger is not None:
                try:
                    batch_images: list[Image.Image] = []
                    batch_image_paths: list[Path] = []
                    for path in missing_paths:
                        try:
                            batch_images.append(self._open_image_for_path(path))
                            batch_image_paths.append(path)
                        except Exception as exc:
                            self.log(f"Warning: failed to open {path.name} for cluster naming: {exc}")
                            self.log(traceback.format_exc())
                    tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", len(batch_images))
                    for start in range(0, len(batch_images), tagger_batch_size):
                        batch_preds = self.manager.wd_tagger_infer_batch(batch_images[start : start + tagger_batch_size])
                        for path, preds in zip(batch_image_paths[start : start + tagger_batch_size], batch_preds):
                            wd_preds_cache[path] = preds
                except Exception as exc:
                    self.log(f"Warning: batch naming-tagging failed: {exc}")
                    self.log(traceback.format_exc())

            total_images = max(len(members), 1)
            tag_count: dict[str, int] = {}
            tag_sum: dict[str, float] = {}
            char_count: dict[str, int] = {}
            char_sum: dict[str, float] = {}

            for path, _sub in members:
                preds = wd_preds_cache.get(path)
                if not preds:
                    continue
                seen_tag_in_image: set[str] = set()
                seen_char_in_image: set[str] = set()
                for raw_tag, score in preds.items():
                    norm_tag = self._normalized_tag(raw_tag)
                    category = self.manager.wd_tag_categories.get(raw_tag, -1)
                    if category in (9, "rating") or norm_tag in exclude_tags:
                        continue
                    if category in (4, "character"):
                        if score >= 0.2 and norm_tag not in seen_char_in_image:
                            char_count[norm_tag] = char_count.get(norm_tag, 0) + 1
                            seen_char_in_image.add(norm_tag)
                        if score >= 0.05:
                            char_sum[norm_tag] = char_sum.get(norm_tag, 0.0) + score
                        continue
                    if score < min_tag_score:
                        continue
                    if norm_tag not in seen_tag_in_image:
                        tag_count[norm_tag] = tag_count.get(norm_tag, 0) + 1
                        seen_tag_in_image.add(norm_tag)
                    tag_sum[norm_tag] = tag_sum.get(norm_tag, 0.0) + score

            consensus_tags: list[tuple[str, float, float]] = []
            for tag, count in tag_count.items():
                coverage = count / total_images
                avg_score = tag_sum.get(tag, 0.0) / max(count, 1)
                if coverage >= consensus_ratio:
                    consensus_tags.append((tag, coverage, avg_score))
            consensus_tags.sort(key=lambda item: (item[1], item[2]), reverse=True)

            char_candidates: list[tuple[str, float, float]] = []
            for tag, count in char_count.items():
                coverage = count / total_images
                avg_score = char_sum.get(tag, 0.0) / max(count, 1)
                char_candidates.append((tag, coverage, avg_score))
            char_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)

            uncertain_reasons: list[str] = []
            selected_char: str | None = None
            if char_candidates:
                top_tag, top_cov, top_avg = char_candidates[0]
                second_avg = char_candidates[1][2] if len(char_candidates) > 1 else 0.0
                if top_avg >= char_min_score and (top_avg - second_avg) >= char_margin and top_cov >= max(0.4, consensus_ratio - 0.15):
                    selected_char = top_tag
                else:
                    uncertain_reasons.append("character_conflict_or_low_confidence")
                if len([item for item in char_candidates if item[1] >= 0.3]) > 1:
                    uncertain_reasons.append("multiple_characters")

            if not consensus_tags:
                uncertain_reasons.append("weak_tag_consensus")

            candidate_tags = list(consensus_tags)
            if selected_char is not None:
                candidate_tags.insert(0, (selected_char, 1.0, char_candidates[0][2] if char_candidates else 1.0))
            cluster_candidate_tags[cluster_id] = candidate_tags

            name_parts: list[str] = []
            if selected_char is not None:
                name_parts.append(self._display_tag(selected_char))

            for tag, _coverage, _avg_score in consensus_tags:
                if len(name_parts) >= max_parts:
                    break
                if tag == selected_char or tag in weak_name_tags:
                    continue
                if self._contains_year_token(tag):
                    continue
                display = self._display_tag(tag)
                if display not in name_parts:
                    name_parts.append(display)

            if safe_naming_mode:
                name_parts = name_parts[:max_parts]
            if not name_parts and selected_char is not None:
                name_parts = [self._display_tag(selected_char)]
            if not name_parts:
                uncertain_reasons.append("fallback_name_used")
                folder_name = f"group_{cluster_id + 1:03d}"
            else:
                folder_name = safe_filename(self._strip_year_tokens(" ".join(name_parts)))[:60]

            cluster_names[cluster_id] = folder_name
            cluster_uncertain[cluster_id] = bool(uncertain_reasons)
            cluster_uncertain_reasons[cluster_id] = sorted(set(uncertain_reasons))
            self.log(
                f"    Cluster {cluster_id + 1}: \"{folder_name}\" "
                f"({len(members)} imgs, uncertain={cluster_uncertain[cluster_id]})"
            )

        name_groups: dict[str, list[int]] = {}
        for cluster_id, name in cluster_names.items():
            name_groups.setdefault(name, []).append(cluster_id)

        final_names: dict[int, str] = {}
        for base_name, cluster_ids in name_groups.items():
            for cluster_id in cluster_ids:
                final_names[cluster_id] = base_name

        final_names = self._optimize_cluster_folder_names(final_names, cluster_candidate_tags)

        if self.config.get("remember_cluster_names", True):
            memory = read_json(self.paths.cache_dir / "cluster_name_memory.json", {}, self.log_callback)
            if isinstance(memory, dict) and memory:
                applied = 0
                for cluster_id, folder in list(final_names.items()):
                    remembered = memory.get(folder) or memory.get(self._sanitize_subfolder_path(folder))
                    if remembered:
                        final_names[cluster_id] = safe_filename(str(remembered))
                        applied += 1
                if applied:
                    self.log(f"Applied remembered cluster names: {applied}")

        result: list[tuple[Path, str]] = []
        for cluster_id, members in clustered_groups.items():
            folder = final_names.get(cluster_id, f"group_{cluster_id + 1:03d}")
            for path, sub in members:
                parts = [part for part in [self._sanitize_subfolder_path(sub), self._sanitize_subfolder_path(folder)] if part]
                combined = str(Path(*parts)) if parts else ""
                result.append((path, combined))
                if combined:
                    self._folder_uncertainty[combined] = {
                        "uncertain": cluster_uncertain.get(cluster_id, False),
                        "uncertain_reasons": cluster_uncertain_reasons.get(cluster_id, []),
                        "safe_naming_mode": safe_naming_mode,
                        "consensus_ratio": consensus_ratio,
                    }
        return result

    def _get_florence_profile_defaults(self) -> tuple[int, int]:
        profile = str(self.config.get("florence_profile", "Balanced")).strip().lower()
        if profile == "fast":
            return 8, 640
        if profile == "quality":
            return 2, 1024
        return 4, 768

    def _resolution_bucket_key(self, path: Path, max_side: int) -> tuple[int, int]:
        try:
            image = self._open_image_for_path(path)
            width, height = image.size
            scale = min(1.0, max_side / max(width, height, 1))
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))
            ratio_bin = int(round((width / max(height, 1)) * 8))
            area_bin = (width * height) // (128 * 128)
            return ratio_bin, area_bin
        except Exception:
            return 0, 0

    def _run_florence_metadata_queue(
        self,
        caption_queue: list[tuple[Path, str]],
        images_meta: dict[str, Any],
    ) -> None:
        if not caption_queue or self.manager.florence_model is None:
            return
        if self.manager.device == "cuda" and bool(self.config.get("unload_inactive_models", True)):
            self.manager.offload_non_florence_to_cpu()
        default_batch, default_max_side = self._get_florence_profile_defaults()
        batch_size = self._batch_size("florence", "florence_batch_size", default_batch)
        max_side = max(256, int(self.config.get("florence_max_side", default_max_side)))
        ordered_queue = sorted(caption_queue, key=lambda item: self._resolution_bucket_key(item[0], max_side))

        cpu_count = os.cpu_count() or 1
        num_workers = 0 if sys.platform == "win32" else min(4, max(cpu_count - 1, 1))
        loader = DataLoader(
            FlorenceQueueDataset(
                ordered_queue,
                max_side=max_side,
                lmdb_dir=self._image_lmdb_dir,
                lmdb_source=self._image_lmdb_source,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.manager.device == "cuda"),
            collate_fn=florence_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=3 if num_workers > 0 else None,
        )

        self.log(
            f"  Florence-2 batch queue: {len(ordered_queue)} images "
            f"(batch={batch_size}, max_side={max_side}, workers={num_workers})"
        )
        total_items = len(ordered_queue)
        done_items = 0
        florence_start = time.time()
        for records in loader:
            if self.is_cancelled():
                break
            valid = [record for record in records if record.image is not None]
            bad = [record for record in records if record.image is None]
            for record in bad:
                self.log(f"Warning: Florence skipped {record.path.name}: {record.error or 'image load error'}")
                if record.path.name in images_meta:
                    images_meta[record.path.name]["caption_florence2_error"] = record.error or "image load error"

            if valid:
                if self.is_cancelled():
                    break
                captions = self.manager.florence_infer_batch(
                    [record.image for record in valid if record.image is not None],
                    [record.prompt_tags for record in valid],
                )
                for record, caption in zip(valid, captions):
                    if caption:
                        images_meta.setdefault(record.path.name, {})["caption_florence2"] = caption

            done_items += len(records)
            self._emit_eta(
                stage="Florence metadata",
                completed=min(done_items, total_items),
                total=total_items,
                start_ts=florence_start,
                force=(done_items >= total_items),
            )
            if self._is_soft_stop_requested():
                self._mark_soft_stop("Stop-after-batch requested during Florence metadata. Saving current results...")
                break

    def _wd_tagger_generate_metadata(
        self,
        accepted: list[tuple[Path, str]],
        target: Path,
        wd_preds_cache: dict[Path, dict[str, float]],
    ) -> None:
        folder_map: dict[str, list[Path]] = {}
        for path, sub in accepted:
            folder_map.setdefault(sub or "", []).append(path)

        mature_patterns = [
            "nude",
            "sex",
            "penis",
            "vagina",
            "nipple",
            "pussy",
            "ass",
            "breast",
            "cum",
            "genital",
            "anus",
            "orgasm",
            "erect",
            "pubic",
            "naked",
            "masturbat",
            "porn",
            "hentai",
            "dildo",
            "bondage",
            "rape",
            "tentacle",
        ]

        total_folders = len(folder_map)
        metadata_start = time.time()
        for done, (subfolder, paths) in enumerate(folder_map.items(), start=1):
            if self.is_cancelled():
                break
            if self._is_soft_stop_requested():
                self._mark_soft_stop("Stop-after-batch requested before next metadata folder. Continuing to file save...")
                break
            max_n = int(self.config.get("meta_max_per_folder", 0))
            sample = paths[:] if max_n <= 0 else random.sample(paths, min(max_n, len(paths)))
            tags_per_img = int(self.config.get("meta_tags_per_image", 30))
            images_meta: dict[str, Any] = {}
            has_mature_folder = False
            florence_queue: list[tuple[Path, str]] = []
            folder_hint = self._folder_uncertainty.get(subfolder or "", {})
            folder_uncertain = bool(folder_hint.get("uncertain", False))
            uncertain_reasons = list(folder_hint.get("uncertain_reasons", []))
            dominant_character_counts: dict[str, int] = {}

            for path in sample:
                if self.is_cancelled():
                    break
                preds = wd_preds_cache.get(path, {})
                if not preds:
                    images_meta[path.name] = {"error": "no tags found"}
                    continue

                rating = {tag: round(preds.get(tag, 0.0), 3) for tag in ["explicit", "questionable", "sensitive", "general"]}
                general_tags: list[str] = []
                character_tags: list[str] = []
                mature_tags: list[str] = []
                all_scores: dict[str, float] = {}
                seen_general: set[str] = set()
                seen_characters: set[str] = set()
                seen_mature: set[str] = set()

                content_tags = [
                    (tag, score)
                    for tag, score in preds.items()
                    if self.manager.wd_tag_categories.get(tag, -1) not in (9, "rating") and score > 0.07
                ]
                content_tags.sort(key=lambda item: item[1], reverse=True)

                for tag, score in content_tags[:tags_per_img]:
                    norm_tag = self._normalized_tag(tag)
                    if self._contains_year_token(norm_tag):
                        continue
                    all_scores[norm_tag] = max(all_scores.get(norm_tag, 0.0), round(score, 3))
                    category = self.manager.wd_tag_categories.get(tag, 0)
                    is_mature = any(pattern in norm_tag.lower() for pattern in mature_patterns)
                    display = self._display_tag(norm_tag)
                    if is_mature:
                        if display not in seen_mature:
                            mature_tags.append(display)
                            seen_mature.add(display)
                        has_mature_folder = True
                    elif category in (4, "character"):
                        if display not in seen_characters:
                            character_tags.append(display)
                            seen_characters.add(display)
                    else:
                        if display not in seen_general:
                            general_tags.append(display)
                            seen_general.add(display)

                if character_tags:
                    dominant_character_counts[character_tags[0]] = dominant_character_counts.get(character_tags[0], 0) + 1

                images_meta[path.name] = {
                    "rating": rating,
                    "general_tags": general_tags,
                    "character_tags": character_tags,
                    "mature_tags": mature_tags,
                    "all_scores": all_scores,
                }
                if self.config.get("use_florence") and self.manager.florence_model is not None:
                    top_wd = ", ".join(character_tags + general_tags[:10])
                    florence_queue.append((path, top_wd))

                log_tag_str = ", ".join(general_tags[:5] + character_tags[:2] + mature_tags[:2])
                self.log(f"    {path.name}: {log_tag_str[:60]}...")

            if florence_queue and not self.is_cancelled():
                self._run_florence_metadata_queue(florence_queue, images_meta)

            if self.is_cancelled():
                break

            if dominant_character_counts:
                total_with_char = sum(dominant_character_counts.values())
                top_char_count = max(dominant_character_counts.values())
                if total_with_char >= 2 and (top_char_count / total_with_char) < 0.6:
                    folder_uncertain = True
                    uncertain_reasons.append("mixed_characters_in_folder")

            metadata = {
                "folder": self._sanitize_subfolder_path(subfolder) or "root",
                "total_images": len(paths),
                "sampled": len(sample),
                "images": images_meta,
                "uncertain": folder_uncertain,
                "uncertain_reasons": sorted(set(uncertain_reasons)),
                "safe_naming_mode": bool(folder_hint.get("safe_naming_mode", True)),
            }

            if has_mature_folder:
                self.log(f"  {subfolder or 'root'}: contains explicit/mature tags")

            sanitized_subfolder = self._sanitize_subfolder_path(subfolder)
            dst_dir = target / sanitized_subfolder if sanitized_subfolder else target
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                with (dst_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                    import json

                    json.dump(metadata, handle, ensure_ascii=False, indent=2)
            except Exception as exc:
                self.log(f"Warning: could not write metadata for '{subfolder or 'root'}': {exc}")
                self.log(traceback.format_exc())

            self.prog(92 + done / max(total_folders, 1) * 3)
            self._emit_eta(
                stage="Metadata folders",
                completed=done,
                total=total_folders,
                start_ts=metadata_start,
                force=(done >= total_folders),
            )
            if self._is_soft_stop_requested():
                self._mark_soft_stop("Stop-after-batch requested during metadata. Continuing to file save...")
                break

    def _flush_move_journal(self, result: dict[str, Any]) -> dict[str, Any]:
        if not self.config.get("move_files"):
            return result
        if not self._move_journal:
            result["undo_available"] = False
            return result
        target = self._move_journal_target or Path(self.config.get("target_dir", ".")).expanduser().resolve()
        journal_file = target / ".undo_last_move.json"
        payload = {
            "created_at": self._run_started_at,
            "source_dir": str(Path(self.config.get("source_dir", "")).expanduser()),
            "target_dir": str(target),
            "operations": self._move_journal,
            "summary": {
                "ok": bool(result.get("ok")),
                "message": str(result.get("message", "")),
                "processed": int(result.get("processed", 0)),
                "saved": int(result.get("saved", 0)),
                "errors": int(result.get("errors", 0)),
            },
        }
        if write_json(journal_file, payload, self.log_callback):
            self.log(f"Undo journal saved: {journal_file}")
            result["undo_available"] = True
            result["undo_journal_path"] = str(journal_file)
        else:
            result["undo_available"] = False
        return result

    def _append_run_history(self, result: dict[str, Any]) -> None:
        try:
            history_file = self.paths.logs_dir / "run_history.json"
            existing = read_json(history_file, [], self.log_callback)
            if not isinstance(existing, list):
                existing = []
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ok": bool(result.get("ok")),
                "message": str(result.get("message", "")),
                "source": str(self.config.get("source_dir", "")),
                "target": str(self.config.get("target_dir", "")),
                "processed": int(result.get("processed", 0)),
                "saved": int(result.get("saved", 0)),
                "filtered": int(result.get("filtered", 0)),
                "errors": int(result.get("errors", 0)),
                "elapsed": float(result.get("elapsed", 0.0)),
                "needs_review": int(result.get("uncertain_saved", 0)),
                "mode": "move" if self.config.get("move_files") else "copy",
                "stopped_after_batch": bool(result.get("stopped_after_batch", False)),
                "criteria": {
                    "ai_human": bool(self.config.get("sort_ai_human")),
                    "content": bool(self.config.get("sort_content")),
                    "style": bool(self.config.get("sort_style")),
                    "grouping": bool(self.config.get("sort_grouping")),
                    "dedup": bool(self.config.get("sort_dedup")),
                    "metadata": bool(self.config.get("gen_metadata")),
                },
                "settings": {
                    key: self.config.get(key)
                    for key in SETTINGS_BOOL_KEYS + SETTINGS_STRING_KEYS + SETTINGS_NUMBER_KEYS
                    if key in self.config
                },
            }
            existing.append(entry)
            existing = existing[-200:]
            write_json(history_file, existing, self.log_callback)
        except Exception as exc:
            self.log(f"Warning: failed to append run history: {exc}")

    def _place(self, src: Path, dst_dir: Path) -> bool:
        destination = dst_dir / src.name
        if src.resolve() == destination.resolve():
            return True
        counter = 1
        while destination.exists():
            destination = dst_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            if self.config.get("dry_run"):
                action = "move" if self.config.get("move_files") else "copy"
                self.log(f"Dry run: would {action} {src.name} -> {destination}")
                return True
            dst_dir.mkdir(parents=True, exist_ok=True)
            mover = shutil.move if self.config.get("move_files") else shutil.copy2
            original_src = src
            mover(src, destination)
            if self.config.get("move_files"):
                self._move_journal.append({"src": str(original_src), "dst": str(destination)})
            return True
        except shutil.SameFileError:
            self.log(f"Warning: source and destination are the same for {src}")
            return True
        except OSError as exc:
            self.log(f"Warning: could not save {src}: {exc}")
            return False
