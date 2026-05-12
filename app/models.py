from __future__ import annotations

import gc
import json
import logging
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .utils import (
    LoadingBar,
    MODEL_SIZES_GB,
    detect_runtime,
    emit_log,
    emit_progress,
    format_size,
    get_paths,
    make_ensembles,
    resolve_vram_limit_bytes,
)

logger = logging.getLogger(__name__)


class DINOv2Adapter(nn.Module):
    """MLP adapter: projects DINOv2 multi-scale vector (1536) -> text-embedding space."""

    def __init__(
        self,
        input_dim: int = 1536,
        output_dim: int = 384,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ModelManager:
    def __init__(
        self,
        progress_callback: Callable[[float], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.paths = get_paths()
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.ai_model = self.ai_proc = None
        self.sdxl_model = self.sdxl_proc = None
        self.siglip_model = self.siglip_proc = None
        self.dino_model = self.dino_proc = None
        self.dino_adapter = None
        self.wd_tagger = None
        self.wd_tags = None
        self.wd_tag_categories = None
        self.florence_model = None
        self.florence_processor = None
        self.runtime: dict[str, Any] = detect_runtime("auto")
        self.precision_choice = "auto"
        self.optimize_models = False
        self.tagger_engine = "Camie-Tagger-v2"
        self.use_dino_adapter_flag = True
        self.group_sequences = True
        self.camie_threshold = 0.05
        self.florence_mode = "<DETAILED_CAPTION>"
        self.florence_char = ""
        self.florence_profile = "Balanced"
        self.florence_batch_size = 4
        self.florence_max_side = 768
        self.florence_max_new_tokens = 0
        self.auto_vram_batch = True
        self.unload_inactive_models = True
        self.vram_profile = "Auto"
        self.vram_limit_gb = 0.0
        self.vram_limit_bytes = 0
        self._logged_batch_suggestions: set[tuple[str, int, int]] = set()
        self._last_gpu_status_at = 0.0
        self._last_gpu_status_text = ""
        self._compile_block_reason: str | None = None
        self._logged_non_florence_offload = False
        self._florence_generation_defaults = {
            "<DETAILED_CAPTION>": {"max_new_tokens": 256, "num_beams": 2},
            "<MORE_DETAILED_CAPTION>": {"max_new_tokens": 384, "num_beams": 2},
            "<MIXED_CAPTION>": {"max_new_tokens": 320, "num_beams": 2},
            "<GENERATE_TAGS>": {"max_new_tokens": 192, "num_beams": 1},
        }

    @property
    def device(self) -> str:
        return self.runtime["device"]

    @property
    def use_amp(self) -> bool:
        return self.runtime["use_amp"]

    @property
    def amp_dtype(self) -> torch.dtype:
        return self.runtime["amp_dtype"]

    def log(self, message: str) -> None:
        emit_log(self.log_callback, message)

    def prog(self, value: float) -> None:
        emit_progress(self.progress_callback, value)

    def set_callbacks(
        self,
        progress_callback: Callable[[float], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def get_status_text(self, precision_choice: str = "auto") -> str:
        return detect_runtime(precision_choice)["status_text"]

    def has_loaded_models(self) -> bool:
        return any(
            [
                self.ai_model,
                self.sdxl_model,
                self.siglip_model,
                self.dino_model,
                self.dino_adapter,
                self.wd_tagger,
                self.florence_model,
            ]
        )

    def _release_attribute(self, name: str) -> None:
        current = getattr(self, name, None)
        if current is None:
            return
        try:
            del current
        except Exception as exc:
            logger.debug("Failed to delete attribute %s cleanly", name, exc_info=exc)
            self.log(f"Warning: failed to release {name}: {exc}")
        setattr(self, name, None)

    def _empty_cuda_cache(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all_models(self) -> None:
        self.ai_model = self.ai_proc = None
        self.sdxl_model = self.sdxl_proc = None
        self.siglip_model = self.siglip_proc = None
        self.dino_model = self.dino_proc = None
        self.dino_adapter = None
        self.wd_tagger = None
        self.wd_tags = None
        self.wd_tag_categories = None
        self.florence_model = None
        self.florence_processor = None
        gc.collect()
        self._empty_cuda_cache()

    def unload_tagger(self) -> None:
        if self.wd_tagger is not None:
            self._release_attribute("wd_tagger")
        self.wd_tagger = None
        self.wd_tags = None
        self.wd_tag_categories = None
        gc.collect()
        self._empty_cuda_cache()

    def offload_to_cpu(self) -> None:
        moved: list[str] = []
        for name in ("ai_model", "sdxl_model", "siglip_model", "dino_model", "dino_adapter", "florence_model"):
            model = getattr(self, name, None)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
                moved.append(name)
        if moved:
            gc.collect()
            self._empty_cuda_cache()
            self.log(f"  -> Offloaded to CPU: {', '.join(moved)}")

    def offload_non_florence_to_cpu(self) -> None:
        moved: list[str] = []
        for name in ("ai_model", "sdxl_model", "siglip_model", "dino_model", "dino_adapter"):
            model = getattr(self, name, None)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
                moved.append(name)
        if moved:
            gc.collect()
            self._empty_cuda_cache()
            if not self._logged_non_florence_offload:
                self.log(f"  -> Offloaded non-Florence models to CPU: {', '.join(moved)}")
                self._logged_non_florence_offload = True

    def reload_from_cpu(self) -> None:
        moved: list[str] = []
        for name in ("ai_model", "sdxl_model", "siglip_model", "dino_model", "dino_adapter", "florence_model"):
            model = getattr(self, name, None)
            if model is not None and hasattr(model, "to"):
                model.to(self.device)
                moved.append(name)
        if moved:
            self.log(f"  -> Reloaded to {self.device}: {', '.join(moved)}")

    def cleanup(self) -> None:
        self._empty_cuda_cache()
        gc.collect()

    def cuda_memory_text(self) -> str:
        if self.device != "cuda" or not torch.cuda.is_available():
            return "CPU"
        try:
            reserved = int(torch.cuda.memory_reserved())
            allocated = int(torch.cuda.memory_allocated())
            total = int(self.runtime.get("total_vram") or torch.cuda.get_device_properties(0).total_memory)
            return (
                f"allocated={format_size(allocated, 1)}, "
                f"reserved={format_size(reserved, 1)}, total={format_size(total, 1)}"
            )
        except Exception:
            return "CUDA"

    def gpu_status_text(self) -> str:
        if self.device != "cuda" or not torch.cuda.is_available():
            return ""
        now = time.time()
        if now - self._last_gpu_status_at < 0.8:
            return self._last_gpu_status_text
        self._last_gpu_status_at = now
        parts: list[str] = []
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=0.8,
            ).strip()
            first_line = output.splitlines()[0] if output else ""
            values = [value.strip() for value in first_line.split(",")]
            if len(values) >= 3:
                util = int(float(values[0]))
                used = float(values[1]) * 1024**2
                total = float(values[2]) * 1024**2
                parts.append(f"GPU {util}%")
                parts.append(f"VRAM {format_size(used, 1)}/{format_size(total, 1)}")
        except Exception:
            pass
        try:
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            if reserved > 0 or allocated > 0:
                parts.append(f"torch {format_size(allocated, 1)}/{format_size(reserved, 1)}")
        except Exception:
            pass
        self._last_gpu_status_text = " | ".join(parts)
        return self._last_gpu_status_text

    def resolve_batch_size(self, kind: str, requested: int, *, minimum: int = 1) -> int:
        """Pick a conservative batch for the active VRAM profile."""
        requested = max(minimum, int(requested or minimum))
        if not self.auto_vram_batch or self.device != "cuda":
            return requested
        total = int(self.runtime.get("total_vram") or 0)
        limit = self.vram_limit_bytes or int(total * 0.90)
        if total <= 0 or limit <= 0:
            return requested

        gib = limit / 1024**3
        caps = {
            "main": 8 if gib <= 8.5 else 12 if gib <= 12.5 else 16,
            "siglip": 8 if gib <= 8.5 else 12 if gib <= 12.5 else 16,
            "dino": 4 if gib <= 8.5 else 8 if gib <= 12.5 else 16,
            "tagger": 16 if gib <= 8.5 else 24 if gib <= 12.5 else 32,
            "florence": 2 if gib <= 8.5 else 3 if gib <= 12.5 else 4,
        }
        capped = max(minimum, min(requested, caps.get(kind, requested)))
        key = (kind, requested, capped)
        if capped < requested and key not in self._logged_batch_suggestions:
            self.log(
                f"  -> Auto VRAM batch: {kind} {requested} -> {capped} "
                f"(limit {format_size(limit, 1)})"
            )
            self._logged_batch_suggestions.add(key)
        return capped

    def _can_use_torch_compile(self) -> tuple[bool, str]:
        if not hasattr(torch, "compile"):
            return False, "torch.compile is not available in this PyTorch build"
        if self.device != "cuda":
            return True, ""
        try:
            import triton  # noqa: F401
        except Exception:
            return False, "Triton is not installed or not working for CUDA backend"
        return True, ""

    def _maybe_compile_model(self, model: Any, label: str, mode: str | None = None) -> Any:
        if not self.optimize_models or not hasattr(torch, "compile"):
            return model
        can_compile, reason = self._can_use_torch_compile()
        if not can_compile:
            if reason != self._compile_block_reason:
                self.log(f"    Skipping torch.compile for {label}: {reason}")
                self._compile_block_reason = reason
            return model
        try:
            kwargs = {"mode": mode} if mode else {}
            return torch.compile(model, **kwargs)
        except Exception as exc:
            self.log(f"    Compile failed for {label}, falling back to eager mode: {exc}")
            return model

    def update_runtime(self, config: dict[str, Any]) -> None:
        precision_choice = config.get("precision", "auto")
        next_runtime = detect_runtime(precision_choice)
        runtime_changed = (
            self.runtime["device"] != next_runtime["device"]
            or self.runtime["amp_dtype"] != next_runtime["amp_dtype"]
            or self.runtime["use_amp"] != next_runtime["use_amp"]
        )
        optimize_models = bool(config.get("optimize_models", False))
        if optimize_models:
            can_compile, reason = self._can_use_torch_compile()
            if not can_compile:
                self.log(f"torch.compile disabled automatically: {reason}")
                optimize_models = False
            else:
                try:
                    import torch._dynamo as dynamo

                    dynamo.config.suppress_errors = True
                except Exception as exc:
                    self.log(f"Warning: unable to enable torch._dynamo suppress_errors: {exc}")
        tagger_engine = config.get("tagger_engine_var", "Camie-Tagger-v2")
        use_dino_adapter_flag = bool(config.get("use_dino_adapter", True))
        group_sequences = bool(config.get("group_sequences", True))

        if runtime_changed and self.has_loaded_models():
            self.log("Precision changed - models unloaded, will reload on next run.")
            self.unload_all_models()

        if tagger_engine != self.tagger_engine and self.wd_tagger is not None:
            self.unload_tagger()
            self.log(f"Tagger engine changed to {tagger_engine}. Old model unloaded.")

        if (
            (use_dino_adapter_flag != self.use_dino_adapter_flag or group_sequences != self.group_sequences)
            and (self.dino_model is not None or self.dino_adapter is not None)
        ):
            self.dino_model = self.dino_proc = None
            self.dino_adapter = None
            gc.collect()
            self._empty_cuda_cache()
            self.log("DINO adapter settings changed - DINOv2 will reload on next run.")

        if optimize_models != self.optimize_models and self.has_loaded_models():
            self.log("torch.compile setting changed - models unloaded, will reload on next run.")
            self.unload_all_models()

        self.runtime = next_runtime
        self.precision_choice = precision_choice
        self.optimize_models = optimize_models
        self.tagger_engine = tagger_engine
        self.use_dino_adapter_flag = use_dino_adapter_flag
        self.group_sequences = group_sequences
        self.camie_threshold = float(config.get("camie_threshold", 0.05))
        self.florence_mode = config.get("florence_mode", "<DETAILED_CAPTION>")
        self.florence_char = config.get("florence_char", "")
        self.florence_profile = str(config.get("florence_profile", "Balanced") or "Balanced")
        self.florence_batch_size = max(1, int(config.get("florence_batch_size", 4)))
        self.florence_max_side = max(256, int(config.get("florence_max_side", 768)))
        self.florence_max_new_tokens = max(
            0,
            int(config.get("florence_max_new_tokens", config.get("meta_max_tokens", 0))),
        )
        self.auto_vram_batch = bool(config.get("auto_vram_batch", True))
        self.unload_inactive_models = bool(config.get("unload_inactive_models", True))
        self.vram_profile = str(config.get("vram_profile", "Auto") or "Auto")
        self.vram_limit_gb = float(config.get("vram_limit_gb", 0.0) or 0.0)
        self.vram_limit_bytes = resolve_vram_limit_bytes(
            self.vram_profile,
            self.vram_limit_gb,
            int(self.runtime.get("total_vram") or 0),
        )

    def load_for_config(
        self,
        config: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> bool:
        self.set_callbacks(progress_callback, log_callback)
        self.update_runtime(config)

        dtype_map = {torch.bfloat16: "BFloat16", torch.float16: "FP16", torch.float32: "FP32"}
        self.log(f"Loading models ({dtype_map.get(self.amp_dtype, 'FP32')})...")
        if self.device == "cuda" and self.vram_limit_bytes:
            self.log(
                "VRAM guard: "
                f"auto_batch={'on' if self.auto_vram_batch else 'off'}, "
                f"unload_inactive={'on' if self.unload_inactive_models else 'off'}, "
                f"limit={format_size(self.vram_limit_bytes, 1)}"
            )

        need_siglip = bool(config.get("sort_content") or config.get("sort_style") or config.get("sort_grouping"))
        need_dino = bool(
            (config.get("sort_grouping") and not str(config.get("group_desc_var", "")).strip())
            or config.get("sort_dedup")
        )
        need_tagger = bool(
            (config.get("sort_grouping") and not str(config.get("group_desc_var", "")).strip())
            or config.get("gen_metadata")
            or config.get("sort_tagger_filter")
            or (config.get("character_aware_recursive") and config.get("recursive_scan"))
        )

        queue: list[str] = []
        if config.get("sort_ai_human") and self.ai_model is None:
            queue.append("ai")
        if config.get("sort_ai_human") and config.get("use_sdxl_vote") and self.sdxl_model is None:
            queue.append("sdxl")
        if need_siglip and self.siglip_model is None:
            queue.append("siglip")
        if need_dino and self.dino_model is None:
            queue.append("dino")
        if need_tagger and self.wd_tagger is None:
            queue.append("tagger")
        if config.get("use_florence") and self.florence_model is None:
            queue.append("florence")

        if not queue:
            self.log("All models already in memory.")
            return True

        step = 100.0 / max(len(queue), 1)
        try:
            for index, key in enumerate(queue):
                bar = LoadingBar(self.prog, index * step, (index + 1) * step, MODEL_SIZES_GB[key] / 0.8)
                bar.start()
                try:
                    if key == "ai":
                        self.log(f"  [{index + 1}/{len(queue)}] AI-vs-Human...")
                        from transformers import AutoImageProcessor, SiglipForImageClassification

                        self.ai_proc = AutoImageProcessor.from_pretrained(
                            str(self.paths.ai_human_model_path),
                            local_files_only=True,
                        )
                        model = SiglipForImageClassification.from_pretrained(
                            str(self.paths.ai_human_model_path),
                            local_files_only=True,
                        )
                        if self.use_amp:
                            model = model.to(dtype=self.amp_dtype)
                        model = model.to(self.device).eval()
                        model = self._maybe_compile_model(model, "AI-vs-Human")
                        self.ai_model = model

                    elif key == "sdxl":
                        self.log(f"  [{index + 1}/{len(queue)}] SDXL-Detector...")
                        from transformers import AutoImageProcessor, SwinForImageClassification

                        self.sdxl_proc = AutoImageProcessor.from_pretrained(
                            str(self.paths.sdxl_detector_path),
                            local_files_only=True,
                        )
                        model = SwinForImageClassification.from_pretrained(
                            str(self.paths.sdxl_detector_path),
                            local_files_only=True,
                        )
                        if self.use_amp:
                            model = model.to(dtype=self.amp_dtype)
                        model = model.to(self.device).eval()
                        model = self._maybe_compile_model(model, "SDXL-Detector")
                        self.sdxl_model = model

                    elif key == "siglip":
                        self.log(f"  [{index + 1}/{len(queue)}] SigLIP2 so400m...")
                        from transformers import AutoModel, AutoProcessor
                        import transformers.utils.logging

                        transformers.utils.logging.set_verbosity_error()
                        self.siglip_proc = AutoProcessor.from_pretrained(
                            str(self.paths.siglip2_model_path),
                            local_files_only=True,
                        )
                        load_dtype = self.amp_dtype if self.use_amp else torch.float32
                        model = AutoModel.from_pretrained(
                            str(self.paths.siglip2_model_path),
                            local_files_only=True,
                            torch_dtype=load_dtype,
                            low_cpu_mem_usage=True,
                        )
                        model = model.to(self.device).eval()
                        if self.optimize_models:
                            self.log("    Compiling SigLIP2 for maximum speed...")
                            self.log("    NOTE: This can take a few minutes on Windows.")
                        model = self._maybe_compile_model(model, "SigLIP2", mode="default")
                        self.siglip_model = model
                        self.log(f"    Model class: {type(model).__name__}")

                    elif key == "dino":
                        self.log(f"  [{index + 1}/{len(queue)}] DINOv2 base...")
                        from transformers import AutoImageProcessor, AutoModel

                        self.dino_proc = AutoImageProcessor.from_pretrained(
                            str(self.paths.dinov2_model_path),
                            local_files_only=True,
                        )
                        model = AutoModel.from_pretrained(
                            str(self.paths.dinov2_model_path),
                            local_files_only=True,
                        )
                        if self.use_amp:
                            model = model.to(dtype=self.amp_dtype)
                        model = model.to(self.device).eval()
                        model = self._maybe_compile_model(model, "DINOv2")
                        self.dino_model = model
                        self.dino_adapter = None
                        if self.use_dino_adapter_flag:
                            self._load_dino_adapter()

                    elif key == "tagger":
                        self.log(f"  [{index + 1}/{len(queue)}] {self.tagger_engine}...")
                        if not self._load_wd_tagger():
                            return False

                    elif key == "florence":
                        self.log(f"  [{index + 1}/{len(queue)}] Florence-2 PromptGen...")
                        from transformers import AutoModelForCausalLM, AutoProcessor

                        self.florence_processor = AutoProcessor.from_pretrained(
                            str(self.paths.florence_model_path),
                            local_files_only=True,
                            trust_remote_code=True,
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            str(self.paths.florence_model_path),
                            local_files_only=True,
                            trust_remote_code=True,
                        )
                        if self.use_amp:
                            model = model.to(dtype=self.amp_dtype)
                        self.florence_model = model.to(self.device).eval()
                finally:
                    bar.complete()
            self.log("All models loaded.")
            return True
        except Exception as exc:
            self.log(f"Model loading failed: {exc}")
            self.log(traceback.format_exc())
            return False

    def _load_dino_adapter(self) -> None:
        adapter_path = self.paths.manga_adapter_path if self.group_sequences else self.paths.adapter_path
        adapter_name = "Manga Adapter" if self.group_sequences else "Standard Adapter"
        if not adapter_path.is_file():
            self.log(f"    Warning: {adapter_name} not found, using raw DINOv2")
            return
        try:
            checkpoint = torch.load(str(adapter_path), map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("adapter", checkpoint)
            first_layer = state_dict.get("net.0.weight")
            hidden_dim = first_layer.shape[0] if first_layer is not None else 512
            adapter_input_dim = first_layer.shape[1] if first_layer is not None else 768
            dino_out_dim = 1536
            if adapter_input_dim != dino_out_dim:
                self.log(
                    f"    Warning: {adapter_name} expects {adapter_input_dim}d input but DINOv2 outputs {dino_out_dim}d"
                )
                return
            adapter = DINOv2Adapter(input_dim=adapter_input_dim, hidden_dim=hidden_dim)
            adapter.load_state_dict(state_dict)
            if self.use_amp:
                adapter = adapter.to(dtype=self.amp_dtype)
            self.dino_adapter = adapter.to(self.device).eval()
            self.log(f"    DINOv2 {adapter_name} loaded.")
        except Exception as exc:
            self.log(f"    Warning: {adapter_name} load failed: {exc}")
            self.log(traceback.format_exc())
            self.dino_adapter = None

    def _amp_context(self) -> dict[str, Any]:
        return {
            "device_type": self.device,
            "dtype": self.amp_dtype,
            "enabled": self.use_amp and self.device != "cpu",
        }

    def ensure_siglip_loaded(self) -> bool:
        if self.siglip_model is not None and self.siglip_proc is not None:
            return True
        config = {
            "sort_content": True,
            "precision": self.precision_choice,
            "optimize_models": self.optimize_models,
            "tagger_engine_var": self.tagger_engine,
            "use_dino_adapter": self.use_dino_adapter_flag,
            "group_sequences": self.group_sequences,
            "camie_threshold": self.camie_threshold,
            "florence_mode": self.florence_mode,
            "florence_char": self.florence_char,
            "florence_profile": self.florence_profile,
            "florence_batch_size": self.florence_batch_size,
            "florence_max_side": self.florence_max_side,
            "meta_max_tokens": self.florence_max_new_tokens,
            "auto_vram_batch": self.auto_vram_batch,
            "unload_inactive_models": self.unload_inactive_models,
            "vram_profile": self.vram_profile,
            "vram_limit_gb": self.vram_limit_gb,
        }
        return self.load_for_config(config, self.progress_callback, self.log_callback)

    def _get_florence_generation_kwargs(self, mode: str | None = None) -> dict[str, int]:
        target_mode = mode or self.florence_mode
        base = dict(
            self._florence_generation_defaults.get(
                target_mode,
                {"max_new_tokens": 256, "num_beams": 2},
            )
        )
        profile = self.florence_profile.strip().lower()
        if profile == "fast":
            base["max_new_tokens"] = max(96, int(base["max_new_tokens"] * 0.65))
            base["num_beams"] = 1
        elif profile == "quality":
            base["max_new_tokens"] = min(768, int(base["max_new_tokens"] * 1.25))
            base["num_beams"] = min(4, max(2, int(base["num_beams"]) + 1))
        else:
            base["num_beams"] = max(1, int(base["num_beams"]))
            base["max_new_tokens"] = max(64, int(base["max_new_tokens"]))
        if self.florence_max_new_tokens > 0:
            # Soft cap: keep a small grace window so the model can finish
            # the current phrase/tag instead of cutting it abruptly.
            grace_tokens = max(4, min(24, self.florence_max_new_tokens // 8))
            soft_cap = self.florence_max_new_tokens + grace_tokens
            base["max_new_tokens"] = min(base["max_new_tokens"], soft_cap)
        return base

    def _build_florence_prompt(self, prepended_tags: str = "") -> str:
        prompt = self.florence_mode
        if self.florence_char.strip() or prepended_tags:
            context = f"{self.florence_char.strip()}, {prepended_tags}".strip(", ")
            prompt = f"{self.florence_mode}{context}"
        return prompt

    def siglip_embed_images_batch(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0, device=self.device)
        if self.siglip_model is None or self.siglip_proc is None:
            raise RuntimeError("SigLIP2 is not loaded.")
        inputs = self.siglip_proc(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.amp.autocast(**self._amp_context()):
            embeddings = self.siglip_model.get_image_features(**inputs)
        if hasattr(embeddings, "pooler_output") and embeddings.pooler_output is not None:
            embeddings = embeddings.pooler_output
        elif hasattr(embeddings, "image_embeds") and embeddings.image_embeds is not None:
            embeddings = embeddings.image_embeds
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = embeddings[1] if len(embeddings) > 1 else embeddings[0]
        return F.normalize(embeddings.float(), dim=-1)

    def siglip_embed_texts(self, texts: list[str]) -> torch.Tensor:
        if self.siglip_model is None or self.siglip_proc is None:
            raise RuntimeError("SigLIP2 is not loaded.")
        inputs = self.siglip_proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.amp.autocast(**self._amp_context()):
            embeddings = self.siglip_model.get_text_features(**inputs)
        if hasattr(embeddings, "pooler_output") and embeddings.pooler_output is not None:
            embeddings = embeddings.pooler_output
        elif hasattr(embeddings, "text_embeds") and embeddings.text_embeds is not None:
            embeddings = embeddings.text_embeds
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = embeddings[1] if len(embeddings) > 1 else embeddings[0]
        return F.normalize(embeddings.float(), dim=-1)

    def get_tag_embeddings(self, tags: list[Any], is_anchor: bool = False) -> torch.Tensor:
        if not tags:
            return torch.empty(0, device=self.device)
        all_prompts: list[str] = []
        prompt_counts: list[int] = []
        for tag in tags:
            tag_text = tag[0] if isinstance(tag, tuple) else tag
            prompts = [tag_text] if is_anchor else make_ensembles(tag_text)
            all_prompts.extend(prompts)
            prompt_counts.append(len(prompts))
        all_embeddings = self.siglip_embed_texts(all_prompts)
        final_embeddings: list[torch.Tensor] = []
        offset = 0
        for count in prompt_counts:
            tag_slice = all_embeddings[offset : offset + count]
            averaged = tag_slice.mean(dim=0, keepdim=True)
            final_embeddings.append(F.normalize(averaged, dim=-1))
            offset += count
        return torch.cat(final_embeddings, dim=0)

    def siglip_filter(
        self,
        img_embeds: list[torch.Tensor],
        pos_tags_raw: list[tuple[str, float]],
        pos_embeds: torch.Tensor,
        neg_tags_raw: list[tuple[str, float]],
        neg_embeds: torch.Tensor,
        anc_embed: torch.Tensor,
        min_conf: float,
    ) -> list[tuple[bool, str, float]]:
        results: list[tuple[bool, str, float]] = []
        pos_tags = [tag for tag, _weight in pos_tags_raw]
        pos_weights = torch.tensor([weight for _tag, weight in pos_tags_raw], device=self.device)
        neg_tags = [tag for tag, _weight in neg_tags_raw]
        neg_weights = torch.tensor([weight for _tag, weight in neg_tags_raw], device=self.device)

        all_pos = torch.cat([pos_embeds, anc_embed], dim=0) if pos_embeds.numel() > 0 else None
        all_neg = torch.cat([neg_embeds, anc_embed], dim=0) if neg_embeds.numel() > 0 else None

        for img_emb in img_embeds:
            passed = True
            best_tag = ""
            best_margin = 0.0

            if all_pos is not None:
                sims = (img_emb @ all_pos.T).squeeze(0)
                margins = (sims[:-1] - sims[-1]) * pos_weights
                margins_cpu = margins.cpu()
                passed_idx = (margins_cpu >= min_conf).nonzero(as_tuple=True)[0].tolist()
                if passed_idx:
                    passed_idx.sort(key=lambda item: margins_cpu[item].item(), reverse=True)
                    best_tag = "_".join(pos_tags[item] for item in passed_idx)
                    best_margin = margins_cpu[passed_idx[0]].item()
                else:
                    passed = False
                    best_margin = margins_cpu.max().item() if margins_cpu.numel() > 0 else 0.0

            if passed and all_neg is not None:
                sims = (img_emb @ all_neg.T).squeeze(0)
                margins = (sims[:-1] - sims[-1]) * neg_weights
                margins_cpu = margins.cpu()
                fail_idx = (margins_cpu > 0).nonzero(as_tuple=True)[0].tolist()
                if fail_idx:
                    passed = False
                    worst_margin = margins_cpu[fail_idx].max().item()
                    worst_idx = (margins_cpu == worst_margin).nonzero(as_tuple=True)[0][0].item()
                    best_tag = f"-{neg_tags[worst_idx]}"
                    best_margin = -worst_margin

            results.append((passed, best_tag, best_margin))
        return results

    def siglip_classify(self, img_emb: torch.Tensor, all_group_embeds: torch.Tensor) -> tuple[int, float]:
        sims = (img_emb @ all_group_embeds.T).squeeze(0).cpu()
        anchor_sim = sims[-1].item()
        user_sims = sims[:-1]
        best_index = user_sims.argmax().item()
        best_score = user_sims[best_index].item()
        if anchor_sim > best_score:
            return -1, anchor_sim
        return best_index, best_score

    def _load_wd_tagger(self) -> bool:
        try:
            import onnxruntime as ort
            import pandas as pd

            self.log(f"Loading {self.tagger_engine}...")
            bar = LoadingBar(self.prog, 87, 90, MODEL_SIZES_GB["tagger"] / 0.8)
            bar.start()
            try:
                self.unload_tagger()
                providers = ["CPUExecutionProvider"]
                if self.device == "cuda":
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        self.log("Warning: onnxruntime-gpu is not available. Tagger will run on CPU.")

                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3

                if self.tagger_engine == "Camie-Tagger-v2":
                    model_path = self.paths.camie_tagger_path / "camie-tagger-v2.onnx"
                    meta_path = self.paths.camie_tagger_path / "camie-tagger-v2-metadata.json"
                    if not model_path.exists() or not meta_path.exists():
                        self.log("Tagger files are missing. Run download_models.py first.")
                        return False

                    self.wd_tagger = ort.InferenceSession(
                        str(model_path),
                        providers=providers,
                        sess_options=sess_options,
                    )
                    with meta_path.open("r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                    idx_to_tag = metadata["dataset_info"]["tag_mapping"]["idx_to_tag"]
                    tag_to_cat = metadata["dataset_info"]["tag_mapping"]["tag_to_category"]
                    max_index = max(int(key) for key in idx_to_tag.keys())
                    self.wd_tags = [idx_to_tag.get(str(index), f"unknown_{index}") for index in range(max_index + 1)]
                    self.wd_tag_categories = tag_to_cat
                else:
                    model_path = self.paths.wd_tagger_path / "model.onnx"
                    csv_path = self.paths.wd_tagger_path / "selected_tags.csv"
                    if not model_path.exists() or not csv_path.exists():
                        self.log("WD tagger files are missing. Run download_models.py first.")
                        return False

                    self.wd_tagger = ort.InferenceSession(
                        str(model_path),
                        providers=providers,
                        sess_options=sess_options,
                    )
                    frame = pd.read_csv(csv_path)
                    self.wd_tags = frame["name"].tolist()
                    self.wd_tag_categories = dict(zip(frame["name"], frame["category"])) if "category" in frame.columns else {}

                if self.device == "cuda" and "CUDAExecutionProvider" in providers:
                    active_providers = self.wd_tagger.get_providers()
                    if "CUDAExecutionProvider" not in active_providers:
                        self.log("Warning: CUDA failed to initialize in ONNX Runtime. Tagger is running on CPU.")
                self.log(f"{self.tagger_engine} loaded.")
                return True
            finally:
                bar.complete()
        except Exception as exc:
            self.log(f"{self.tagger_engine} failed: {exc}")
            self.log(traceback.format_exc())
            return False

    def wd_tagger_infer_batch(self, images_input: list[Image.Image]) -> list[dict[str, float]]:
        if not images_input:
            return []
        if self.wd_tagger is None:
            raise RuntimeError("Tagger is not loaded.")
        try:
            import numpy as np
            import torchvision.transforms as transforms

            engine = self.tagger_engine
            transform_camie = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            processed_images = []
            for raw_image in images_input:
                image = raw_image.convert("RGB")
                if engine == "Camie-Tagger-v2":
                    width, height = image.size
                    aspect_ratio = width / height
                    if aspect_ratio > 1:
                        new_width, new_height = 512, int(512 / aspect_ratio)
                    else:
                        new_height, new_width = 512, int(512 * aspect_ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
                    pad_color = (124, 116, 104)
                    padded = Image.new("RGB", (512, 512), pad_color)
                    padded.paste(image, ((512 - new_width) // 2, (512 - new_height) // 2))
                    processed_images.append(transform_camie(padded).unsqueeze(0).numpy())
                else:
                    width, height = image.size
                    target_size = 448
                    scale = target_size / max(width, height)
                    new_width, new_height = int(width * scale), int(height * scale)
                    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
                    padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
                    padded.paste(image, ((target_size - new_width) // 2, (target_size - new_height) // 2))
                    image_np = np.array(padded, dtype=np.float32) / 255.0
                    image_np = image_np[:, :, ::-1].copy()
                    processed_images.append(np.expand_dims(image_np, axis=0).copy())

            batch_input = np.concatenate(processed_images, axis=0)
            input_name = self.wd_tagger.get_inputs()[0].name
            outputs = self.wd_tagger.run(None, {input_name: batch_input})
            if engine == "Camie-Tagger-v2":
                logits_batch = outputs[1] if len(outputs) >= 2 else outputs[0]
                preds_batch = 1.0 / (1.0 + np.exp(-logits_batch))
            else:
                preds_batch = outputs[0]

            blocked = {"no_humans", "text_focus"}
            mature_words = {
                "nude",
                "sex",
                "penis",
                "vagina",
                "nipple",
                "pussy",
                "breast",
                "cum",
                "genital",
                "anus",
                "orgasm",
                "erect",
                "pubic",
                "naked",
            }
            results: list[dict[str, float]] = []
            for batch_index in range(len(images_input)):
                predictions = preds_batch[batch_index]
                result: dict[str, float] = {}
                active_indices = np.where(predictions > 0.05)[0]
                for tag_index in active_indices:
                    tag = self.wd_tags[tag_index]
                    if tag in blocked:
                        continue
                    score = float(predictions[tag_index])
                    if engine != "Camie-Tagger-v2":
                        category = self.wd_tag_categories.get(tag, -1)
                        is_character = category == 4
                        if not is_character:
                            if any(word in tag.lower() for word in mature_words):
                                if score < 0.15:
                                    continue
                            elif score < 0.1:
                                continue
                    elif score < self.camie_threshold:
                        continue
                    result[tag] = score
                results.append(result)
            return results
        except Exception as exc:
            self.log(f"Tagger batch inference error: {exc}")
            self.log(traceback.format_exc())
            return [{} for _ in images_input]

    def wd_tagger_infer(self, image_input: Image.Image) -> dict[str, float]:
        try:
            return self.wd_tagger_infer_batch([image_input])[0]
        except Exception as exc:
            self.log(f"Tagger single-image inference error: {exc}")
            self.log(traceback.format_exc())
            return {}

    @staticmethod
    def _normalize_filename_text(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", " ", value.lower())).strip()

    @classmethod
    def filename_contains_tag(cls, image_name: str | Path, tag: str) -> bool:
        normalized_name = cls._normalize_filename_text(Path(image_name).stem if isinstance(image_name, Path) else Path(image_name).stem)
        normalized_tag = cls._normalize_filename_text(tag)
        if not normalized_name or not normalized_tag:
            return False
        return re.search(rf"(^|\s){re.escape(normalized_tag)}($|\s)", normalized_name) is not None

    @staticmethod
    def tagger_filter(
        preds: dict[str, float],
        pos_tags: list[tuple[str, float]],
        neg_tags: list[tuple[str, float]],
        threshold: float,
        image_name: str | Path | None = None,
    ) -> tuple[bool, str]:
        for tag, _weight in neg_tags:
            score = preds.get(tag, 0.0)
            if score >= threshold:
                return False, f"-{tag} ({score:.2f})"
            if image_name is not None and ModelManager.filename_contains_tag(image_name, tag):
                return False, f"filename:-{tag}"

        for tag, _weight in pos_tags:
            score = preds.get(tag, 0.0)
            if score < threshold:
                return False, f"missing {tag} ({score:.2f})"
        return True, ""

    def _florence_infer_single(self, image_input: str | Path | Image.Image, prepended_tags: str = "") -> str:
        if self.florence_model is None or self.florence_processor is None:
            return ""
        try:
            if isinstance(image_input, (str, Path)):
                with Image.open(image_input) as raw:
                    image = raw.convert("RGB")
            else:
                image = image_input.convert("RGB")

            prompt = self._build_florence_prompt(prepended_tags)
            with torch.inference_mode():
                inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt")
                inputs = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                }
                generation_kwargs = self._get_florence_generation_kwargs(self.florence_mode)
                with torch.amp.autocast(**self._amp_context()):
                    generated_ids = self.florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=generation_kwargs["max_new_tokens"],
                        num_beams=generation_kwargs["num_beams"],
                        do_sample=False,
                        use_cache=True,
                    )

            generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.florence_processor.post_process_generation(
                generated_text,
                task=self.florence_mode,
                image_size=(image.width, image.height),
            )
            if self.florence_mode in parsed:
                return str(parsed[self.florence_mode])
            return str(parsed)
        except torch.cuda.OutOfMemoryError:
            self.log("Florence-2 single-image OOM. Clearing CUDA cache and skipping this image.")
            self._empty_cuda_cache()
            return ""
        except Exception as exc:
            self.log(f"Florence-2 inference error: {exc}")
            self.log(traceback.format_exc())
            return ""

    def florence_infer_batch(
        self,
        image_inputs: Sequence[str | Path | Image.Image],
        prepended_tags_list: Sequence[str] | None = None,
    ) -> list[str]:
        if self.florence_model is None or self.florence_processor is None:
            return [""] * len(image_inputs)
        if not image_inputs:
            return []
        if prepended_tags_list is None:
            prepended_tags_list = [""] * len(image_inputs)
        if len(prepended_tags_list) != len(image_inputs):
            raise ValueError("prepended_tags_list length must match image_inputs length")

        try:
            images: list[Image.Image] = []
            prompts: list[str] = []
            for image_input, prepended_tags in zip(image_inputs, prepended_tags_list):
                if isinstance(image_input, (str, Path)):
                    with Image.open(image_input) as raw:
                        image = raw.convert("RGB")
                else:
                    image = image_input.convert("RGB")
                images.append(image)
                prompts.append(self._build_florence_prompt(prepended_tags))

            with torch.inference_mode():
                inputs = self.florence_processor(
                    text=prompts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                }
                generation_kwargs = self._get_florence_generation_kwargs(self.florence_mode)
                with torch.amp.autocast(**self._amp_context()):
                    generated_ids = self.florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=generation_kwargs["max_new_tokens"],
                        num_beams=generation_kwargs["num_beams"],
                        do_sample=False,
                        use_cache=True,
                    )

            generated_texts = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)
            captions: list[str] = []
            for index, generated_text in enumerate(generated_texts):
                parsed = self.florence_processor.post_process_generation(
                    generated_text,
                    task=self.florence_mode,
                    image_size=(images[index].width, images[index].height),
                )
                if self.florence_mode in parsed:
                    captions.append(str(parsed[self.florence_mode]))
                else:
                    captions.append(str(parsed))
            return captions
        except torch.cuda.OutOfMemoryError:
            self._empty_cuda_cache()
            if len(image_inputs) <= 1:
                self.log("Florence-2 OOM at batch=1. Skipping this image to keep the run alive.")
                return ["" for _ in image_inputs]
            split = max(1, len(image_inputs) // 2)
            self.log(f"Florence-2 OOM at batch={len(image_inputs)}; retrying as {split}+{len(image_inputs) - split}.")
            left = self.florence_infer_batch(image_inputs[:split], prepended_tags_list[:split])
            right = self.florence_infer_batch(image_inputs[split:], prepended_tags_list[split:])
            return left + right
        except Exception as exc:
            self.log(f"Florence-2 inference error: {exc}")
            self.log(traceback.format_exc())
            return [
                self._florence_infer_single(image_input, prepended_tags)
                for image_input, prepended_tags in zip(image_inputs, prepended_tags_list)
            ]

    def florence_infer(self, image_input: str | Path | Image.Image, prepended_tags: str = "") -> str:
        return self._florence_infer_single(image_input, prepended_tags)
