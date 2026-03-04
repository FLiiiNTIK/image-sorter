"""
Image Sorting Application — v2.0.2
─────────────────────────────────────────────────────────────────
WD SwinV2 Tagger v3 for cluster naming and metadata.
Sequential VRAM management: classification models unloaded
before tagger is loaded.
"""

import gc
import json
import math
import os
import shutil
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
AI_HUMAN_MODEL_PATH = os.path.join(SCRIPT_DIR, "ai-vs-human-image-detector")
SDXL_DETECTOR_PATH  = os.path.join(SCRIPT_DIR, "sdxl-detector")
SIGLIP2_MODEL_PATH  = os.path.join(SCRIPT_DIR, "siglip2-so400m-patch16-512")
DINOV2_MODEL_PATH   = os.path.join(SCRIPT_DIR, "dinov2-base")
WD_TAGGER_PATH      = os.path.join(SCRIPT_DIR, "wd-swinv2-tagger-v3")
NSFW_MODEL_PATH     = os.path.join(SCRIPT_DIR, "nsfw_image_detection")
WEIGHTS_DIR         = os.path.join(SCRIPT_DIR, "weights")
ADAPTER_PATH        = os.path.join(WEIGHTS_DIR, "best_adapter.pth")
CACHE_DIR           = os.path.join(SCRIPT_DIR, ".cache")
SCAN_CACHE_FILE     = os.path.join(CACHE_DIR, "scan_cache.json")
SETTINGS_FILE       = os.path.join(CACHE_DIR, "settings.json")

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif'}

CONTENT_NEG_ANCHOR = "abstract pattern, random noise, blank image, nothing recognizable"
STYLE_NEG_ANCHOR   = "unknown style, unrecognizable technique, random visual"

MODEL_SIZES_GB = {"ai": 0.4, "sdxl": 0.35, "siglip": 4.5, "dino": 0.35, "tagger": 0.5}


# ═════════════════════════════════════════════════════════════════════════════
class LoadingBar:
    def __init__(self, set_fn, start_pct, end_pct, est_seconds):
        self._set = set_fn
        self._s, self._e = start_pct, end_pct
        self._est = max(est_seconds, 1.0)
        self._done = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        t0 = time.time()
        while not self._done.is_set():
            frac = 1.0 - math.exp(-2.5 * (time.time() - t0) / self._est)
            self._set(self._s + frac * (self._e - self._s) * 0.95)
            time.sleep(0.08)

    def complete(self):
        self._done.set()
        self._set(self._e)


# ═════════════════════════════════════════════════════════════════════════════
class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        except Exception:
            img = None
        return p, img


def pil_collate(batch):
    paths, imgs = zip(*batch)
    return list(paths), list(imgs)


# ═════════════════════════════════════════════════════════════════════════════
#  DINOv2 Adapter MLP (mirror of train_dinov.py)
# ═════════════════════════════════════════════════════════════════════════════
class DINOv2Adapter(nn.Module):
    """MLP adapter: projects DINOv2 vector (768) → text-embedding space."""
    def __init__(self, input_dim: int = 768, output_dim: int = 384,
                 hidden_dim: int = 512, dropout: float = 0.3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Main App
# ═════════════════════════════════════════════════════════════════════════════
class ImageSorterApp:
    _PERSIST_BOOLS = [
        "sort_nsfw_fast", "sort_ai_human", "use_sdxl_vote", "sort_content", "sort_style",
        "sort_grouping", "sort_dedup", "move_files", "recursive_scan",
        "gen_metadata", "use_dino_adapter", "optimize_models",
    ]
    _PERSIST_STRINGS = [
        "source_dir", "target_dir", "content_tags", "style_tags", "group_desc_var",
        "content_neg_anchor", "style_neg_anchor", "tag_preset_var",
    ]
    _PERSIST_NUMBERS = [
        "content_min_conf", "style_min_conf",
        "group_threshold", "dedup_threshold", "batch_size_var",
        "meta_max_per_folder", "meta_max_tokens", "meta_tags_per_image",
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Image Sorter v2.0.2")
        self.root.geometry("900x850")
        self.root.minsize(800, 700)

        # Tk variables
        self.source_dir         = tk.StringVar()
        self.target_dir         = tk.StringVar()
        self.tag_preset_var     = tk.StringVar(value="None")
        self.sort_nsfw_fast     = tk.BooleanVar(value=False)
        self.sort_ai_human      = tk.BooleanVar(value=True)
        self.use_sdxl_vote      = tk.BooleanVar(value=True)
        self.sort_content       = tk.BooleanVar(value=False)
        self.content_min_conf   = tk.DoubleVar(value=0.05)
        self.content_neg_anchor = tk.StringVar(value=CONTENT_NEG_ANCHOR)
        self.sort_style         = tk.BooleanVar(value=False)
        self.style_min_conf     = tk.DoubleVar(value=0.05)
        self.style_neg_anchor   = tk.StringVar(value=STYLE_NEG_ANCHOR)
        self.sort_grouping      = tk.BooleanVar(value=False)
        self.sort_dedup         = tk.BooleanVar(value=False)
        self.move_files         = tk.BooleanVar(value=False)
        self.recursive_scan     = tk.BooleanVar(value=True)
        self.optimize_models    = tk.BooleanVar(value=False)
        self.content_tags       = tk.StringVar(value="person, animal, landscape, vehicle, food, building, game character")
        self.style_tags         = tk.StringVar(value="anime, realistic photo, sketch, oil painting, 3d render, pixel art, watercolor")
        self.group_threshold    = tk.DoubleVar(value=0.35)
        self.dedup_threshold    = tk.DoubleVar(value=0.88)
        self.batch_size_var     = tk.IntVar(value=4)
        self.group_desc_var     = tk.StringVar(value="")
        self.precision_var      = tk.StringVar(value="auto")
        self.gen_metadata       = tk.BooleanVar(value=False)
        self.use_dino_adapter   = tk.BooleanVar(value=True)
        self.meta_max_per_folder = tk.IntVar(value=0)
        self.meta_max_tokens    = tk.IntVar(value=0)
        self.meta_tags_per_image = tk.IntVar(value=30)

        self._group_desc_history = []
        self._cancel = threading.Event()
        
        # Tag Presets Storage (Name -> {"content": ..., "style": ...})
        self.tag_presets = {
            "Photography": {"content": "person, animal, nature, portrait, landscape, cityscape, vehicle", "style": "black and white, macro, polaroid, long exposure, cinematic lighting"},
            "Anime/Art": {"content": "1girl, 1boy, solo, multiple girls, animal ears, building, mecha", "style": "anime, sketch, watercolor, line art, 3d render, flat color, concept art"}
        }

        self._build_ui()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._apply_precision()

        self.ai_model = self.ai_proc = None
        self.sdxl_model = self.sdxl_proc = None
        self.nsfw_model = self.nsfw_proc = None
        self.siglip_model = self.siglip_proc = None
        self.dino_model = self.dino_proc = None
        self.dino_adapter = None
        self.wd_tagger = None
        self.wd_tags = None
        self.wd_tag_categories = None  # tag_name → category_id

    # ═══════════════════════════════════════════════════════════════════════
    #  Precision
    # ═══════════════════════════════════════════════════════════════════════
    def _apply_precision(self):
        choice = self.precision_var.get()
        if self.device == "cpu":
            self.use_amp = False
            self.amp_dtype = torch.float32
        elif choice == "fp32":
            self.use_amp = False
            self.amp_dtype = torch.float32
        elif choice == "fp16":
            self.use_amp = True
            self.amp_dtype = torch.float16
        elif choice == "bf16":
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
        else:  # auto
            self.use_amp = True
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._update_status_label()

    def _update_status_label(self):
        dn = {torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"}.get(self.amp_dtype, "FP32")
        self.status.config(text=f"Ready — {'CUDA' if self.device == 'cuda' else 'CPU'} ({dn})")

    def _on_precision_change(self, *_):
        had_models = any([self.ai_model, self.sdxl_model, self.nsfw_model, self.siglip_model,
                          self.dino_model, self.dino_adapter, self.wd_tagger])
        self._unload_all_models()
        self._apply_precision()
        if had_models:
            self.log("Precision changed — models unloaded, will reload on next run.")

    def _unload_all_models(self):
        """Release ALL models from memory (used on precision change)."""
        self.ai_model = self.ai_proc = None
        self.sdxl_model = self.sdxl_proc = None
        self.nsfw_model = self.nsfw_proc = None
        self.siglip_model = self.siglip_proc = None
        self.dino_model = self.dino_proc = None
        self.dino_adapter = None
        self.wd_tagger = None
        self.wd_tags = None
        self.wd_tag_categories = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _offload_to_cpu(self):
        """Move GPU models to CPU RAM (frees VRAM, keeps models for fast reload)."""
        moved = []
        for name in ("ai_model", "sdxl_model", "nsfw_model", "siglip_model", "dino_model", "dino_adapter"):
            m = getattr(self, name, None)
            if m is not None and hasattr(m, "to"):
                m.to("cpu")
                moved.append(name)
        if moved:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.log(f"  → Offloaded to CPU: {', '.join(moved)}")

    def _reload_from_cpu(self):
        """Move previously offloaded models back to GPU."""
        moved = []
        for name in ("ai_model", "sdxl_model", "nsfw_model", "siglip_model", "dino_model", "dino_adapter"):
            m = getattr(self, name, None)
            if m is not None and hasattr(m, "to"):
                m.to(self.device)
                moved.append(name)
        if moved:
            self.log(f"  → Reloaded to {self.device}: {', '.join(moved)}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Settings persistence
    # ═══════════════════════════════════════════════════════════════════════
    def _save_settings(self):
        d = {}
        for k in self._PERSIST_BOOLS:
            d[k] = getattr(self, k).get()
        for k in self._PERSIST_STRINGS:
            d[k] = getattr(self, k).get()
        for k in self._PERSIST_NUMBERS:
            d[k] = getattr(self, k).get()
        d["precision"] = self.precision_var.get()
        d["group_desc_history"] = self._group_desc_history[:20]
        d["tag_presets"] = self.tag_presets
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        try:
            if not os.path.exists(SETTINGS_FILE):
                return
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                d = json.load(f)
            for k in self._PERSIST_BOOLS:
                if k in d: getattr(self, k).set(d[k])
            for k in self._PERSIST_STRINGS:
                if k in d: getattr(self, k).set(d[k])
            for k in self._PERSIST_NUMBERS:
                if k in d: getattr(self, k).set(d[k])
            if "precision" in d:
                self.precision_var.set(d["precision"])
            if "group_desc_history" in d:
                self._group_desc_history = d["group_desc_history"]
            if "tag_presets" in d:
                self.tag_presets = d["tag_presets"]
                self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.root.destroy()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        m = ttk.Frame(self.root, padding=10)
        m.pack(fill=tk.BOTH, expand=True)

        # Directories
        d = ttk.LabelFrame(m, text="Directories", padding=8)
        d.pack(fill=tk.X, pady=(0, 4)); d.columnconfigure(1, weight=1)
        ttk.Label(d, text="Source:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(d, textvariable=self.source_dir).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(d, text="Browse…", command=lambda: self._browse(self.source_dir)).grid(row=0, column=2)
        ttk.Label(d, text="Target:").grid(row=1, column=0, sticky=tk.W, pady=(3, 0))
        ttk.Entry(d, textvariable=self.target_dir).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=(3, 0))
        ttk.Button(d, text="Browse…", command=lambda: self._browse(self.target_dir)).grid(row=1, column=2, pady=(3, 0))

        ctrl = ttk.Frame(d); ctrl.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(3, 0))
        ttk.Checkbutton(ctrl, text="Recursive", variable=self.recursive_scan).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Clear cache", command=self._clear_cache).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(ctrl, text="🔍 AI Search", command=self._open_ai_search).pack(side=tk.LEFT, padx=(6, 0))
        self.cache_lbl = ttk.Label(ctrl, text="", foreground="gray")
        self.cache_lbl.pack(side=tk.LEFT, padx=4)
        self.cancel_btn = ttk.Button(ctrl, text="Cancel", command=self._do_cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self.start_btn = ttk.Button(ctrl, text="▶ Start", command=self._start)
        self.start_btn.pack(side=tk.RIGHT)

        # Criteria
        o = ttk.LabelFrame(m, text="Criteria  (Content & Style = FILTERS)", padding=8)
        o.pack(fill=tk.X, pady=4); o.columnconfigure(2, weight=1)

        ttk.Checkbutton(o, text="AI vs Human", variable=self.sort_ai_human).grid(row=0, column=0, sticky=tk.W, pady=1)
        ttk.Checkbutton(o, text="Dual-vote (SigLIP+SDXL)", variable=self.use_sdxl_vote).grid(
            row=0, column=1, columnspan=2, sticky=tk.W, padx=8)

        # Tag Presets
        pf = ttk.Frame(o); pf.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(4, 2))
        ttk.Label(pf, text="Tag Preset:").pack(side=tk.LEFT)
        self.tag_preset_cb = ttk.Combobox(pf, textvariable=self.tag_preset_var, state="readonly", width=15)
        self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
        self.tag_preset_cb.pack(side=tk.LEFT, padx=(6, 4))
        self.tag_preset_cb.bind("<<ComboboxSelected>>", self._on_preset_select)
        ttk.Button(pf, text="[+] Save", command=self._save_preset, width=8).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(pf, text="[-] Del", command=self._delete_preset, width=6).pack(side=tk.LEFT)

        ttk.Checkbutton(o, text="Content", variable=self.sort_content).grid(row=2, column=0, sticky=tk.W, pady=1)
        ttk.Label(o, text="Tags:").grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Entry(o, textvariable=self.content_tags).grid(row=2, column=2, sticky=tk.EW)

        ttk.Checkbutton(o, text="Style", variable=self.sort_style).grid(row=3, column=0, sticky=tk.W, pady=1)
        ttk.Label(o, text="Tags:").grid(row=3, column=1, sticky=tk.W, padx=5)
        ttk.Entry(o, textvariable=self.style_tags).grid(row=3, column=2, sticky=tk.EW)

        ttk.Label(o, text="  use -tag to EXCLUDE  (e.g. \"anime, -table\")",
                  foreground="#888").grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))

        ttk.Separator(o, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=4)

        ttk.Checkbutton(o, text="Visual Grouping", variable=self.sort_grouping).grid(row=6, column=0, sticky=tk.W, pady=1)
        ttk.Label(o, text="Sens:").grid(row=6, column=1, sticky=tk.W, padx=5)
        gf = ttk.Frame(o); gf.grid(row=6, column=2, sticky=tk.EW)
        ttk.Entry(gf, textvariable=self.group_threshold, width=5).pack(side=tk.LEFT)
        ttk.Label(gf, text="(0.1=few big, 0.5=medium, 0.9=many small)", foreground="gray").pack(side=tk.LEFT, padx=4)

        gd = ttk.Frame(o); gd.grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=(2, 0))
        gd.columnconfigure(1, weight=1)
        ttk.Label(gd, text="  Semantic:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(gd, textvariable=self.group_desc_var).grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(gd, text="Hist▾", command=self._show_desc_history, width=5).grid(row=0, column=2)
        ttk.Button(gd, text="✕", command=self._clear_desc_history, width=3).grid(row=0, column=3, padx=(2, 0))
        ttk.Label(o, text="  (empty = DINOv2+SigLIP2 → WD Tagger naming)",
                  foreground="gray").grid(row=8, column=0, columnspan=3, sticky=tk.W)

        ttk.Separator(o, orient="horizontal").grid(row=9, column=0, columnspan=3, sticky=tk.EW, pady=4)

        ttk.Checkbutton(o, text="Remove Duplicates", variable=self.sort_dedup).grid(row=10, column=0, sticky=tk.W, pady=1)
        ttk.Label(o, text="Thr:").grid(row=10, column=1, sticky=tk.W, padx=5)
        df2 = ttk.Frame(o); df2.grid(row=10, column=2, sticky=tk.EW)
        ttk.Entry(df2, textvariable=self.dedup_threshold, width=5).pack(side=tk.LEFT)
        ttk.Label(df2, text="(0.95=strict)", foreground="gray").pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(o, text="DINOv2 Adapter", variable=self.use_dino_adapter).grid(
            row=11, column=0, sticky=tk.W, pady=1)
        ttk.Label(o, text="  weights/best_adapter.pth → improved features",
                  foreground="gray").grid(row=11, column=1, columnspan=2, sticky=tk.W, padx=5)

        ttk.Separator(o, orient="horizontal").grid(row=12, column=0, columnspan=3, sticky=tk.EW, pady=4)

        ttk.Checkbutton(o, text="Generate Detailed Metadata (Slows process)",
                        variable=self.gen_metadata).grid(row=13, column=0, sticky=tk.W, pady=1)

        mf = ttk.Frame(o); mf.grid(row=13, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(4, 0))
        ttk.Label(mf, text="Max/folder:").pack(side=tk.LEFT)
        ttk.Spinbox(mf, from_=0, to=9999, width=5, textvariable=self.meta_max_per_folder).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(mf, text="(0=all)", foreground="gray").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(mf, text="Tags/img:").pack(side=tk.LEFT)
        ttk.Spinbox(mf, from_=5, to=100, width=5, textvariable=self.meta_tags_per_image).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(mf, text="(30)", foreground="gray").pack(side=tk.LEFT)

        # Settings
        s = ttk.LabelFrame(m, text="Settings", padding=8)
        s.pack(fill=tk.X, pady=4)
        ttk.Radiobutton(s, text="Copy", variable=self.move_files, value=False).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(s, text="Move", variable=self.move_files, value=True).pack(side=tk.LEFT)
        ttk.Separator(s, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(s, text="Batch:").pack(side=tk.LEFT)
        ttk.Spinbox(s, from_=1, to=16, width=3, textvariable=self.batch_size_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Separator(s, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(s, text="Precision:").pack(side=tk.LEFT)
        prec = ttk.Combobox(s, textvariable=self.precision_var, width=5, state="readonly",
                            values=["auto", "bf16", "fp16", "fp32"])
        prec.pack(side=tk.LEFT, padx=4)
        prec.bind("<<ComboboxSelected>>", self._on_precision_change)

        # Progress
        self.pvar = tk.DoubleVar()
        ttk.Progressbar(m, variable=self.pvar, maximum=100).pack(fill=tk.X, pady=(6, 2))
        self.status = ttk.Label(m, text="Ready")
        self.status.pack(fill=tk.X)

        # Log
        lf = ttk.Frame(m); lf.pack(fill=tk.BOTH, expand=True, pady=4)
        sb = ttk.Scrollbar(lf); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.logw = tk.Text(lf, height=10, state=tk.DISABLED, yscrollcommand=sb.set)
        self.logw.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.config(command=self.logw.yview)
        # Allow copy from log
        self.logw.bind("<Button-3>", self._log_context_menu)
        self.logw.bind("<Control-c>", self._copy_log_selection)
        self.logw.bind("<Control-a>", self._select_all_log)

    # ═══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════════════
    def _on_preset_select(self, event=None):
        name = self.tag_preset_var.get()
        if name in self.tag_presets:
            p = self.tag_presets[name]
            self.content_tags.set(p.get("content", ""))
            self.style_tags.set(p.get("style", ""))

    def _save_preset(self):
        import tkinter.simpledialog as sd
        name = sd.askstring("Save Preset", "Enter preset name:", parent=self.root)
        if name:
            self.tag_presets[name] = {
                "content": self.content_tags.get(),
                "style": self.style_tags.get()
            }
            self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
            self.tag_preset_var.set(name)
            self._save_settings()
            self.log(f"Preset '{name}' saved.")

    def _delete_preset(self):
        name = self.tag_preset_var.get()
        if name in self.tag_presets:
            if messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
                del self.tag_presets[name]
                self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
                self.tag_preset_var.set("None")
                self._save_settings()
                self.log(f"Preset '{name}' deleted.")

    def _open_ai_search(self):
        target = self.target_dir.get()
        if not target or not os.path.exists(target):
            messagebox.showerror("Error", "Please set a valid Target directory first.")
            return
        
        feat_cache_file = os.path.join(CACHE_DIR, "feat_cache.pt")
        if not os.path.exists(feat_cache_file):
            messagebox.showinfo("AI Search", f"No feature cache found in Target directory:\n{feat_cache_file}\n\nPlease run a scan with caching enabled first.")
            return
        
        try:
            feat_cache = torch.load(feat_cache_file, map_location="cpu", weights_only=True)
            # Filter only SigLIP embeddings
            siglip_cache = {k: v for k, v in feat_cache.items() if k.startswith("siglip_")}
            if not siglip_cache:
                messagebox.showinfo("AI Search", "No SigLIP embeddings found in cache. Ensure Content/Style filters were used.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load cache: {e}")
            return

        sw = tk.Toplevel(self.root)
        sw.title("Local AI Search")
        sw.geometry("600x400")
        sw.transient(self.root)
        sw.grab_set()

        ttk.Label(sw, text=f"Searching within {len(siglip_cache)} cached images", font=("", 10, "bold")).pack(pady=10)
        
        qf = ttk.Frame(sw)
        qf.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(qf, text="Search Query:").pack(side=tk.LEFT)
        query_var = tk.StringVar()
        ttk.Entry(qf, textvariable=query_var, width=50).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(sw, text="Example: 'a red apple on a white table'").pack(pady=2)

        nf = ttk.Frame(sw)
        nf.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(nf, text="Top Results:").pack(side=tk.LEFT)
        top_k_var = tk.IntVar(value=20)
        ttk.Spinbox(nf, from_=1, to=1000, width=5, textvariable=top_k_var).pack(side=tk.LEFT, padx=10)

        res_lbl = ttk.Label(sw, text="", foreground="blue")
        res_lbl.pack(pady=10)
        
        btn = ttk.Button(sw, text="Search & Copy", width=20)
        btn.pack(pady=10)

        def perform_search():
            query = query_var.get().strip()
            if not query: return
            btn.config(state=tk.DISABLED)
            res_lbl.config(text="Loading Model & Encoding Query...")
            sw.update()
            
            def _worker():
                try:
                    if self.siglip_model is None:
                        from transformers import AutoProcessor, AutoModel
                        self.siglip_proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL_PATH, local_files_only=True)
                        m = AutoModel.from_pretrained(SIGLIP2_MODEL_PATH, local_files_only=True)
                        if self.use_amp: m = m.to(dtype=self.amp_dtype)
                        if self.optimize_models.get() and hasattr(torch, "compile"): m = torch.compile(m)
                        self.siglip_model = m.to(self.device).eval()
                    
                    # Encode Text
                    with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                        txt_inp = self.siglip_proc(text=[query], padding="max_length", max_length=64, return_tensors="pt")
                        txt_inp = {k: v.to(self.device) for k, v in txt_inp.items()}
                        query_feat = self.siglip_model.get_text_features(**txt_inp)
                        query_feat = F.normalize(query_feat, p=2, dim=-1)[0].cpu() # (D,)
                    
                    # Compute similarities
                    paths, feats = [], []
                    for k, v in siglip_cache.items():
                        # key format: siglip_path_mtime
                        p = k.split("_", 1)[1].rsplit("_", 1)[0]
                        paths.append(p)
                        feats.append(v.view(1, -1))
                    
                    all_feats = F.normalize(torch.cat(feats, dim=0), p=2, dim=1) # (N, D)
                    sims = torch.mm(all_feats, query_feat.unsqueeze(1)).squeeze(1) # (N,)
                    
                    top_k = min(top_k_var.get(), len(paths))
                    scores, indices = torch.topk(sims, top_k)
                    
                    # Copy to results directory
                    res_dir = os.path.join(target, "AI_Search_Results", self._safe(query)[:50])
                    os.makedirs(res_dir, exist_ok=True)
                    
                    copied = 0
                    for i, idx in enumerate(indices):
                        p = paths[idx.item()]
                        score = scores[i].item()
                        if os.path.exists(p):
                            ext = os.path.splitext(p)[1]
                            base = f"{i+1:03d}_{score:.2f}{ext}"
                            shutil.copy2(p, os.path.join(res_dir, base))
                            copied += 1
                            
                    sw.after(0, lambda: res_lbl.config(text=f"Success! Copied {copied} files to:\n{res_dir}", foreground="green"))
                except Exception as ex:
                    sw.after(0, lambda: messagebox.showerror("Error", f"Search failed:\n{ex}"))
                    sw.after(0, lambda: res_lbl.config(text="An error occurred.", foreground="red"))
                finally:
                    sw.after(0, lambda: btn.config(state=tk.NORMAL))
            
            threading.Thread(target=_worker, daemon=True).start()

        btn.config(command=perform_search)

    def _browse(self, v):
        p = filedialog.askdirectory()
        if p: v.set(p)

    def log(self, msg):
        self.root.after(0, self._log_ui, msg)

    def _log_ui(self, msg):
        self.logw.config(state=tk.NORMAL)
        self.logw.insert(tk.END, msg + "\n")
        self.logw.see(tk.END)
        self.logw.config(state=tk.DISABLED)
        self.status.config(text=msg)

    def _copy_log_selection(self, event=None):
        try:
            text = self.logw.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except tk.TclError:
            pass
        return "break"

    def _select_all_log(self, event=None):
        self.logw.tag_add(tk.SEL, "1.0", tk.END)
        return "break"

    def _log_context_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy selection", command=self._copy_log_selection)
        menu.add_command(label="Select all", command=self._select_all_log)
        menu.add_separator()
        menu.add_command(label="Copy all", command=self._copy_all_log)
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_all_log(self):
        text = self.logw.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _prog(self, v):
        self.root.after(0, self.pvar.set, min(max(v, 0), 100))

    def _do_cancel(self):
        self._cancel.set(); self.log("⏹ Cancelling…")

    def _finish(self, msg):
        self.log(msg)
        self._cleanup_vram()
        self.root.after(0, self.start_btn.config, {"state": tk.NORMAL})
        self.root.after(0, self.cancel_btn.config, {"state": tk.DISABLED})

    def _cleanup_vram(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _clear_cache(self):
        try:
            if os.path.exists(SCAN_CACHE_FILE):
                os.remove(SCAN_CACHE_FILE)
            self.cache_lbl.config(text="cleared")
        except Exception:
            pass

    def _show_desc_history(self):
        if not self._group_desc_history:
            return messagebox.showinfo("History", "No history yet.")
        win = tk.Toplevel(self.root); win.title("History"); win.geometry("400x250")
        lb = tk.Listbox(win, selectmode=tk.SINGLE)
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for h in self._group_desc_history:
            lb.insert(tk.END, h)

        def pick():
            sel = lb.curselection()
            if sel: self.group_desc_var.set(self._group_desc_history[sel[0]])
            win.destroy()

        bf = ttk.Frame(win); bf.pack(pady=5)
        ttk.Button(bf, text="Use selected", command=pick).pack(side=tk.LEFT, padx=4)

        def clear_all():
            self._group_desc_history.clear()
            self._save_settings()
            win.destroy()
            messagebox.showinfo("History", "History cleared.")

        ttk.Button(bf, text="Clear all", command=clear_all).pack(side=tk.LEFT, padx=4)

    def _clear_desc_history(self):
        if self._group_desc_history:
            self._group_desc_history.clear()
            self._save_settings()
            self.log("Semantic history cleared.")

    # ═══════════════════════════════════════════════════════════════════════
    #  File scan (cached)
    # ═══════════════════════════════════════════════════════════════════════
    def _scan_images(self, source):
        try:
            mtime = str(os.path.getmtime(source))
        except OSError:
            mtime = "0"
        recursive = self.recursive_scan.get()
        cache_key = f"{source}|{recursive}|{mtime}"

        try:
            if os.path.exists(SCAN_CACHE_FILE):
                with open(SCAN_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                if cache.get("key") == cache_key:
                    paths = [p for p in cache["paths"] if os.path.isfile(p)]
                    self.root.after(0, self.cache_lbl.config, {"text": f"✓ cached ({len(paths)})"})
                    return paths
        except Exception:
            pass

        self.root.after(0, self.cache_lbl.config, {"text": "scanning…"})
        paths = []
        if recursive:
            for dp, _, fns in os.walk(source):
                for fn in fns:
                    if os.path.splitext(fn)[1].lower() in VALID_EXTS:
                        paths.append(os.path.join(dp, fn))
        else:
            for fn in os.listdir(source):
                fp = os.path.join(source, fn)
                if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in VALID_EXTS:
                    paths.append(fp)

        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(SCAN_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({"key": cache_key, "paths": paths}, f, ensure_ascii=False)
        except Exception:
            pass

        self.root.after(0, self.cache_lbl.config, {"text": f"scanned ({len(paths)})"})
        return paths

    @staticmethod
    def _parse_tags(tags_str):
        pos, neg = [], []
        for t in tags_str.split(","):
            t = t.strip()
            if not t: continue
            
            # Simple weight support: "tag:1.5"
            weight = 1.0
            if ":" in t:
                try: t, w = t.rsplit(":", 1); weight = float(w)
                except: pass
            
            if t.startswith("-"): neg.append((t[1:].strip(), weight))
            else: pos.append((t, weight))
        return pos, neg

    @staticmethod
    def _make_ensembles(tag: str):
        """Generate multiple prompts for a single tag to improve zero-shot accuracy."""
        return [f"a photo of {tag}", f"an image of {tag}", f"a picture of {tag}", tag]

    def _get_tag_embeddings(self, tags, is_anchor=False):
        """Return (N, D) embeddings. Averages ensemble prompts for each tag."""
        if not tags: return torch.empty(0, device=self.device)
        embeds = []
        for t in tags:
            # t may be a (tag, weight) tuple from _parse_tags or a plain string
            tag_text = t[0] if isinstance(t, tuple) else t
            prompts = [tag_text] if is_anchor else self._make_ensembles(tag_text)
            e = self._siglip_embed_texts(prompts) # (K, D)
            e = e.mean(dim=0, keepdim=True)       # (1, D)
            embeds.append(F.normalize(e, dim=-1))
        return torch.cat(embeds, dim=0)

    # ═══════════════════════════════════════════════════════════════════════
    #  Model loading
    # ═══════════════════════════════════════════════════════════════════════
    def _load_models(self):
        dtype_map = {torch.bfloat16: "BFloat16", torch.float16: "FP16", torch.float32: "FP32"}
        self.log(f"Loading models ({dtype_map.get(self.amp_dtype, 'FP32')})…")

        need_siglip = (self.sort_content.get() or self.sort_style.get() or
                       self.sort_grouping.get())
        need_dino = ((self.sort_grouping.get() and not self.group_desc_var.get().strip())
                     or self.sort_dedup.get())
        need_tagger = ((self.sort_grouping.get() and not self.group_desc_var.get().strip())
                       or self.gen_metadata.get())

        queue = []
        if self.sort_ai_human.get() and self.ai_model is None:
            queue.append("ai")
        if self.sort_ai_human.get() and self.use_sdxl_vote.get() and self.sdxl_model is None:
            queue.append("sdxl")
        if need_siglip and self.siglip_model is None:
            queue.append("siglip")
        if need_dino and self.dino_model is None:
            queue.append("dino")
        if need_tagger and self.wd_tagger is None:
            queue.append("tagger")

        if not queue:
            self.log("All models already in memory ✓")
            return True

        n = len(queue)
        step = 100.0 / n

        try:
            for i, key in enumerate(queue):
                bar = LoadingBar(self._prog, i * step, (i + 1) * step, MODEL_SIZES_GB[key] / 0.8)
                bar.start()

                if key == "ai":
                    self.log(f"  [{i+1}/{n}] AI-vs-Human…")
                    from transformers import AutoImageProcessor, SiglipForImageClassification
                    self.ai_proc = AutoImageProcessor.from_pretrained(AI_HUMAN_MODEL_PATH, local_files_only=True)
                    m = SiglipForImageClassification.from_pretrained(AI_HUMAN_MODEL_PATH, local_files_only=True)
                    if self.use_amp: m = m.to(dtype=self.amp_dtype)
                    m = m.to(self.device).eval()
                    if self.optimize_models.get() and hasattr(torch, "compile"): m = torch.compile(m)
                    self.ai_model = m

                elif key == "sdxl":
                    self.log(f"  [{i+1}/{n}] SDXL-Detector…")
                    from transformers import AutoImageProcessor, SwinForImageClassification
                    self.sdxl_proc = AutoImageProcessor.from_pretrained(SDXL_DETECTOR_PATH, local_files_only=True)
                    m = SwinForImageClassification.from_pretrained(SDXL_DETECTOR_PATH, local_files_only=True)
                    if self.use_amp: m = m.to(dtype=self.amp_dtype)
                    m = m.to(self.device).eval()
                    if self.optimize_models.get() and hasattr(torch, "compile"): m = torch.compile(m)
                    self.sdxl_model = m

                elif key == "siglip":
                    self.log(f"  [{i+1}/{n}] SigLIP2 so400m…")
                    from transformers import AutoProcessor, AutoModel
                    self.siglip_proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL_PATH, local_files_only=True)
                    m = AutoModel.from_pretrained(SIGLIP2_MODEL_PATH, local_files_only=True)
                    if self.use_amp: m = m.to(dtype=self.amp_dtype)
                    m = m.to(self.device).eval()
                    if self.optimize_models.get() and hasattr(torch, "compile"): m = torch.compile(m)
                    self.siglip_model = m
                    self.log(f"    Model class: {type(m).__name__}")

                elif key == "dino":
                    self.log(f"  [{i+1}/{n}] DINOv2 base…")
                    from transformers import AutoImageProcessor, AutoModel
                    self.dino_proc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_PATH, local_files_only=True)
                    m = AutoModel.from_pretrained(DINOV2_MODEL_PATH, local_files_only=True)
                    if self.use_amp: m = m.to(dtype=self.amp_dtype)
                    m = m.to(self.device).eval()
                    if self.optimize_models.get() and hasattr(torch, "compile"): m = torch.compile(m)
                    self.dino_model = m

                    # — DINOv2 adapter —
                    self.dino_adapter = None
                    if self.use_dino_adapter.get():
                        if os.path.isfile(ADAPTER_PATH):
                            try:
                                ckpt = torch.load(ADAPTER_PATH, map_location=self.device, weights_only=True)
                                sd = ckpt.get("adapter", ckpt)
                                # Auto-detect dims from first linear layer
                                w0 = sd.get("net.0.weight", None)
                                if w0 is not None:
                                    hidden_dim = w0.shape[0]
                                    adapter_input_dim = w0.shape[1]
                                else:
                                    hidden_dim = 512
                                    adapter_input_dim = 768
                                # Multi-scale DINOv2 produces 1536-dim; old adapters expect 768
                                # If mismatch, skip adapter with warning
                                dino_out_dim = 1536  # CLS(768) + spatial(768)
                                if adapter_input_dim != dino_out_dim:
                                    self.log(f"    ⚠ Adapter expects {adapter_input_dim}d input but DINOv2 outputs {dino_out_dim}d (multi-scale)")
                                    self.log(f"      → Skipping adapter. Retrain with input_dim={dino_out_dim} to use it.")
                                    self.dino_adapter = None
                                else:
                                    adapter = DINOv2Adapter(input_dim=adapter_input_dim, hidden_dim=hidden_dim)
                                    adapter.load_state_dict(sd)
                                    if self.use_amp:
                                        adapter = adapter.to(dtype=self.amp_dtype)
                                    self.dino_adapter = adapter.to(self.device).eval()
                                    self.log(f"    DINOv2 adapter loaded ✓ (in={adapter_input_dim}, hidden={hidden_dim})")
                            except Exception as ae:
                                self.log(f"    ⚠ Adapter load failed: {ae}")
                                self.dino_adapter = None
                        else:
                            self.log(f"    ⚠ Adapter not found: {ADAPTER_PATH} — using raw DINOv2")

                elif key == "tagger":
                    self.log(f"  [{i+1}/{n}] WD SwinV2 Tagger…")
                    self._load_wd_tagger()

                bar.complete()

            self.log("All models loaded ✓")
            return True
        except Exception as e:
            self.log(f"❌ {e}")
            self.log(traceback.format_exc())
            return False

    # ═══════════════════════════════════════════════════════════════════════
    #  SigLIP2 zero-shot: compute embeddings separately for robustness
    # ═══════════════════════════════════════════════════════════════════════
    def _siglip_embed_image(self, img):
        """Get normalized image embedding (1, D)."""
        inp = self.siglip_proc(images=img, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
            emb = self.siglip_model.get_image_features(**inp)
        return F.normalize(emb.float(), dim=-1)

    def _siglip_embed_texts(self, texts):
        """Get normalized text embeddings (N, D)."""
        inp = self.siglip_proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
            emb = self.siglip_model.get_text_features(**inp)
        return F.normalize(emb.float(), dim=-1)

    # ═══════════════════════════════════════════════════════════════════════
    #  Content / Style filter using separate embeddings
    # ═══════════════════════════════════════════════════════════════════════
    def _siglip_filter(self, img_embeds, pos_tags_raw, pos_embeds, neg_tags_raw, neg_embeds,
                         anc_embed, min_conf):
        """
        Filter images by zero-shot similarity with min confidence and multi-tagging.
        img_embeds: list of pre-computed (1, D) image embedding tensors.
        Returns list of (passed, best_tag, margin) per image.
        """
        results = []
        pos_tags = [t for t, w in pos_tags_raw]
        pos_weights = torch.tensor([w for t, w in pos_tags_raw], device=self.device)
        
        neg_tags = [t for t, w in neg_tags_raw]
        neg_weights = torch.tensor([w for t, w in neg_tags_raw], device=self.device)

        # Optimization: concat anchor to embeds for a single matrix multiply per image
        if pos_embeds.numel() > 0:
            all_pos = torch.cat([pos_embeds, anc_embed], dim=0)
        else:
            all_pos = None

        if neg_embeds.numel() > 0:
            all_neg = torch.cat([neg_embeds, anc_embed], dim=0)
        else:
            all_neg = None

        for img_emb in img_embeds:
            passed = True
            best_tag = ""
            best_margin = 0.0

            # Positive check
            if all_pos is not None:
                sims = (img_emb @ all_pos.T).squeeze(0)  # (N+1,)
                user_sims = sims[:-1]
                anchor_sim = sims[-1]
                margins = (user_sims - anchor_sim) * pos_weights

                passed_idx = (margins >= min_conf).nonzero(as_tuple=True)[0].tolist()
                if passed_idx:
                    # Sort tags by margin descending
                    passed_idx.sort(key=lambda i: margins[i].item(), reverse=True)
                    # Multi-tag combination
                    tags = [self._safe(pos_tags[i]) for i in passed_idx]
                    best_tag = "_".join(tags)
                    best_margin = margins[passed_idx[0]].item()
                else:
                    passed = False
                    best_margin = margins.max().item() if user_sims.numel() > 0 else 0.0

            # Negative check
            if passed and all_neg is not None:
                sims = (img_emb @ all_neg.T).squeeze(0)  # (N+1,)
                user_sims = sims[:-1]
                anchor_sim = sims[-1]
                margins = (user_sims - anchor_sim) * neg_weights
                
                # If any neg tag has a margin > 0 above anchor, fail it
                fail_idx = (margins > 0).nonzero(as_tuple=True)[0].tolist()
                if fail_idx:
                    passed = False
                    # grab the strongest negative margin
                    worst_margin = margins[fail_idx].max().item()
                    worst_idx = (margins == worst_margin).nonzero(as_tuple=True)[0][0].item()
                    best_tag = f"-{self._safe(neg_tags[worst_idx])}"
                    best_margin = -worst_margin

            results.append((passed, best_tag, best_margin))
        return results

    # ═══════════════════════════════════════════════════════════════════════
    #  Semantic grouping: assign image to best-matching description
    # ═══════════════════════════════════════════════════════════════════════
    def _siglip_classify(self, img_emb, all_group_embeds):
        """Assign image to best matching description. Returns (index, score)."""
        sims = (img_emb @ all_group_embeds.T).squeeze(0)
        user_sims = sims[:-1]  # exclude "none" anchor
        bi = user_sims.argmax().item()
        return bi, user_sims[bi].item()

    # ═══════════════════════════════════════════════════════════════════════
    #  Entry
    # ═══════════════════════════════════════════════════════════════════════
    def _start(self):
        src = self.source_dir.get().strip()
        tgt = self.target_dir.get().strip()
        if not src or not tgt:
            return messagebox.showerror("Error", "Specify both Source and Target.")
        if not os.path.isdir(src):
            return messagebox.showerror("Error", f"Source not found:\n{src}")
        if not any([self.sort_ai_human.get(), self.sort_content.get(),
                     self.sort_style.get(), self.sort_grouping.get(), self.sort_dedup.get()]):
            return messagebox.showwarning("Warning", "Select at least one criterion.")

        self._save_settings()
        self._cancel.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.pvar.set(0)
        self.logw.config(state=tk.NORMAL); self.logw.delete("1.0", tk.END); self.logw.config(state=tk.DISABLED)
        threading.Thread(target=self._run, args=(src, tgt), daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════════
    #  Pipeline
    # ═══════════════════════════════════════════════════════════════════════
    def _run(self, source, target):
        if not self._load_models():
            return self._finish("Model loading failed.")

        self.log("Scanning images…")
        self._prog(0)
        img_paths = self._scan_images(source)
        total = len(img_paths)
        if not total:
            return self._finish("No images found.")
        self.log(f"Found {total} images.  Batch: {self.batch_size_var.get()}")

        # Flags
        do_ai    = self.sort_ai_human.get()
        do_sdxl  = do_ai and self.use_sdxl_vote.get()
        do_cont  = self.sort_content.get()
        do_style = self.sort_style.get()
        do_group = self.sort_grouping.get()
        do_dedup = self.sort_dedup.get()

        group_descs = [t.strip() for t in self.group_desc_var.get().split(",") if t.strip()]
        use_sem_group = do_group and bool(group_descs)
        use_dino_group = do_group and not group_descs
        need_dino_feats = do_dedup or use_dino_group

        if use_sem_group:
            desc_str = self.group_desc_var.get().strip()
            if desc_str not in self._group_desc_history:
                self._group_desc_history.insert(0, desc_str)
                self._group_desc_history = self._group_desc_history[:20]

        cont_pos, cont_neg = self._parse_tags(self.content_tags.get()) if do_cont else ([], [])
        sty_pos, sty_neg   = self._parse_tags(self.style_tags.get())   if do_style else ([], [])

        with torch.no_grad():
            self.log("Initializing models and feature cache...")
            cont_min_conf = self.content_min_conf.get()
            sty_min_conf = self.style_min_conf.get()
            
            cont_pos_embeds = self._get_tag_embeddings(cont_pos)
            cont_neg_embeds = self._get_tag_embeddings(cont_neg)
            cont_anc_embed = self._get_tag_embeddings([self.content_neg_anchor.get()], is_anchor=True)

            sty_pos_embeds = self._get_tag_embeddings(sty_pos)
            sty_neg_embeds = self._get_tag_embeddings(sty_neg)
            sty_anc_embed = self._get_tag_embeddings([self.style_neg_anchor.get()], is_anchor=True)

            if use_sem_group:
                group_descs_embeds = self._get_tag_embeddings(group_descs)
                group_anc_embed = self._get_tag_embeddings(["none of the above, other, miscellaneous"], is_anchor=True)
                all_group_embeds = torch.cat([group_descs_embeds, group_anc_embed], dim=0)

            feat_cache_file = os.path.join(CACHE_DIR, "feat_cache.pt")
            try:
                feat_cache = torch.load(feat_cache_file, weights_only=True) if os.path.exists(feat_cache_file) else {}
            except Exception:
                feat_cache = {}
            cache_dirty = False
            
            can_stream = not need_dino_feats and not self.gen_metadata.get()

        bs = max(1, self.batch_size_var.get())
        dataset = ImagePathDataset(img_paths)
        loader = DataLoader(
            dataset, batch_size=bs, shuffle=False,
            num_workers=2, pin_memory=(self.device == "cuda"),
            collate_fn=pil_collate, persistent_workers=True
        )

        accepted = []
        dino_paths = []
        dino_feats = []
        dino_subs = []
        wd_preds_cache = {}  # path -> tag_dict
        errors = skipped = processed = 0
        PROC_END = 80

        need_tagger = ((self.sort_grouping.get() and not self.group_desc_var.get().strip())
                       or self.gen_metadata.get())

        with torch.no_grad():
            for batch_paths, batch_imgs in loader:
                if self._cancel.is_set():
                    return self._finish("⏹ Cancelled.")

                valid = [(p, im) for p, im in zip(batch_paths, batch_imgs) if im is not None]
                bad = len(batch_paths) - len(valid)
                if bad:
                    errors += bad
                    for p, im in zip(batch_paths, batch_imgs):
                        if im is None: self.log(f"⚠ Cannot open: {os.path.basename(p)}")
                if not valid:
                    processed += len(batch_paths)
                    self._prog(processed / total * PROC_END); continue

                paths_v = [v[0] for v in valid]
                imgs_v  = [v[1] for v in valid]
                n = len(imgs_v)
                ok = [True] * n
                subs = [""] * n

                # ── 1) AI / Human ────────────────────────────────
                if do_ai:
                    try:
                        inp = self.ai_proc(images=imgs_v, return_tensors="pt")
                        inp = {k: v.to(self.device) for k, v in inp.items()}
                        with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                            lg1 = self.ai_model(**inp).logits
                        p1 = torch.softmax(lg1.float(), dim=-1)
                        i1 = lg1.argmax(-1)

                        if do_sdxl and self.sdxl_model is not None:
                            inp2 = self.sdxl_proc(images=imgs_v, return_tensors="pt")
                            inp2 = {k: v.to(self.device) for k, v in inp2.items()}
                            with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                                lg2 = self.sdxl_model(**inp2).logits
                            p2 = torch.softmax(lg2.float(), dim=-1)
                            i2 = lg2.argmax(-1)
                            for j in range(n):
                                l1 = self.ai_model.config.id2label[i1[j].item()]
                                c1 = p1[j, i1[j]].item()
                                l2 = self.sdxl_model.config.id2label[i2[j].item()]
                                c2 = p2[j, i2[j]].item()
                                subs[j] = "ai" if (l1 == "ai" and l2 == "artificial") else "hum"
                                self.log(f"  {os.path.basename(paths_v[j])}: "
                                         f"SigLIP={l1}({c1:.0%}) SDXL={l2}({c2:.0%}) → {subs[j]}")
                        else:
                            for j in range(n):
                                l1 = self.ai_model.config.id2label[i1[j].item()]
                                c1 = p1[j, i1[j]].item()
                                subs[j] = l1
                                self.log(f"  {os.path.basename(paths_v[j])} → {l1} ({c1:.0%})")

                    except torch.cuda.OutOfMemoryError:
                        self.log("⚠ OOM — retrying one by one")
                        torch.cuda.empty_cache()
                        for j in range(n):
                            try:
                                inp = self.ai_proc(images=imgs_v[j], return_tensors="pt")
                                inp = {k: v.to(self.device) for k, v in inp.items()}
                                with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                                    lg = self.ai_model(**inp).logits
                                subs[j] = self.ai_model.config.id2label[lg.argmax(-1).item()]
                            except Exception as e:
                                self.log(f"⚠ {os.path.basename(paths_v[j])}: {e}")
                    except Exception as e:
                        self.log(f"⚠ AI/Human error: {e}")
                        self.log(traceback.format_exc())

                # ── Get SigLIP embeddings for caching ──
                siglip_embs = [None] * n
                if do_cont or do_style or use_sem_group or use_dino_group:
                    for j in range(n):
                        if not ok[j]: continue
                        try:
                            try:
                                mtime = str(os.path.getmtime(paths_v[j]))
                            except OSError:
                                mtime = "0"
                            ckey = f"siglip_{paths_v[j]}_{mtime}"
                            if ckey in feat_cache:
                                emb = feat_cache[ckey].to(self.device)
                            else:
                                emb = self._siglip_embed_image(imgs_v[j])
                                feat_cache[ckey] = emb.cpu()
                                cache_dirty = True
                            siglip_embs[j] = emb
                        except Exception as e:
                            self.log(f"⚠ SigLIP caching error for {os.path.basename(paths_v[j])}: {e}")
                            ok[j] = False

                # ── 2) Content FILTER ────────────────────────────
                if do_cont and (cont_pos or cont_neg):
                    alive = [j for j in range(n) if ok[j]]
                    if alive:
                        try:
                            res = self._siglip_filter(
                                [siglip_embs[j] for j in alive],
                                cont_pos, cont_pos_embeds,
                                cont_neg, cont_neg_embeds,
                                cont_anc_embed, cont_min_conf)
                            for k, j in enumerate(alive):
                                passed, tag, margin = res[k]
                                if not passed:
                                    ok[j] = False; skipped += 1
                                    self.log(f"  ✗ {os.path.basename(paths_v[j])} content: {tag} (margin={margin:.2f})")
                                else:
                                    subs[j] = os.path.join(subs[j], tag) if subs[j] else tag
                                    self.log(f"  ✓ {os.path.basename(paths_v[j])} content: {tag} (margin={margin:+.2f})")
                        except Exception as e:
                            self.log(f"⚠ Content filter ERROR: {e}")
                            self.log(traceback.format_exc())

                # ── 3) Style FILTER ──────────────────────────────
                if do_style and (sty_pos or sty_neg):
                    alive = [j for j in range(n) if ok[j]]
                    if alive:
                        try:
                            res = self._siglip_filter(
                                [siglip_embs[j] for j in alive],
                                sty_pos, sty_pos_embeds,
                                sty_neg, sty_neg_embeds,
                                sty_anc_embed, sty_min_conf)
                            for k, j in enumerate(alive):
                                passed, tag, margin = res[k]
                                if not passed:
                                    ok[j] = False; skipped += 1
                                    self.log(f"  ✗ {os.path.basename(paths_v[j])} style: {tag} (margin={margin:.2f})")
                                else:
                                    subs[j] = os.path.join(subs[j], tag) if subs[j] else tag
                                    self.log(f"  ✓ {os.path.basename(paths_v[j])} style: {tag} (margin={margin:+.2f})")
                        except Exception as e:
                            self.log(f"⚠ Style filter ERROR: {e}")
                            self.log(traceback.format_exc())

                # ── 4) Semantic grouping (SigLIP2) ──────────────
                if use_sem_group:
                    alive = [j for j in range(n) if ok[j]]
                    for j in alive:
                        try:
                            bi, score = self._siglip_classify(siglip_embs[j], all_group_embeds)
                            group_name = self._safe(group_descs[bi])
                            subs[j] = os.path.join(subs[j], group_name) if subs[j] else group_name
                            self.log(f"  ⊛ {os.path.basename(paths_v[j])} → {group_descs[bi]} ({score:.2f})")
                        except Exception as e:
                            self.log(f"  ⚠ Semantic group error for {os.path.basename(paths_v[j])}: {e}")
                            self.log(traceback.format_exc())

                # ── 5) DINOv2 (+SigLIP2) features (for dedup / visual clustering) ──
                if need_dino_feats:
                    for j in range(n):
                        if not ok[j]: continue
                        try:
                            try:
                                mtime = str(os.path.getmtime(paths_v[j]))
                            except OSError:
                                mtime = "0"
                            ckey = f"dinov2ms_{paths_v[j]}_{mtime}_{self.use_dino_adapter.get()}"
                            
                            if ckey in feat_cache:
                                dino_vec = feat_cache[ckey]
                            else:
                                inp = self.dino_proc(images=imgs_v[j], return_tensors="pt")
                                inp = {k: v.to(self.device) for k, v in inp.items()}
                                with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                                    hs = self.dino_model(**inp).last_hidden_state
                                    cls_tok = hs[:, 0, :]               # (1, 768)
                                    spatial = hs[:, 1:, :].mean(dim=1)   # (1, 768)
                                    dino_feat = torch.cat([cls_tok, spatial], dim=-1)  # (1, 1536)
                                    if self.dino_adapter is not None:
                                        dino_feat = self.dino_adapter(dino_feat)
                                dino_vec = dino_feat.float().cpu()
                                feat_cache[ckey] = dino_vec
                                cache_dirty = True

                            # ── 6) WD Tagger Semantic Features (expanded 64-dim) ──
                            sem_vec = None
                            if need_tagger and self.wd_tagger is not None:
                                try:
                                    preds = self._wd_tagger_infer(imgs_v[j])
                                    if preds:
                                        wd_preds_cache[paths_v[j]] = preds
                                        
                                        # Tag sets for dense semantic grouping
                                        nude_tags = {"nude", "unskirt", "nipples", "pussy", "penis", "breasts", "cleavage", "underwear", "bikini", "swimsuit", "naked"}
                                        p_stand_tags = {"standing", "on_one_leg"}
                                        p_sit_tags = {"sitting", "kneeling", "squatting", "seiza", "crossed_legs"}
                                        p_lie_tags = {"lying", "on_stomach", "on_back", "on_side"}
                                        c_shirt_tags = {"shirt", "t-shirt", "collared_shirt", "blouse", "tank_top", "sweater", "hoodie"}
                                        c_jacket_tags = {"jacket", "coat", "cardigan", "suit_jacket", "cloak"}
                                        c_dress_tags = {"dress", "sundress", "wedding_dress"}
                                        c_skirt_tags = {"skirt", "pleated_skirt", "miniskirt", "pencil_skirt"}
                                        c_pants_tags = {"pants", "jeans", "shorts", "sweatpants", "trousers", "leggings"}
                                        # Scene / Setting tags
                                        sc_indoor = {"indoors", "bedroom", "classroom", "office", "kitchen", "bathroom", "living_room"}
                                        sc_outdoor = {"outdoors", "sky", "cloud", "street", "road", "bridge"}
                                        sc_nature = {"nature", "forest", "mountain", "ocean", "lake", "river", "field", "tree", "flower"}
                                        sc_urban = {"city", "building", "skyscraper", "cityscape", "town", "alley"}
                                        # Hair color
                                        h_black = {"black_hair"}
                                        h_blonde = {"blonde_hair", "light_brown_hair"}
                                        h_brown = {"brown_hair"}
                                        h_red = {"red_hair", "pink_hair"}
                                        h_blue = {"blue_hair", "aqua_hair"}
                                        h_white = {"white_hair", "grey_hair", "silver_hair"}
                                        h_green = {"green_hair"}
                                        h_purple = {"purple_hair"}
                                        # Expression
                                        ex_happy = {"smile", "grin", "laughing", "open_mouth", ":d"}
                                        ex_sad = {"crying", "tears", "frown", "sad"}
                                        ex_angry = {"angry", "furrowed_brow", "clenched_teeth"}
                                        ex_neutral = {"expressionless", "closed_mouth", "serious"}
                                        
                                        def get_max_sc(tset):
                                            return max([sc for t, sc in preds.items() if t in tset] + [0.0])
                                            
                                        # Core state (2), Pose (3), Clothing (5), Scene (4), Hair (8), Expression (4) = 26 dims
                                        nude_score = get_max_sc(nude_tags)
                                        state_vec = torch.tensor([
                                            nude_score, max(0.0, 1.0 - nude_score),
                                            get_max_sc(p_stand_tags), get_max_sc(p_sit_tags), get_max_sc(p_lie_tags),
                                            get_max_sc(c_shirt_tags), get_max_sc(c_jacket_tags),
                                            get_max_sc(c_dress_tags), get_max_sc(c_skirt_tags), get_max_sc(c_pants_tags),
                                            get_max_sc(sc_indoor), get_max_sc(sc_outdoor),
                                            get_max_sc(sc_nature), get_max_sc(sc_urban),
                                            get_max_sc(h_black), get_max_sc(h_blonde),
                                            get_max_sc(h_brown), get_max_sc(h_red),
                                            get_max_sc(h_blue), get_max_sc(h_white),
                                            get_max_sc(h_green), get_max_sc(h_purple),
                                            get_max_sc(ex_happy), get_max_sc(ex_sad),
                                            get_max_sc(ex_angry), get_max_sc(ex_neutral),
                                        ], dtype=torch.float32, device='cpu')
                                        
                                        # Character Tags (Category 4) — larger hash space (38 dims), top 8
                                        char_preds = [(t, sc) for t, sc in preds.items() if self.wd_tag_categories.get(t, -1) == 4]
                                        top_chars = sorted(char_preds, key=lambda x: x[1], reverse=True)[:8]
                                        CHAR_DIM = 38
                                        char_vec = torch.zeros(CHAR_DIM, device='cpu')
                                        for t, sc in top_chars:
                                            if sc > 0.35:
                                                idx = hash(t) % CHAR_DIM
                                                char_vec[idx] = max(char_vec[idx].item(), sc)
                                                
                                        # state(26) + chars(38) = 64 dims
                                        sem_vec = torch.cat([state_vec, char_vec], dim=-1).unsqueeze(0)

                                except Exception as e:
                                    self.log(f"  ⚠ WD Tagger error on {os.path.basename(paths_v[j])}: {e}")

                            if use_dino_group:
                                sig_vec = siglip_embs[j].cpu()
                                # Per-modality L2-normalize + weight before concatenation
                                dino_n = F.normalize(dino_vec, p=2, dim=-1) * 1.0
                                sig_n  = F.normalize(sig_vec,  p=2, dim=-1) * 0.7
                                parts = [dino_n, sig_n]
                                if sem_vec is not None:
                                    sem_n = F.normalize(sem_vec, p=2, dim=-1) * 0.3
                                    parts.append(sem_n)
                                combined = F.normalize(torch.cat(parts, dim=-1), p=2, dim=-1)
                                dino_feats.append(combined)
                            else:
                                dino_feats.append(dino_vec)

                            dino_paths.append(paths_v[j])
                            dino_subs.append(subs[j])
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            self.log(f"⚠ OOM {os.path.basename(paths_v[j])}, skipping")
                            ok[j] = False
                        except Exception as e:
                            self.log(f"⚠ Feature extract {os.path.basename(paths_v[j])}: {e}")
                            ok[j] = False
                else:
                    for j in range(n):
                        if ok[j]:
                            if can_stream:
                                dst = os.path.join(target, self._safe(subs[j])) if subs[j] else target
                                self._place(paths_v[j], dst)
                            accepted.append((paths_v[j], subs[j]))

                processed += len(batch_paths)
                self._prog(processed / total * PROC_END)

        # ── Dedup ────────────────────────────────────────────────
        if do_dedup and dino_feats and not self._cancel.is_set():
            before = len(dino_feats)
            self.log(f"Deduplicating {before} images…")
            dino_paths, dino_feats, dino_subs = self._deduplicate(
                dino_paths, dino_feats, dino_subs, self.dedup_threshold.get())
            self.log(f"  Removed {before - len(dino_feats)} duplicates")
            self._prog(85)

        # ── DINOv2 clustering (only if no semantic descs) ────────
        clustered_groups = {}  # cluster_id → list of (path, sub)
        if use_dino_group and dino_feats and not self._cancel.is_set():
            self.log(f"Clustering {len(dino_feats)} images…")
            try:
                clustered_groups = self._cluster_images(
                    dino_paths, dino_feats, dino_subs,
                    self.group_threshold.get())
            except Exception as e:
                self.log(f"⚠ Clustering ERROR: {e}")
                self.log(traceback.format_exc())
                for p, s in zip(dino_paths, dino_subs):
                    accepted.append((p, s))
        elif dino_feats:
            for p, s in zip(dino_paths, dino_subs):
                accepted.append((p, s))
        self._prog(85)

        if self._cancel.is_set():
            return self._finish("⏹ Cancelled.")

        # ── Handle Uncategorized 'Other' images ──────────────────
        uncat_indices = [i for i, (p, s) in enumerate(accepted) if not s]

        if need_tagger and uncat_indices and not self._cancel.is_set():
            self.log(f"Categorizing {len(uncat_indices)} 'Other' images using cached tags…")
            
            # Tags that are too generic to be good folder names on their own
            generic_tags = {
                "monochrome", "greyscale", "no_humans", "negative_space", "simple_background",
                "white_background", "black_background", "transparent_background", "sketch",
                "lineart", "comic", "manga", "gradient_background", "pattern_background"
            }
            
            for i in uncat_indices:
                p, s = accepted[i]
                preds = wd_preds_cache.get(p)
                if preds:
                    # Filter out ratings and heavily penalize generic tags for naming
                    filtered_tags = []
                    for t, sc in preds.items():
                        if self.wd_tag_categories.get(t, -1) != 9 and sc > 0.35:
                            score = sc * (0.1 if t in generic_tags else 1.0)
                            filtered_tags.append((t, score))
                            
                    top_tags = sorted(filtered_tags, key=lambda x: x[1], reverse=True)[:2]
                    if top_tags:
                        name_parts = [t.replace('_', ' ').title() for t, _ in top_tags]
                        folder_name = self._safe(" ".join(name_parts))[:40]
                    else:
                        folder_name = "Other"
                else:
                    folder_name = "Other"
                accepted[i] = (p, folder_name)
                self.log(f"  ⊛ {os.path.basename(p)} → {folder_name}")

        # ── WD Tagger: name clusters ────────────────────────────
        if need_tagger and clustered_groups and not self._cancel.is_set():
            self.log(f"Naming {len(clustered_groups)} clusters from tags…")
            named = self._wd_tagger_name_clusters(clustered_groups, wd_preds_cache)
            accepted.extend(named)
            self._prog(92)
        
        # ── WD Tagger: detailed metadata ────────────────────────
        do_meta = need_tagger and self.gen_metadata.get() and not self._cancel.is_set()
        if do_meta:
            self.log(f"Generating rich metadata for {len(accepted)} images…")
            self._wd_tagger_generate_metadata(accepted, target, wd_preds_cache)

        if need_tagger:
            # Force full ONNX session destruction to prevent CUDA buffer contamination
            if self.wd_tagger is not None:
                try:
                    del self.wd_tagger
                except Exception:
                    pass
            self.wd_tagger = None
            self.wd_tags = None
            self.wd_tag_categories = None
            wd_preds_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cache_dirty and not self._cancel.is_set():
            try:
                self.log("Saving feature cache…")
                torch.save(feat_cache, feat_cache_file)
            except Exception as e:
                self.log(f"⚠ Could not save cache: {e}")

        self._prog(95)
        if self._cancel.is_set():
            return self._finish("⏹ Cancelled.")

        # ── Save ──────────────────────────────────────────────────
        if not can_stream:
            self.log(f"Saving {len(accepted)} images…")
            for j, (p, s) in enumerate(accepted):
                dst = os.path.join(target, self._safe(s)) if s else target
                self._place(p, dst)
                self._prog(95 + (j + 1) / max(len(accepted), 1) * 5)
        elif uncat_indices:
            self.log(f"Saving remaining {len(uncat_indices)} images…")
            for j, i in enumerate(uncat_indices):
                p, s = accepted[i]
                dst = os.path.join(target, self._safe(s)) if s else target
                self._place(p, dst)
                self._prog(95 + (j + 1) / max(len(uncat_indices), 1) * 5)
        else:
            self.log("Files were streamed successfully.")

        self._prog(100)
        self._finish(f"✅ Done!  Total: {total} | Saved: {len(accepted)} | "
                     f"Filtered: {skipped} | Errors: {errors}")

    # ═══════════════════════════════════════════════════════════════════════
    def _deduplicate(self, paths, feat_list, subs, threshold):
        feats = F.normalize(torch.cat(feat_list, dim=0), p=2, dim=1)
        n = len(paths)
        removed = set()
        for i in range(n):
            if i in removed: continue
            if i + 1 < n:
                sims = torch.mm(feats[i:i+1], feats[i+1:].t())[0]
                idx = (sims >= threshold).nonzero(as_tuple=True)[0].tolist()
                for j in idx: removed.add(i + 1 + j)
        keep = [i for i in range(n) if i not in removed]
        return [paths[i] for i in keep], [feat_list[i] for i in keep], [subs[i] for i in keep]

    def _cluster_images(self, paths, feat_list, subs, sensitivity):
        """Cluster images with UMAP + Agglomerative. sensitivity 0..1: 0=few big clusters, 1=many small."""
        if not feat_list:
            return {}
        if len(feat_list) == 1:
            self.log("  Found 1 group (only 1 image)")
            return {0: [(paths[0], subs[0])]}

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import euclidean_distances
        import numpy as np

        feats = F.normalize(torch.cat(feat_list, dim=0), p=2, dim=1).numpy()
        n_samples = feats.shape[0]
        feat_dim = feats.shape[1]
        self.log(f"  📊 Features: {n_samples} samples × {feat_dim} dims")

        # ── UMAP dimensionality reduction ──
        umap_dim = min(20, feat_dim - 1, max(2, n_samples - 2))
        if feat_dim > 32 and n_samples > umap_dim + 2:
            try:
                import umap
                n_neighbors = max(2, min(15, n_samples - 1))
                self.log(f"  🔄 UMAP: {feat_dim}d → {umap_dim}d (neighbors={n_neighbors})")
                reducer = umap.UMAP(
                    n_components=umap_dim,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42,
                )
                feats = reducer.fit_transform(feats)
                self.log(f"  ✓ UMAP done → {feats.shape[1]}d")
            except Exception as e:
                self.log(f"  ⚠ UMAP failed ({e}), using raw features")

        # ── Agglomerative clustering on UMAP-reduced features ──
        dists = euclidean_distances(feats)
        tri_idx = np.triu_indices_from(dists, k=1)
        flat_dists = dists[tri_idx]

        d_min = float(np.min(flat_dists))
        d_mean = float(np.mean(flat_dists))
        d_median = float(np.median(flat_dists))

        # sensitivity → percentile of distance distribution
        # sensitivity=0.1 → 10th pct (tight → few big clusters)
        # sensitivity=0.5 → 50th pct (median → balanced)
        # sensitivity=0.9 → 90th pct (loose → many small clusters)
        threshold = float(np.percentile(flat_dists, sensitivity * 100))

        self.log(f"  📊 Distances: min={d_min:.3f} mean={d_mean:.3f} median={d_median:.3f}")
        self.log(f"  📊 Sensitivity={sensitivity:.2f} → threshold={threshold:.3f}")

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='euclidean',
            linkage='ward',
        ).fit_predict(feats)

        n_groups = len(set(labels))
        self.log(f"  ✓ Found {n_groups} clusters")

        groups = {}
        for p, s, cl in zip(paths, subs, labels):
            groups.setdefault(int(cl), []).append((p, s))
        return groups

    # ═══════════════════════════════════════════════════════════════════════
    #  WD SwinV2 Tagger v3
    # ═══════════════════════════════════════════════════════════════════════
    def _load_wd_tagger(self):
        try:
            self.log("Loading WD SwinV2 Tagger…")
            bar = LoadingBar(self._prog, 87, 90, MODEL_SIZES_GB["tagger"] / 0.8)
            bar.start()

            import onnxruntime as ort
            import pandas as pd

            # Destroy any existing session first (prevents CUDA buffer leak)
            if self.wd_tagger is not None:
                try:
                    del self.wd_tagger
                except Exception:
                    pass
                self.wd_tagger = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load model
            model_path = os.path.join(WD_TAGGER_PATH, "model.onnx")
            providers = ['CUDAExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            self.wd_tagger = ort.InferenceSession(model_path, providers=providers)

            # Load tags and categories
            csv_path = os.path.join(WD_TAGGER_PATH, "selected_tags.csv")
            df = pd.read_csv(csv_path)
            self.wd_tags = df['name'].tolist()
            if 'category' in df.columns:
                self.wd_tag_categories = dict(zip(df['name'], df['category']))
            else:
                self.wd_tag_categories = {}

            bar.complete()
            self.log("WD Tagger loaded ✓")
            return True
        except Exception as e:
            self.log(f"❌ WD Tagger: {e}")
            self.log(traceback.format_exc())
            return False

    def _wd_tagger_infer(self, img_input):
        """Run WD Tagger on an image and return dict of tag -> confidence.
        img_input can be a file path (str) or a PIL Image."""
        try:
            import numpy as np
            from PIL import Image

            if isinstance(img_input, str):
                img = Image.open(img_input).convert("RGB")
            else:
                img = img_input.convert("RGB")
            
            # Smart crop to square and resize
            w, h = img.size
            sid = min(w, h)
            left = (w - sid) // 2
            top = (h - sid) // 2
            img = img.crop((left, top, left + sid, top + sid))
            # Tagger v3 input size is 448x448
            img = img.resize((448, 448), Image.Resampling.BICUBIC)

            img_np = np.array(img, dtype=np.float32) / 255.0
            # WD SwinV2 Tagger v3 expects BGR, NHWC, float32 in [0,1]
            img_np = img_np[:, :, ::-1].copy()  # RGB → BGR; .copy() prevents ONNX buffer reuse
            img_np = np.expand_dims(img_np, axis=0).copy()  # NHWC: (1, 448, 448, 3) — contiguous

            # Infer
            input_name = self.wd_tagger.get_inputs()[0].name
            preds = self.wd_tagger.run(None, {input_name: img_np})[0][0]

            # Extract global ratings first (category 9) to determine if image is genuinely NSFW
            ratings = {
                "explicit": 0.0, "questionable": 0.0, "sensitive": 0.0, "general": 0.0
            }
            for i in range(len(self.wd_tags)):
                tag = self.wd_tags[i]
                if tag in ratings:
                    ratings[tag] = float(preds[i])
            
            # Image is considered genuinely mature only if explicit is the dominant rating
            is_genuinely_explicit = ratings["explicit"] > max(ratings["general"], ratings["sensitive"], ratings["questionable"])

            # Construct dictionary — tiered thresholds by tag type
            blocked = {"no_humans", "text_focus", "implied_fellatio"}  # Known false-positive / irrelevant tags
            mature_words = {"nude", "sex", "penis", "vagina", "nipple", "pussy", "breast",
                           "cum", "genital", "anus", "orgasm", "erect", "pubic", "naked",
                           "masturbat", "porn", "hentai", "dildo", "bondage", "tentacle",
                           "unskirt", "cameltoe", "areola", "groin", "topless", "bottomless"}
            res = {}
            for i in range(len(self.wd_tags)):
                tag = self.wd_tags[i]
                if tag in blocked:
                    continue
                score = float(preds[i])
                cat = self.wd_tag_categories.get(tag, -1)
                
                # Characters: lowest threshold (0.05) — maximize detection
                if cat == 4:
                    if score > 0.05:
                        res[tag] = score
                # Mature tags: only allow if the image is genuinely explicit AND tag confidence > 0.35
                elif any(m in tag.lower() for m in mature_words):
                    if is_genuinely_explicit and score > 0.35:
                        res[tag] = score
                # General/other: 0.1 for quality
                elif score > 0.1:
                    res[tag] = score
            return res
        except Exception as e:
            lname = os.path.basename(img_input) if isinstance(img_input, str) else "PIL Image"
            self.log(f"  ⚠ WD Tagger inference error on {lname}: {e}")
            return {}

    def _wd_tagger_refine_clusters(self, clustered_groups):
        """Use WD Tagger to verify similarity within clusters; split if needed."""
        refined = {}
        next_id = 0

        for cl_id, members in clustered_groups.items():
            if self._cancel.is_set():
                refined[next_id] = members
                next_id += 1
                continue

            # Small clusters — keep as-is
            if len(members) <= 3:
                refined[next_id] = members
                next_id += 1
                continue

            # Tag a sample of images
            paths_in = [p for p, _ in members]
            n = len(paths_in)
            step = max(1, n // 10)
            sample_idx = list(range(0, n, step))[:10]

            tag_sets = {}  # idx → set of top tags
            for idx in sample_idx:
                preds = self._wd_tagger_infer(paths_in[idx])
                if preds:
                    # Keep top 30 tags
                    top_tags = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:30]
                    tag_sets[idx] = set(t for t, _ in top_tags)
                else:
                    tag_sets[idx] = set()

            if len(tag_sets) < 2:
                refined[next_id] = members
                next_id += 1
                continue

            # Check pairwise overlap
            ts_list = list(tag_sets.values())
            all_tags = set()
            for ts in ts_list:
                all_tags |= ts

            if not all_tags:
                refined[next_id] = members
                next_id += 1
                continue

            overlaps = []
            for i in range(len(ts_list)):
                for j in range(i + 1, len(ts_list)):
                    union = ts_list[i] | ts_list[j]
                    inter = ts_list[i] & ts_list[j]
                    if union:
                        overlaps.append(len(inter) / len(union))

            avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0

            # 0.25 IoU threshold for tags is usually decent for similarity
            if avg_overlap >= 0.25:
                self.log(f"    ✓ Cluster {cl_id+1} ({n} imgs): coherent (overlap={avg_overlap:.0%})")
                refined[next_id] = members
                next_id += 1
            else:
                self.log(f"    ✂ Cluster {cl_id+1} ({n} imgs): splitting (overlap={avg_overlap:.0%})")
                anchor_idx = list(tag_sets.keys())
                sub_groups = {ai: [] for ai in anchor_idx}

                for mi, (p, s) in enumerate(members):
                    if mi in tag_sets and tag_sets[mi]:
                        sub_groups[mi].append((p, s))
                    else:
                        best = anchor_idx[0]
                        preds = self._wd_tagger_infer(p)
                        if preds:
                            top_tags = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:30]
                            words = set(t for t, _ in top_tags)
                            best_score = -1
                            for ai in anchor_idx:
                                union = words | tag_sets[ai]
                                inter = words & tag_sets[ai]
                                score = len(inter) / len(union) if union else 0
                                if score > best_score:
                                    best_score = score
                                    best = ai
                        sub_groups[best].append((p, s))

                for ai, sg in sub_groups.items():
                    if sg:
                        refined[next_id] = sg
                        self.log(f"      sub-group {next_id+1}: {len(sg)} imgs")
                        next_id += 1

        self.log(f"  Refinement: {len(clustered_groups)} → {len(refined)} clusters")
        return refined

    def _wd_tagger_name_clusters(self, clustered_groups, wd_preds_cache):
        """Name clusters using TF-IDF-style tag scoring for distinctive names."""
        import math
        result = []

        # Tags that are stylistic / too generic — excluded from names entirely
        exclude_tags = {
            "monochrome", "greyscale", "no_humans", "negative_space", "simple_background",
            "white_background", "black_background", "transparent_background", "sketch",
            "lineart", "comic", "manga", "gradient_background", "pattern_background",
            "traditional_media", "watercolor_(medium)", "pixel_art", "oikakeko",
            "highres", "absurdres", "commentary_request", "commentary", "translated",
            "bad_id", "bad_pixiv_id", "revision", "character_request",
        }

        n_clusters = len(clustered_groups)

        # ── Phase 1: Aggregate per-cluster tag scores ──
        cluster_aggr = {}  # {cl_id: {tag: avg_score}}
        cluster_chars = {}  # {cl_id: [character_tags]}
        for cl_id, members in clustered_groups.items():
            aggr = {}
            chars = []
            n_mem = len(members)
            for p, _ in members:
                preds = wd_preds_cache.get(p)
                # Tag uncached images on-the-fly for better naming
                if preds is None and self.wd_tagger is not None:
                    preds = self._wd_tagger_infer(p)
                    if preds:
                        wd_preds_cache[p] = preds
                if not preds:
                    continue
                for tag, score in preds.items():
                    cat = self.wd_tag_categories.get(tag, -1)
                    if cat == 9:  # skip rating
                        continue
                    if tag in exclude_tags:
                        continue
                    if cat == 4 and score > 0.15:  # character tag (lower threshold)
                        if tag not in chars:
                            chars.append(tag)
                    if score < 0.1:  # Skip very low confidence tags for naming
                        continue
                    aggr[tag] = aggr.get(tag, 0) + score
            # Normalize by cluster size → avg relevance
            cluster_aggr[cl_id] = {t: s / n_mem for t, s in aggr.items()}
            cluster_chars[cl_id] = chars

        # ── Phase 2: Compute IDF (Inverse Document/cluster Frequency) ──
        # Tags appearing in ALL clusters get low IDF → they're generic
        tag_cluster_count = {}  # how many clusters contain this tag
        for cl_id, aggr in cluster_aggr.items():
            for tag in aggr:
                tag_cluster_count[tag] = tag_cluster_count.get(tag, 0) + 1

        def idf(tag):
            df = tag_cluster_count.get(tag, 1)
            return math.log(1 + n_clusters / df)

        # ── Phase 3: Score = avg_relevance × IDF ──
        cluster_tfidf = {}
        for cl_id, aggr in cluster_aggr.items():
            scored = {}
            for tag, avg_sc in aggr.items():
                scored[tag] = avg_sc * idf(tag)
            # Sort by TF-IDF score descending
            cluster_tfidf[cl_id] = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        # ── Phase 4: Build names ──
        # Priority: character names first, then top TF-IDF content tags
        cluster_names = {}
        for cl_id in clustered_groups:
            chars = cluster_chars.get(cl_id, [])
            tfidf_tags = cluster_tfidf.get(cl_id, [])

            name_parts = []

            # Add character name(s) first (max 2)
            for ch in chars[:2]:
                clean = ch.replace("_", " ").title()
                if clean not in name_parts:
                    name_parts.append(clean)

            # Fill with top TF-IDF tags (skip characters, already added)
            for tag, sc in tfidf_tags:
                if len(name_parts) >= 4:
                    break
                cat = self.wd_tag_categories.get(tag, -1)
                if cat == 4:  # character already handled
                    continue
                clean = tag.replace("_", " ").title()
                if clean not in name_parts:
                    name_parts.append(clean)

            if name_parts:
                name = self._safe(" ".join(name_parts))[:60]
            else:
                name = f"group_{cl_id + 1:03d}"

            cluster_names[cl_id] = name
            self.log(f"    Cluster {cl_id+1}: \"{name}\" ({len(clustered_groups[cl_id])} imgs)")

        # ── Phase 5: Resolve duplicate names with distinguishing subtags ──
        # Instead of _002 suffix, add the NEXT most distinctive tag
        name_groups = {}
        for cl_id, name in cluster_names.items():
            name_groups.setdefault(name, []).append(cl_id)

        final_names = {}
        for base_name, cl_ids in name_groups.items():
            if len(cl_ids) == 1:
                final_names[cl_ids[0]] = base_name
            else:
                # Find tags that differentiate these specific clusters
                for cl_id in cl_ids:
                    tfidf_tags = cluster_tfidf.get(cl_id, [])
                    # Find a tag that other clusters with same name DON'T have highly
                    extra = None
                    for tag, sc in tfidf_tags:
                        cat = self.wd_tag_categories.get(tag, -1)
                        clean = tag.replace("_", " ").title()
                        if clean in base_name:
                            continue  # already in name
                        # Check if this tag is less important in other same-name clusters
                        is_unique = True
                        for other_id in cl_ids:
                            if other_id == cl_id:
                                continue
                            other_aggr = cluster_aggr.get(other_id, {})
                            if other_aggr.get(tag, 0) > sc * 0.5:
                                is_unique = False
                                break
                        if is_unique:
                            extra = clean
                            break
                    if extra:
                        final_names[cl_id] = self._safe(f"{base_name} {extra}")[:60]
                    else:
                        final_names[cl_id] = f"{base_name}_{cl_ids.index(cl_id) + 1}"

        for cl_id, members in clustered_groups.items():
            folder = final_names.get(cl_id, f"group_{cl_id+1:03d}")
            for p, s in members:
                parts = ([s] if s else []) + [folder]
                result.append((p, os.path.join(*parts)))

        return result

    def _wd_tagger_generate_metadata(self, accepted, target, wd_preds_cache):
        """Generate rich metadata using cached WD Tagger results."""
        import random
        folder_map = {}
        for p, s in accepted:
            folder_map.setdefault(s or "", []).append(p)

        total_folders = len(folder_map)
        done = 0

        # Define explicit tag patterns (mature)
        mature_patterns = ['nude', 'sex', 'penis', 'vagina', 'nipple', 'pussy', 'ass', 'breast', 
                           'cum', 'genital', 'anus', 'orgasm', 'erect', 'pubic', 'naked', 
                           'masturbat', 'porn', 'hentai', 'dildo', 'bondage', 'rape', 'tentacle']

        for subfolder, paths in folder_map.items():
            if self._cancel.is_set(): break
            
            max_n = self.meta_max_per_folder.get()
            sample = paths[:] if max_n <= 0 else random.sample(paths, min(max_n, len(paths)))
            tags_per_img = self.meta_tags_per_image.get()
            
            images_meta = {}
            has_mature_folder = False
            
            for p in sample:
                fname = os.path.basename(p)
                preds = wd_preds_cache.get(p, {})
                
                if not preds:
                    images_meta[fname] = {"error": "no tags found"}
                    continue
                
                # Extract rating scores
                rating = {t: round(preds.get(t, 0), 3) for t in ["explicit", "questionable", "sensitive", "general"]}
                
                # Separate tags
                general_tags = []
                character_tags = []
                mature_tags = []
                all_scores = {}
                
                # Filter out rating tags (category 9), use low threshold for diversity
                content_tags = [(t, sc) for t, sc in preds.items() if self.wd_tag_categories.get(t, -1) != 9 and sc > 0.07]
                content_tags.sort(key=lambda x: x[1], reverse=True)
                
                for t, sc in content_tags[:tags_per_img]:
                    all_scores[t] = round(sc, 3)
                    cat = self.wd_tag_categories.get(t, 0)
                    
                    # Check for explicit tags
                    is_mature = any(m in t.lower() for m in mature_patterns)
                    if is_mature:
                        mature_tags.append(t)
                        has_mature_folder = True
                    elif cat == 4:
                        character_tags.append(t)
                    else:
                        general_tags.append(t)
                
                images_meta[fname] = {
                    "rating": rating,
                    "general_tags": general_tags,
                    "character_tags": character_tags,
                    "mature_tags": mature_tags,
                    "all_scores": all_scores
                }
                
                log_tag_str = ", ".join(general_tags[:5] + character_tags[:2] + mature_tags[:2])
                self.log(f"    📝 {fname}: {log_tag_str[:60]}…")

            metadata = {
                "folder": subfolder or "root",
                "total_images": len(paths),
                "sampled": len(sample),
                "images": images_meta
            }
            
            if has_mature_folder:
                self.log(f"  🔞 {subfolder or 'root'}: Contains explicit/mature tags")

            dst_dir = os.path.join(target, self._safe(subfolder)) if subfolder else target
            os.makedirs(dst_dir, exist_ok=True)
            try:
                with open(os.path.join(dst_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                pass
            
            done += 1
            self._prog(92 + done / max(total_folders, 1) * 3)

    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _safe(n):
        return "".join(c for c in n if c.isalnum() or c in ' _-/\\').strip() or "other"

    def _place(self, src, dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        nm, ext = os.path.splitext(os.path.basename(src))
        dst = os.path.join(dst_dir, f"{nm}{ext}")
        c = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, f"{nm}_{c}{ext}"); c += 1
        (shutil.move if self.move_files.get() else shutil.copy2)(src, dst)


if __name__ == "__main__":
    r = tk.Tk()
    ImageSorterApp(r)
    r.mainloop()

