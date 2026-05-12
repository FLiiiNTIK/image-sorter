from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from .dataset import clear_cache
from .lmdb_cache import configured_lmdb_dir
from .pipeline import get_run_history, get_runtime_status, run_ai_search, run_metadata_export, run_pipeline, undo_last_move_run
from .utils import (
    CONTENT_NEG_ANCHOR,
    MODEL_SIZES_GB,
    SETTINGS_BOOL_KEYS,
    SETTINGS_NUMBER_KEYS,
    SETTINGS_STRING_KEYS,
    STYLE_NEG_ANCHOR,
    format_size,
    load_settings,
    get_paths,
    read_json,
    save_settings,
    safe_filename,
    write_json,
)


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self._tip_window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self._tip_window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(f"+{x}+{y}")
        label = tk.Label(win, text=self.text, background="#fffde7", relief=tk.SOLID, borderwidth=1, padx=6, pady=3, justify=tk.LEFT)
        label.pack()
        self._tip_window = win

    def _hide(self, _event=None) -> None:
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None


class ImageSorterUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Sorter v1.2.1")
        self.root.geometry("900x900")
        self.root.minsize(800, 700)
        self.cancel_event = threading.Event()
        self.stop_after_batch_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.training_process: subprocess.Popen[str] | None = None
        self.training_stop_file: Path | None = None
        self._run_start_time: float | None = None
        self._group_desc_history: list[str] = []
        self._tooltips: list[Tooltip] = []
        self._search_results: list[dict[str, object]] = []
        self._search_preview_image: ImageTk.PhotoImage | None = None
        self.quick_start_profiles = {
            "Manual": None,
            "Fast Sort": {
                "sort_ai_human": True,
                "use_sdxl_vote": False,
                "sort_content": False,
                "sort_style": False,
                "sort_grouping": False,
                "sort_dedup": False,
                "gen_metadata": False,
                "batch_size_var": 16,
                "precision_var": "auto",
                "route_uncertain": True,
                "uncertainty_threshold": 0.7,
                "dry_run": False,
            },
            "Max Quality": {
                "sort_ai_human": True,
                "use_sdxl_vote": True,
                "sort_content": True,
                "sort_style": True,
                "sort_grouping": True,
                "sort_dedup": True,
                "gen_metadata": True,
                "use_florence": True,
                "batch_size_var": 4,
                "precision_var": "auto",
                "route_uncertain": True,
                "uncertainty_threshold": 0.8,
                "remember_cluster_names": True,
                "dry_run": False,
            },
            "Search/Metadata Only": {
                "sort_ai_human": False,
                "sort_content": False,
                "sort_style": False,
                "sort_grouping": False,
                "sort_dedup": False,
                "gen_metadata": True,
                "use_florence": True,
                "batch_size_var": 4,
                "route_uncertain": True,
                "uncertainty_threshold": 0.75,
                "dry_run": False,
            },
            "Cluster Preview": {
                "sort_ai_human": False,
                "use_sdxl_vote": False,
                "sort_content": False,
                "sort_style": False,
                "sort_grouping": True,
                "sort_dedup": False,
                "gen_metadata": False,
                "group_sequences": True,
                "remember_cluster_names": True,
                "batch_size_var": 8,
                "max_folders_created": 20,
                "dry_run": True,
            },
            "Duplicates Only": {
                "sort_ai_human": False,
                "use_sdxl_vote": False,
                "sort_content": False,
                "sort_style": False,
                "sort_grouping": False,
                "sort_dedup": True,
                "gen_metadata": False,
                "group_sequences": False,
                "move_files": False,
                "dry_run": False,
            },
        }
        self.tag_presets = {
            "Photography": {
                "content": "person, animal, nature, portrait, landscape, cityscape, vehicle",
                "style": "black and white, macro, polaroid, long exposure, cinematic lighting",
            },
            "Anime/Art": {
                "content": "1girl, 1boy, solo, multiple girls, animal ears, building, mecha",
                "style": "anime, sketch, watercolor, line art, 3d render, flat color, concept art",
            },
        }
        self.config_presets: dict[str, dict[str, object]] = {}
        self._init_vars()
        self._build_ui()
        self._load_settings()
        self._update_status_label()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_vars(self) -> None:
        self.source_dir = tk.StringVar()
        self.target_dir = tk.StringVar()
        self.blocked_subfolders = tk.StringVar()
        self.quick_start_var = tk.StringVar(value="Manual")
        self.config_preset_var = tk.StringVar(value="None")
        self.tag_preset_var = tk.StringVar(value="None")
        self.sort_ai_human = tk.BooleanVar(value=True)
        self.use_sdxl_vote = tk.BooleanVar(value=True)
        self.sort_content = tk.BooleanVar(value=False)
        self.content_min_conf = tk.DoubleVar(value=0.05)
        self.content_neg_anchor = tk.StringVar(value=CONTENT_NEG_ANCHOR)
        self.sort_style = tk.BooleanVar(value=False)
        self.style_min_conf = tk.DoubleVar(value=0.05)
        self.style_neg_anchor = tk.StringVar(value=STYLE_NEG_ANCHOR)
        self.sort_grouping = tk.BooleanVar(value=False)
        self.sort_dedup = tk.BooleanVar(value=False)
        self.move_files = tk.BooleanVar(value=False)
        self.recursive_scan = tk.BooleanVar(value=True)
        self.optimize_models = tk.BooleanVar(value=False)
        self.content_tags = tk.StringVar(value="person, animal, landscape, vehicle, food, building, game character")
        self.style_tags = tk.StringVar(value="anime, realistic photo, sketch, oil painting, 3d render, pixel art, watercolor")
        self.group_threshold = tk.DoubleVar(value=0.35)
        self.dedup_threshold = tk.DoubleVar(value=0.88)
        self.batch_size_var = tk.IntVar(value=8)
        self.dino_batch_size = tk.IntVar(value=0)
        self.tagger_batch_size = tk.IntVar(value=0)
        self.group_desc_var = tk.StringVar(value="")
        self.precision_var = tk.StringVar(value="auto")
        self.auto_vram_batch = tk.BooleanVar(value=True)
        self.unload_inactive_models = tk.BooleanVar(value=True)
        self.vram_profile = tk.StringVar(value="Auto")
        self.vram_limit_gb = tk.DoubleVar(value=0.0)
        self.tagger_engine_var = tk.StringVar(value="Camie-Tagger-v2")
        self.camie_threshold = tk.DoubleVar(value=0.05)
        self.gen_metadata = tk.BooleanVar(value=False)
        self.group_sequences = tk.BooleanVar(value=True)
        self.use_dino_adapter = tk.BooleanVar(value=True)
        self.meta_max_per_folder = tk.IntVar(value=0)
        self.meta_max_tokens = tk.IntVar(value=0)
        self.meta_tags_per_image = tk.IntVar(value=30)
        self.use_florence = tk.BooleanVar(value=False)
        self.florence_mode = tk.StringVar(value="<DETAILED_CAPTION>")
        self.florence_char = tk.StringVar(value="")
        self.florence_profile = tk.StringVar(value="Balanced")
        self.florence_batch_size = tk.IntVar(value=4)
        self.florence_max_side = tk.IntVar(value=768)
        self.sort_tagger_filter = tk.BooleanVar(value=False)
        self.tagger_filter_tags = tk.StringVar(value="")
        self.route_uncertain = tk.BooleanVar(value=True)
        self.dry_run = tk.BooleanVar(value=False)
        self.remember_cluster_names = tk.BooleanVar(value=True)
        self.canonical_folder_names = tk.BooleanVar(value=False)
        self.hierarchical_folder_names = tk.BooleanVar(value=False)
        self.global_name_optimization = tk.BooleanVar(value=True)
        self.character_aware_recursive = tk.BooleanVar(value=False)
        self.character_create_multiple = tk.BooleanVar(value=False)
        self.name_uniqueness_level = tk.StringVar(value="Medium")
        self.uncertainty_threshold = tk.DoubleVar(value=0.7)
        self.max_folders_created = tk.IntVar(value=0)
        self.character_min_score = tk.DoubleVar(value=0.35)
        self.character_margin = tk.DoubleVar(value=0.08)
        self.character_max_multi = tk.IntVar(value=4)
        self.lmdb_mode = tk.StringVar(value="auto")
        self.lmdb_cache_dir = tk.StringVar(value=str(configured_lmdb_dir()))
        self.rebuild_lmdb = tk.BooleanVar(value=False)
        self.metadata_skip_existing_captions = tk.BooleanVar(value=True)
        self.metadata_save_every = tk.IntVar(value=500)
        self.metadata_florence_per_folder = tk.IntVar(value=0)
        self.train_metadata_source = tk.StringVar(value="auto")
        self.train_epochs = tk.IntVar(value=30)
        self.train_batch_size = tk.IntVar(value=32)
        self.train_grad_accum = tk.IntVar(value=2)
        self.train_lr = tk.DoubleVar(value=0.0003)
        self.train_min_tag_score = tk.DoubleVar(value=0.1)
        self.train_max_side = tk.IntVar(value=1024)
        self.train_augment_mode = tk.StringVar(value="light")
        self.train_manga = tk.BooleanVar(value=True)
        self.train_finetune = tk.BooleanVar(value=False)
        self.train_fresh = tk.BooleanVar(value=True)
        self.train_no_auto_optimize = tk.BooleanVar(value=False)
        self.train_cpu = tk.BooleanVar(value=False)
        self.eta_var = tk.StringVar(value="ETA: --")

    def _configure_theme(self) -> None:
        self.root.configure(bg="#edf1f7")
        self.root.option_add("*Font", "{Segoe UI} 9")
        self.root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 9")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#edf1f7"
        panel = "#f8fafc"
        border = "#cfd7e6"
        text = "#243041"
        muted = "#657386"
        accent = "#22b8a7"
        accent_dark = "#159789"
        danger = "#e96a5f"
        style.configure(".", background=bg, foreground=text, font="{Segoe UI} 9")
        style.configure("App.TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Toolbar.TFrame", background="#e4eaf3")
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure("Panel.TLabel", background=panel, foreground=text)
        style.configure("TLabelFrame", background=panel, bordercolor=border, relief=tk.GROOVE)
        style.configure("TLabelFrame.Label", background=panel, foreground=text, font="{Segoe UI} 9 bold")
        style.configure("TCheckbutton", background=panel, foreground=text)
        style.configure("TRadiobutton", background=panel, foreground=text)
        style.configure("TEntry", fieldbackground="#ffffff", bordercolor=border, lightcolor=border, darkcolor=border)
        style.configure("TCombobox", fieldbackground="#ffffff", bordercolor=border, arrowcolor=muted)
        style.configure("TSpinbox", fieldbackground="#ffffff", bordercolor=border)
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background="#dfe6f0", foreground=text, padding=(14, 6), font="{Segoe UI} 9")
        style.map("TNotebook.Tab", background=[("selected", panel)], foreground=[("selected", "#101827")])
        style.configure("Hidden.TNotebook", background=bg, borderwidth=0, tabmargins=0)
        style.layout("Hidden.TNotebook.Tab", [])
        style.configure("TButton", background="#e8edf5", foreground=text, padding=(10, 5), bordercolor=border)
        style.map("TButton", background=[("active", "#dce4ef")])
        style.configure("Nav.TButton", background="#edf1f7", foreground=text, padding=(12, 7), borderwidth=0)
        style.map("Nav.TButton", background=[("active", "#dde7f2")])
        style.configure("Primary.TButton", background=accent, foreground="#ffffff", padding=(14, 7), bordercolor=accent)
        style.map("Primary.TButton", background=[("active", accent_dark), ("disabled", "#9fcfc8")])
        style.configure("Danger.TButton", background=danger, foreground="#ffffff", padding=(10, 5), bordercolor=danger)
        style.map("Danger.TButton", background=[("active", "#d4574d"), ("disabled", "#e8aaa5")])
        style.configure("Horizontal.TProgressbar", troughcolor="#dce3ec", background=accent, bordercolor=border)

    def _build_ui(self) -> None:
        self.root.geometry("1120x660")
        self.root.minsize(980, 590)
        self._configure_theme()

        main = ttk.Frame(self.root, padding=(10, 8), style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        nav = ttk.Frame(main, style="App.TFrame")
        nav.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(nav, text="Image Sorter", font="{Segoe UI} 11 bold").pack(side=tk.LEFT, padx=(2, 12))

        files = ttk.LabelFrame(main, text="Files List", padding=8)
        files.pack(fill=tk.X, pady=(0, 6))
        files.columnconfigure(0, weight=5)
        files.columnconfigure(1, weight=4)

        queue_box = tk.Label(
            files,
            text="Drag and drop files or folders here\n\nSource image folder",
            bg="#ffffff",
            fg="#1f2937",
            activebackground="#ffffff",
            relief=tk.SOLID,
            bd=1,
            font="{Segoe UI} 14",
            height=5,
        )
        queue_box.grid(row=0, column=0, rowspan=3, sticky=tk.NSEW, padx=(0, 10))
        self.drop_zone = queue_box

        path_panel = ttk.Frame(files, style="Panel.TFrame")
        path_panel.grid(row=0, column=1, sticky=tk.NSEW)
        path_panel.columnconfigure(1, weight=1)
        ttk.Label(path_panel, text="Source:", style="Panel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(1, 5))
        ttk.Entry(path_panel, textvariable=self.source_dir).grid(row=0, column=1, sticky=tk.EW, padx=6, pady=(1, 5))
        ttk.Button(path_panel, text="Browse...", command=lambda: self._browse(self.source_dir)).grid(row=0, column=2, pady=(1, 5))
        ttk.Label(path_panel, text="Output:", style="Panel.TLabel").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(path_panel, textvariable=self.target_dir).grid(row=1, column=1, sticky=tk.EW, padx=6)
        ttk.Button(path_panel, text="Browse...", command=lambda: self._browse(self.target_dir)).grid(row=1, column=2)
        ttk.Label(path_panel, text="Blocked:", style="Panel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        blocked_entry = ttk.Entry(path_panel, textvariable=self.blocked_subfolders)
        blocked_entry.grid(row=2, column=1, sticky=tk.EW, padx=6, pady=(5, 0))
        ttk.Button(path_panel, text="Add...", command=self._add_blocked_folder).grid(row=2, column=2, pady=(5, 0))

        toolbar = ttk.Frame(files, style="Toolbar.TFrame", padding=(6, 5))
        toolbar.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        ttk.Checkbutton(toolbar, text="Recursive", variable=self.recursive_scan).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Dry run", variable=self.dry_run).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="Clear cache", command=self._clear_cache).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="AI Search", command=self._open_ai_search).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, text="Quick Start:").pack(side=tk.LEFT, padx=(14, 4))
        quick_cb = ttk.Combobox(toolbar, textvariable=self.quick_start_var, state="readonly", width=20, values=list(self.quick_start_profiles.keys()))
        quick_cb.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Apply", command=self._apply_quick_start_profile).pack(side=tk.LEFT, padx=(5, 0))
        self.cache_lbl = ttk.Label(toolbar, text="", foreground="#657386")
        self.cache_lbl.pack(side=tk.LEFT, padx=(10, 0))
        self.start_btn = ttk.Button(toolbar, text="Start", command=self._start, style="Primary.TButton")
        self.start_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self.cancel_btn = ttk.Button(toolbar, text="Cancel", command=self._do_cancel, state=tk.DISABLED, style="Danger.TButton")
        self.cancel_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self.stop_after_btn = ttk.Button(toolbar, text="Stop after batch", command=self._stop_after_current_batch, state=tk.DISABLED)
        self.stop_after_btn.pack(side=tk.RIGHT, padx=(6, 0))

        body = ttk.Frame(main, style="App.TFrame")
        body.pack(fill=tk.X, pady=(0, 6))
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=2)

        notebook = ttk.Notebook(body, style="Hidden.TNotebook", height=118)
        notebook.grid(row=0, column=0, sticky=tk.EW, padx=(0, 8))

        home_tab = ttk.Frame(notebook, padding=10, style="Panel.TFrame")
        criteria_tab = ttk.Frame(notebook, padding=10, style="Panel.TFrame")
        metadata_tab = ttk.Frame(notebook, padding=10, style="Panel.TFrame")
        engine_tab = ttk.Frame(notebook, padding=10, style="Panel.TFrame")
        training_tab = ttk.Frame(notebook, padding=10, style="Panel.TFrame")
        notebook.add(home_tab, text="Home")
        notebook.add(criteria_tab, text="Image settings")
        notebook.add(metadata_tab, text="Metadata")
        notebook.add(engine_tab, text="Engine settings")
        notebook.add(training_tab, text="Training")

        tab_heights = {0: 118, 1: 245, 2: 255, 3: 285, 4: 260}

        def select_settings_tab(index: int) -> None:
            notebook.configure(height=tab_heights.get(index, 160))
            notebook.select(index)

        for label, target in (
            ("Home", 0),
            ("Image settings", 1),
            ("Metadata", 2),
            ("Engine settings", 3),
            ("Training", 4),
        ):
            ttk.Button(nav, text=label, style="Nav.TButton", command=lambda idx=target: select_settings_tab(idx)).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(nav, text="History", style="Nav.TButton", command=self._open_run_history).pack(side=tk.LEFT, padx=(8, 3))
        ttk.Button(nav, text="AI Search", style="Nav.TButton", command=self._open_ai_search).pack(side=tk.LEFT, padx=(0, 3))

        home_tab.columnconfigure(1, weight=1)
        ttk.Label(home_tab, text="Tag Preset:", style="Panel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.tag_preset_cb = ttk.Combobox(home_tab, textvariable=self.tag_preset_var, state="readonly", width=22)
        self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
        self.tag_preset_cb.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)
        self.tag_preset_cb.bind("<<ComboboxSelected>>", self._on_preset_select)
        ttk.Button(home_tab, text="Save", command=self._save_preset, width=8).grid(row=0, column=2, padx=(4, 2), pady=2)
        ttk.Button(home_tab, text="Delete", command=self._delete_preset, width=8).grid(row=0, column=3, pady=2)

        ttk.Label(home_tab, text="Run Preset:", style="Panel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_preset_cb = ttk.Combobox(home_tab, textvariable=self.config_preset_var, state="readonly", width=22)
        self.config_preset_cb["values"] = ["None"]
        self.config_preset_cb.grid(row=1, column=1, sticky=tk.W, padx=6, pady=2)
        self.config_preset_cb.bind("<<ComboboxSelected>>", self._on_config_preset_select)
        ttk.Button(home_tab, text="Save", command=self._save_config_preset, width=8).grid(row=1, column=2, padx=(4, 2), pady=2)
        ttk.Button(home_tab, text="Delete", command=self._delete_config_preset, width=8).grid(row=1, column=3, pady=2)

        home_line = ttk.Frame(home_tab, style="Panel.TFrame")
        home_line.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=(4, 0))
        ttk.Checkbutton(home_line, text="AI vs Human", variable=self.sort_ai_human).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(home_line, text="Dual-vote (SigLIP+SDXL)", variable=self.use_sdxl_vote).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(home_line, text="Needs_Review routing", variable=self.route_uncertain).pack(side=tk.LEFT)
        ttk.Label(home_line, text="Min conf:").pack(side=tk.LEFT, padx=(10, 3))
        uncertain_entry = ttk.Entry(home_line, textvariable=self.uncertainty_threshold, width=6)
        uncertain_entry.pack(side=tk.LEFT)
        ttk.Checkbutton(home_line, text="Dry run", variable=self.dry_run).pack(side=tk.LEFT, padx=(12, 0))

        criteria_tab.columnconfigure(1, weight=1)
        ttk.Checkbutton(criteria_tab, text="Content", variable=self.sort_content).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(criteria_tab, textvariable=self.content_tags).grid(row=0, column=1, sticky=tk.EW, padx=8, pady=5)
        ttk.Label(criteria_tab, text="Min:").grid(row=0, column=2, sticky=tk.E, pady=5)
        content_min_entry = ttk.Entry(criteria_tab, textvariable=self.content_min_conf, width=7)
        content_min_entry.grid(row=0, column=3, padx=(4, 0), pady=5)

        ttk.Checkbutton(criteria_tab, text="Style", variable=self.sort_style).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(criteria_tab, textvariable=self.style_tags).grid(row=1, column=1, sticky=tk.EW, padx=8, pady=5)
        ttk.Label(criteria_tab, text="Min:").grid(row=1, column=2, sticky=tk.E, pady=5)
        style_min_entry = ttk.Entry(criteria_tab, textvariable=self.style_min_conf, width=7)
        style_min_entry.grid(row=1, column=3, padx=(4, 0), pady=5)

        ttk.Checkbutton(criteria_tab, text="Tagger Filter", variable=self.sort_tagger_filter).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(criteria_tab, textvariable=self.tagger_filter_tags).grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=5)

        ttk.Checkbutton(criteria_tab, text="Visual Grouping", variable=self.sort_grouping).grid(row=3, column=0, sticky=tk.W, pady=5)
        group_line = ttk.Frame(criteria_tab, style="Panel.TFrame")
        group_line.grid(row=3, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=5)
        ttk.Label(group_line, text="Sensitivity:").pack(side=tk.LEFT)
        group_thr_entry = ttk.Entry(group_line, textvariable=self.group_threshold, width=7)
        group_thr_entry.pack(side=tk.LEFT, padx=(5, 14))
        ttk.Label(group_line, text="Semantic:").pack(side=tk.LEFT)
        ttk.Entry(group_line, textvariable=self.group_desc_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Label(group_line, text="Max folders:").pack(side=tk.LEFT, padx=(14, 4))
        visual_max_folders_spin = ttk.Spinbox(group_line, from_=0, to=9999, width=7, textvariable=self.max_folders_created)
        visual_max_folders_spin.pack(side=tk.LEFT)
        ttk.Checkbutton(group_line, text="Remember names", variable=self.remember_cluster_names).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(group_line, text="Names...", command=self._open_cluster_name_memory).pack(side=tk.LEFT, padx=(8, 0))

        name_line = ttk.Frame(criteria_tab, style="Panel.TFrame")
        name_line.grid(row=4, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=(0, 5))
        ttk.Label(name_line, text="Name uniqueness:").pack(side=tk.LEFT)
        ttk.Combobox(name_line, textvariable=self.name_uniqueness_level, width=9, state="readonly", values=["Low", "Medium", "High"]).pack(side=tk.LEFT, padx=(5, 12))
        ttk.Checkbutton(name_line, text="Global optimize", variable=self.global_name_optimization).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(name_line, text="Canonical", variable=self.canonical_folder_names).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(name_line, text="Hierarchy", variable=self.hierarchical_folder_names).pack(side=tk.LEFT)

        char_line = ttk.Frame(criteria_tab, style="Panel.TFrame")
        char_line.grid(row=5, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=(0, 5))
        ttk.Checkbutton(char_line, text="Character-aware recursive", variable=self.character_aware_recursive).pack(side=tk.LEFT)
        ttk.Checkbutton(char_line, text="Create multiple", variable=self.character_create_multiple).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(char_line, text="Min:").pack(side=tk.LEFT, padx=(12, 3))
        char_min_entry = ttk.Entry(char_line, textvariable=self.character_min_score, width=6)
        char_min_entry.pack(side=tk.LEFT)
        ttk.Label(char_line, text="Margin:").pack(side=tk.LEFT, padx=(8, 3))
        char_margin_entry = ttk.Entry(char_line, textvariable=self.character_margin, width=6)
        char_margin_entry.pack(side=tk.LEFT)
        ttk.Label(char_line, text="Max multi:").pack(side=tk.LEFT, padx=(8, 3))
        char_multi_spin = ttk.Spinbox(char_line, from_=2, to=8, width=5, textvariable=self.character_max_multi)
        char_multi_spin.pack(side=tk.LEFT)

        ttk.Checkbutton(criteria_tab, text="Remove Duplicates", variable=self.sort_dedup).grid(row=6, column=0, sticky=tk.W, pady=5)
        dedup_line = ttk.Frame(criteria_tab, style="Panel.TFrame")
        dedup_line.grid(row=6, column=1, columnspan=3, sticky=tk.W, padx=8, pady=5)
        ttk.Label(dedup_line, text="Threshold:").pack(side=tk.LEFT)
        dedup_entry = ttk.Entry(dedup_line, textvariable=self.dedup_threshold, width=7)
        dedup_entry.pack(side=tk.LEFT, padx=5)

        metadata_tab.columnconfigure(1, weight=1)
        ttk.Checkbutton(metadata_tab, text="Generate Detailed Metadata", variable=self.gen_metadata).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=3)
        ttk.Label(metadata_tab, text="Max/folder:", style="Panel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=3)
        max_folder_spin = ttk.Spinbox(metadata_tab, from_=0, to=9999, width=7, textvariable=self.meta_max_per_folder)
        max_folder_spin.grid(row=1, column=1, sticky=tk.W, padx=8, pady=3)
        ttk.Label(metadata_tab, text="Tags/img:", style="Panel.TLabel").grid(row=1, column=2, sticky=tk.E, pady=3)
        tags_img_spin = ttk.Spinbox(metadata_tab, from_=5, to=100, width=7, textvariable=self.meta_tags_per_image)
        tags_img_spin.grid(row=1, column=3, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(metadata_tab, text="Florence-2 PromptGen", variable=self.use_florence).grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(metadata_tab, textvariable=self.florence_mode, width=24, state="readonly", values=["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<MIXED_CAPTION>", "<GENERATE_TAGS>"]).grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=8, pady=3)
        ttk.Label(metadata_tab, text="Preset:", style="Panel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=3)
        cb = ttk.Combobox(metadata_tab, textvariable=self.florence_profile, width=12, state="readonly", values=["Fast", "Balanced", "Quality"])
        cb.grid(row=3, column=1, sticky=tk.W, padx=8, pady=3)
        cb.bind("<<ComboboxSelected>>", self._on_florence_profile_change)
        ttk.Label(metadata_tab, text="Batch:", style="Panel.TLabel").grid(row=3, column=2, sticky=tk.E, pady=3)
        ttk.Spinbox(metadata_tab, from_=1, to=64, width=7, textvariable=self.florence_batch_size).grid(row=3, column=3, sticky=tk.W, padx=8, pady=3)
        ttk.Label(metadata_tab, text="Max side:", style="Panel.TLabel").grid(row=4, column=0, sticky=tk.W, pady=3)
        max_side_spin = ttk.Spinbox(metadata_tab, from_=256, to=2048, increment=64, width=7, textvariable=self.florence_max_side)
        max_side_spin.grid(row=4, column=1, sticky=tk.W, padx=8, pady=3)
        ttk.Label(metadata_tab, text="Max tokens:", style="Panel.TLabel").grid(row=4, column=2, sticky=tk.E, pady=3)
        max_tokens_spin = ttk.Spinbox(metadata_tab, from_=0, to=1024, increment=16, width=7, textvariable=self.meta_max_tokens)
        max_tokens_spin.grid(row=4, column=3, sticky=tk.W, padx=8, pady=3)
        metadata_resume_line = ttk.Frame(metadata_tab, style="Panel.TFrame")
        metadata_resume_line.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=(4, 0))
        ttk.Checkbutton(metadata_resume_line, text="Skip existing captions", variable=self.metadata_skip_existing_captions).pack(side=tk.LEFT)
        ttk.Label(metadata_resume_line, text="Save every:").pack(side=tk.LEFT, padx=(12, 3))
        save_every_spin = ttk.Spinbox(metadata_resume_line, from_=0, to=100000, increment=100, width=7, textvariable=self.metadata_save_every)
        save_every_spin.pack(side=tk.LEFT)
        ttk.Label(metadata_resume_line, text="Florence/folder:").pack(side=tk.LEFT, padx=(12, 3))
        florence_per_folder_spin = ttk.Spinbox(metadata_resume_line, from_=0, to=100000, width=7, textvariable=self.metadata_florence_per_folder)
        florence_per_folder_spin.pack(side=tk.LEFT)
        ttk.Label(metadata_resume_line, text="Camie min:").pack(side=tk.LEFT, padx=(12, 3))
        metadata_camie_thr_entry = ttk.Entry(metadata_resume_line, textvariable=self.camie_threshold, width=6)
        metadata_camie_thr_entry.pack(side=tk.LEFT)
        self.metadata_export_btn = ttk.Button(
            metadata_tab,
            text="Generate Source Metadata",
            command=self._start_metadata_export,
            style="Primary.TButton",
        )
        self.metadata_export_btn.grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0))

        engine_tab.columnconfigure(1, weight=1)
        ttk.Checkbutton(engine_tab, text="Group Manga/Sequences", variable=self.group_sequences).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(engine_tab, text="torch.compile", variable=self.optimize_models).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Checkbutton(engine_tab, text="Dry run / preview only", variable=self.dry_run).grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=(10, 0), pady=5)
        ttk.Radiobutton(engine_tab, text="Copy", variable=self.move_files, value=False).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(engine_tab, text="Move", variable=self.move_files, value=True).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(engine_tab, text="Batch:", style="Panel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(engine_tab, from_=1, to=128, width=7, textvariable=self.batch_size_var).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(engine_tab, text="Max folders:", style="Panel.TLabel").grid(row=2, column=2, sticky=tk.E, padx=(10, 4), pady=5)
        max_folders_spin = ttk.Spinbox(engine_tab, from_=0, to=9999, width=7, textvariable=self.max_folders_created)
        max_folders_spin.grid(row=2, column=3, sticky=tk.W, pady=5)
        ttk.Label(engine_tab, text="Precision:", style="Panel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=5)
        pc = ttk.Combobox(engine_tab, textvariable=self.precision_var, width=8, state="readonly", values=["auto", "bf16", "fp16", "fp32"])
        pc.grid(row=3, column=1, sticky=tk.W, pady=5)
        pc.bind("<<ComboboxSelected>>", self._on_precision_change)
        ttk.Label(engine_tab, text="Tagger:", style="Panel.TLabel").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(engine_tab, textvariable=self.tagger_engine_var, width=18, state="readonly", values=["Camie-Tagger-v2", "WD EVA02 v3"]).grid(row=4, column=1, sticky=tk.W, pady=5)
        ttk.Label(engine_tab, text="Camie min:", style="Panel.TLabel").grid(row=4, column=2, sticky=tk.E, padx=(10, 4), pady=5)
        camie_thr_entry = ttk.Entry(engine_tab, textvariable=self.camie_threshold, width=7)
        camie_thr_entry.grid(row=4, column=3, sticky=tk.W, pady=5)
        ttk.Checkbutton(engine_tab, text="Auto VRAM batch", variable=self.auto_vram_batch).grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(engine_tab, text="Unload inactive models", variable=self.unload_inactive_models).grid(row=5, column=1, sticky=tk.W, pady=5)
        ttk.Label(engine_tab, text="VRAM:", style="Panel.TLabel").grid(row=5, column=2, sticky=tk.E, padx=(10, 4), pady=5)
        vram_profile_cb = ttk.Combobox(engine_tab, textvariable=self.vram_profile, width=9, state="readonly", values=["Auto", "8 GB", "12 GB", "16 GB", "Custom"])
        vram_profile_cb.grid(row=5, column=3, sticky=tk.W, pady=5)
        engine_batch_line = ttk.Frame(engine_tab, style="Panel.TFrame")
        engine_batch_line.grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=5)
        ttk.Label(engine_batch_line, text="DINO batch:").pack(side=tk.LEFT)
        dino_batch_spin = ttk.Spinbox(engine_batch_line, from_=0, to=128, width=7, textvariable=self.dino_batch_size)
        dino_batch_spin.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(engine_batch_line, text="Tagger batch:").pack(side=tk.LEFT)
        tagger_batch_spin = ttk.Spinbox(engine_batch_line, from_=0, to=128, width=7, textvariable=self.tagger_batch_size)
        tagger_batch_spin.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(engine_batch_line, text="Custom GB:").pack(side=tk.LEFT)
        vram_limit_spin = ttk.Spinbox(engine_batch_line, from_=0, to=64, increment=0.5, width=7, textvariable=self.vram_limit_gb)
        vram_limit_spin.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(engine_tab, text="LMDB:", style="Panel.TLabel").grid(row=7, column=0, sticky=tk.W, pady=5)
        lmdb_mode_cb = ttk.Combobox(engine_tab, textvariable=self.lmdb_mode, width=8, state="readonly", values=["auto", "on", "off"])
        lmdb_mode_cb.grid(row=7, column=1, sticky=tk.W, pady=5)
        ttk.Checkbutton(engine_tab, text="Rebuild", variable=self.rebuild_lmdb).grid(row=7, column=2, sticky=tk.W, padx=(10, 0), pady=5)
        lmdb_line = ttk.Frame(engine_tab, style="Panel.TFrame")
        lmdb_line.grid(row=8, column=0, columnspan=4, sticky=tk.EW, pady=5)
        lmdb_line.columnconfigure(1, weight=1)
        ttk.Label(lmdb_line, text="Cache dir:", style="Panel.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(lmdb_line, textvariable=self.lmdb_cache_dir).grid(row=0, column=1, sticky=tk.EW, padx=6)
        ttk.Button(lmdb_line, text="Browse...", command=lambda: self._browse(self.lmdb_cache_dir)).grid(row=0, column=2)

        training_tab.columnconfigure(1, weight=1)
        training_tab.columnconfigure(3, weight=1)
        ttk.Label(training_tab, text="Dataset:", style="Panel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(training_tab, textvariable=self.source_dir).grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=8, pady=4)
        ttk.Button(training_tab, text="Browse...", command=lambda: self._browse(self.source_dir)).grid(row=0, column=3, sticky=tk.E, pady=4)
        ttk.Label(training_tab, text="Metadata:", style="Panel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=4)
        train_meta_cb = ttk.Combobox(
            training_tab,
            textvariable=self.train_metadata_source,
            width=18,
            state="readonly",
            values=["auto", "simple", "detailed", "detailed_florence"],
        )
        train_meta_cb.grid(row=1, column=1, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="Min score:", style="Panel.TLabel").grid(row=1, column=2, sticky=tk.E, pady=4)
        train_min_score_entry = ttk.Entry(training_tab, textvariable=self.train_min_tag_score, width=8)
        train_min_score_entry.grid(row=1, column=3, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="Epochs:", style="Panel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4)
        train_epochs_spin = ttk.Spinbox(training_tab, from_=1, to=1000, width=8, textvariable=self.train_epochs)
        train_epochs_spin.grid(row=2, column=1, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="Batch:", style="Panel.TLabel").grid(row=2, column=2, sticky=tk.E, pady=4)
        train_batch_spin = ttk.Spinbox(training_tab, from_=1, to=256, width=8, textvariable=self.train_batch_size)
        train_batch_spin.grid(row=2, column=3, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="Accum:", style="Panel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=4)
        train_accum_spin = ttk.Spinbox(training_tab, from_=1, to=64, width=8, textvariable=self.train_grad_accum)
        train_accum_spin.grid(row=3, column=1, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="LR:", style="Panel.TLabel").grid(row=3, column=2, sticky=tk.E, pady=4)
        train_lr_entry = ttk.Entry(training_tab, textvariable=self.train_lr, width=10)
        train_lr_entry.grid(row=3, column=3, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="Max side:", style="Panel.TLabel").grid(row=4, column=0, sticky=tk.W, pady=4)
        train_max_side_spin = ttk.Spinbox(training_tab, from_=256, to=4096, increment=128, width=8, textvariable=self.train_max_side)
        train_max_side_spin.grid(row=4, column=1, sticky=tk.W, padx=8, pady=4)
        ttk.Label(training_tab, text="Augment:", style="Panel.TLabel").grid(row=4, column=2, sticky=tk.E, pady=4)
        train_augment_cb = ttk.Combobox(training_tab, textvariable=self.train_augment_mode, width=8, state="readonly", values=["off", "light", "full"])
        train_augment_cb.grid(row=4, column=3, sticky=tk.W, padx=8, pady=4)
        training_flags = ttk.Frame(training_tab, style="Panel.TFrame")
        training_flags.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=(5, 0))
        ttk.Checkbutton(training_flags, text="Manga adapter", variable=self.train_manga).pack(side=tk.LEFT)
        ttk.Checkbutton(training_flags, text="Fresh", variable=self.train_fresh).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(training_flags, text="Finetune", variable=self.train_finetune).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(training_flags, text="No auto optimize", variable=self.train_no_auto_optimize).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(training_flags, text="CPU", variable=self.train_cpu).pack(side=tk.LEFT, padx=(12, 0))
        self.train_btn = ttk.Button(
            training_tab,
            text="Train DINO Adapter",
            command=self._start_training,
            style="Primary.TButton",
        )
        self.train_btn.grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=(12, 0))

        side = ttk.LabelFrame(body, text="Output Folder", padding=8)
        side.grid(row=0, column=1, sticky=tk.N + tk.E + tk.W)
        side.columnconfigure(0, weight=1)
        ttk.Entry(side, textvariable=self.target_dir).grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        self.open_target_btn = ttk.Button(side, text="Open", command=self._open_target_dir, state=tk.DISABLED)
        self.open_target_btn.grid(row=1, column=0, sticky=tk.EW, padx=(0, 4), pady=2)
        self.undo_move_btn = ttk.Button(side, text="Undo Last Move", command=self._undo_last_move)
        self.undo_move_btn.grid(row=1, column=1, sticky=tk.EW, pady=2)
        self.history_btn = ttk.Button(side, text="History", command=self._open_run_history)
        self.history_btn.grid(row=2, column=0, sticky=tk.EW, padx=(0, 4), pady=2)
        ttk.Button(side, text="Clear cache", command=self._clear_cache).grid(row=2, column=1, sticky=tk.EW, pady=2)
        self.status = ttk.Label(side, text="Ready", wraplength=280)
        self.status.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(12, 2))

        progress = ttk.Frame(main, style="App.TFrame")
        progress.pack(fill=tk.X, pady=(0, 6))
        progress.columnconfigure(0, weight=1)
        progress.columnconfigure(1, weight=0)
        self.pvar = tk.DoubleVar()
        ttk.Progressbar(progress, variable=self.pvar, maximum=100).grid(row=0, column=0, sticky=tk.EW, padx=(0, 8))
        self.progress_count_label = ttk.Label(progress, text="0/0", relief=tk.SOLID, padding=(8, 4))
        self.progress_count_label.grid(row=0, column=1, sticky=tk.E)
        self.eta_label = ttk.Label(progress, textvariable=self.eta_var, foreground="#657386")
        self.eta_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        log_frame = ttk.LabelFrame(main, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.logw = tk.Text(
            log_frame,
            height=8,
            state=tk.DISABLED,
            yscrollcommand=scrollbar.set,
            bg="#0f172a",
            fg="#d7e2f0",
            insertbackground="#d7e2f0",
            relief=tk.FLAT,
            padx=8,
            pady=6,
            font="Consolas 9",
        )
        self.logw.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.logw.yview)
        self.logw.bind("<Button-3>", self._log_context_menu)
        self.logw.bind("<Control-c>", self._copy_log_selection)
        self.logw.bind("<Control-a>", self._select_all_log)

        self._tooltips = [
            Tooltip(content_min_entry, "Content minimum confidence. Typical: 0.05-0.25"),
            Tooltip(style_min_entry, "Style minimum confidence. Typical: 0.05-0.25"),
            Tooltip(group_thr_entry, "Grouping sensitivity. Lower = fewer/larger groups."),
            Tooltip(dedup_entry, "Duplicate threshold. 0.88 is balanced; higher = stricter duplicates."),
            Tooltip(camie_thr_entry, "Minimum Camie confidence for adding a tag. Also used as tag-filter threshold."),
            Tooltip(metadata_camie_thr_entry, "Minimum Camie confidence for adding tags to metadata. Lower = more tags, higher = cleaner tags."),
            Tooltip(max_tokens_spin, "Soft cap for metadata caption length from Florence-2."),
            Tooltip(max_folder_spin, "Limit metadata generation per folder. 0 = no limit."),
            Tooltip(tags_img_spin, "How many tags keep per image in metadata."),
            Tooltip(save_every_spin, "Write metadata files every N processed images. 0 = only at the end."),
            Tooltip(florence_per_folder_spin, "Maximum Florence captions per source subfolder. 0 = all images."),
            Tooltip(self.metadata_export_btn, "Write Source/metadata.json and Source/metadata_detailed.json without sorting or copying files."),
            Tooltip(max_side_spin, "Resize long side before Florence. Bigger = quality, slower."),
            Tooltip(uncertain_entry, "If prediction confidence is below this value, image goes to Needs_Review."),
            Tooltip(max_folders_spin, "Maximum number of created subfolders. 0 = unlimited."),
            Tooltip(visual_max_folders_spin, "Visual grouping folder limit. Extra clusters are merged into the most similar kept clusters. 0 = unlimited."),
            Tooltip(name_line, "Controls how aggressively similar folder names are merged or clarified."),
            Tooltip(char_line, "When scanning recursively, use source folder names and character tags as the top-level output folder."),
            Tooltip(char_min_entry, "Minimum character tag confidence for character-aware routing."),
            Tooltip(char_margin_entry, "Required confidence gap between top characters for single-character routing."),
            Tooltip(char_multi_spin, "Maximum number of character names used when Create multiple is enabled."),
            Tooltip(vram_profile_cb, "Conservative VRAM target used by auto-batch. Auto keeps a reserve below total GPU memory."),
            Tooltip(dino_batch_spin, "DINOv2 batch override. 0 = use main batch, then auto-limit for VRAM."),
            Tooltip(tagger_batch_spin, "Tagger batch override. 0 = use main batch, then auto-limit for VRAM."),
            Tooltip(vram_limit_spin, "Only used when VRAM is Custom. Set slightly below physical VRAM, e.g. 7.2 for an 8 GB card."),
            Tooltip(lmdb_mode_cb, "Use a shared image-byte LMDB cache. Auto uses it when lmdb is installed."),
            Tooltip(lmdb_line, "Shared with extract_tags.py and train_dinov.py when this folder is the same."),
            Tooltip(train_meta_cb, "auto prefers metadata_detailed.json. detailed_florence also adds Florence captions to the training text."),
            Tooltip(train_min_score_entry, "Minimum all_scores confidence used from metadata_detailed.json."),
            Tooltip(train_epochs_spin, "Maximum epochs. Early stopping can finish sooner."),
            Tooltip(train_batch_spin, "Training batch before auto optimization."),
            Tooltip(train_accum_spin, "Gradient accumulation. Effective batch = batch x accum."),
            Tooltip(train_lr_entry, "Learning rate for the adapter optimizer."),
            Tooltip(train_max_side_spin, "Downscale very large images before training transforms. 1024 is fast and enough for 224px DINO input."),
            Tooltip(train_augment_cb, "light is the fast default. full enables RandAugment and can be much slower on CPU."),
            Tooltip(blocked_entry, "Subfolders to skip completely. Use names, relative paths, or absolute paths separated by semicolon."),
        ]
        self._init_drag_drop()

    def _collect_settings_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key in SETTINGS_BOOL_KEYS + SETTINGS_STRING_KEYS + SETTINGS_NUMBER_KEYS:
            payload[key] = getattr(self, key).get()
        payload["precision"] = self.precision_var.get()
        payload["group_desc_history"] = self._group_desc_history[:20]
        payload["tag_presets"] = self.tag_presets
        payload["config_presets"] = self.config_presets
        return payload

    def _collect_run_config(self) -> dict[str, object]:
        config = self._collect_settings_payload()
        config["cancel_event"] = self.cancel_event
        config["stop_after_batch_event"] = self.stop_after_batch_event
        config["scan_status_callback"] = self._set_cache_status
        config["eta_callback"] = self._set_eta
        return config

    def _save_settings(self) -> None:
        save_settings(self._collect_settings_payload(), self.log)

    def _load_settings(self) -> None:
        data = load_settings(self.log)
        for key in SETTINGS_BOOL_KEYS + SETTINGS_STRING_KEYS + SETTINGS_NUMBER_KEYS:
            if key in data:
                getattr(self, key).set(data[key])
        if "precision" in data:
            self.precision_var.set(data["precision"])
        if "group_desc_history" in data:
            self._group_desc_history = list(data["group_desc_history"])
        if "tag_presets" in data and isinstance(data["tag_presets"], dict):
            self.tag_presets = data["tag_presets"]
            self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
        if "config_presets" in data and isinstance(data["config_presets"], dict):
            self.config_presets = data["config_presets"]
            self.config_preset_cb["values"] = ["None"] + list(self.config_presets.keys())

    def _update_status_label(self) -> None:
        self.status.config(text=get_runtime_status(self.precision_var.get()))

    def _on_precision_change(self, *_args) -> None:
        self._update_status_label()

    def _browse(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def _add_blocked_folder(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            return
        selected = Path(path).expanduser()
        source = self.source_dir.get().strip()
        value = str(selected)
        if source:
            try:
                value = str(selected.resolve().relative_to(Path(source).expanduser().resolve()))
            except Exception:
                value = str(selected)
        current = self.blocked_subfolders.get().strip()
        parts = [part.strip() for part in current.split(";") if part.strip()]
        if value not in parts:
            parts.append(value)
        self.blocked_subfolders.set("; ".join(parts))

    def _init_drag_drop(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES

            for widget in (self.root, self.drop_zone):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_files_dropped)
            self.log("Drag and drop enabled.")
        except Exception as exc:
            self.log(f"Drag and drop unavailable: {exc}")

    def _on_files_dropped(self, event) -> None:
        items = [Path(item) for item in self.root.tk.splitlist(event.data)]
        if not items:
            return
        first = items[0]
        if first.is_dir():
            self.source_dir.set(str(first))
            self.log(f"Source set from drop: {first}")
        elif first.is_file():
            self.source_dir.set(str(first.parent))
            self.log(f"Source set from dropped file folder: {first.parent}")

    def log(self, message: str) -> None:
        self.root.after(0, self._log_ui, message)

    def _log_ui(self, message: str) -> None:
        self.logw.config(state=tk.NORMAL)
        self.logw.insert(tk.END, message + "\n")
        self.logw.see(tk.END)
        self.logw.config(state=tk.DISABLED)
        self.status.config(text=message)

    def _set_cache_status(self, text: str) -> None:
        self.root.after(0, self.cache_lbl.config, {"text": text})

    def _set_eta(self, payload: dict[str, object]) -> None:
        self.root.after(0, self._set_eta_ui, payload)

    def _set_eta_ui(self, payload: dict[str, object]) -> None:
        stage = str(payload.get("stage", "")).strip()
        completed = int(payload.get("completed", 0))
        total = int(payload.get("total", 0))
        speed = float(payload.get("speed", 0.0))
        eta_text = str(payload.get("eta_text", "--"))
        gpu_text = str(payload.get("gpu_text", "")).strip()
        suffix = f" | {gpu_text}" if gpu_text else ""
        if total > 0:
            self.eta_var.set(f"{stage}: {completed}/{total} | {speed:.1f} i/s | ETA {eta_text}{suffix}")
            if hasattr(self, "progress_count_label"):
                self.progress_count_label.config(text=f"{completed}/{total}")
        else:
            self.eta_var.set(f"{stage}: ETA {eta_text}{suffix}")

    def _prog(self, value: float) -> None:
        self.root.after(0, self.pvar.set, min(max(value, 0), 100))

    def _copy_log_selection(self, _event=None) -> str:
        try:
            text = self.logw.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except tk.TclError:
            pass
        return "break"

    def _select_all_log(self, _event=None) -> str:
        self.logw.tag_add(tk.SEL, "1.0", tk.END)
        return "break"

    def _log_context_menu(self, event) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy selection", command=self._copy_log_selection)
        menu.add_command(label="Select all", command=self._select_all_log)
        menu.add_separator()
        menu.add_command(label="Copy all", command=self._copy_all_log)
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_all_log(self) -> None:
        text = self.logw.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _on_preset_select(self, _event=None) -> None:
        name = self.tag_preset_var.get()
        if name in self.tag_presets:
            preset = self.tag_presets[name]
            self.content_tags.set(preset.get("content", ""))
            self.style_tags.set(preset.get("style", ""))

    def _apply_quick_start_profile(self) -> None:
        name = self.quick_start_var.get().strip()
        profile = self.quick_start_profiles.get(name)
        if not profile:
            return
        for key, value in profile.items():
            if key == "precision_var":
                self.precision_var.set(str(value))
                continue
            if hasattr(self, key):
                getattr(self, key).set(value)
        self._update_status_label()
        self.log(f"Quick start profile applied: {name}")

    def _snapshot_config_preset(self) -> dict[str, object]:
        payload = self._collect_settings_payload()
        payload.pop("source_dir", None)
        payload.pop("target_dir", None)
        payload.pop("tag_presets", None)
        payload.pop("config_presets", None)
        return payload

    def _apply_config_payload(self, payload: dict[str, object], include_dirs: bool = False) -> None:
        for key in SETTINGS_BOOL_KEYS + SETTINGS_STRING_KEYS + SETTINGS_NUMBER_KEYS:
            if not include_dirs and key in {"source_dir", "target_dir"}:
                continue
            if key in payload and hasattr(self, key):
                getattr(self, key).set(payload[key])
        if "precision" in payload:
            self.precision_var.set(str(payload["precision"]))
        self._update_status_label()

    def _on_config_preset_select(self, _event=None) -> None:
        name = self.config_preset_var.get().strip()
        if name in self.config_presets:
            self._apply_config_payload(self.config_presets[name])
            self.log(f"Run preset applied: {name}")

    def _on_florence_profile_change(self, _event=None) -> None:
        profile = self.florence_profile.get().strip()
        defaults = {"Fast": (8, 640), "Balanced": (4, 768), "Quality": (2, 1024)}
        batch_size, max_side = defaults.get(profile, defaults["Balanced"])
        self.florence_batch_size.set(batch_size)
        self.florence_max_side.set(max_side)
        self.log(f"Florence preset: {profile} (batch={batch_size}, max_side={max_side})")

    def _save_preset(self) -> None:
        name = simpledialog.askstring("Save Preset", "Enter preset name:", parent=self.root)
        if name:
            self.tag_presets[name] = {"content": self.content_tags.get(), "style": self.style_tags.get()}
            self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
            self.tag_preset_var.set(name)
            self._save_settings()
            self.log(f"Preset '{name}' saved.")

    def _save_config_preset(self) -> None:
        name = simpledialog.askstring("Save Run Preset", "Enter run preset name:", parent=self.root)
        if not name:
            return
        self.config_presets[name] = self._snapshot_config_preset()
        self.config_preset_cb["values"] = ["None"] + list(self.config_presets.keys())
        self.config_preset_var.set(name)
        self._save_settings()
        self.log(f"Run preset '{name}' saved.")

    def _delete_preset(self) -> None:
        name = self.tag_preset_var.get()
        if name in self.tag_presets and messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
            del self.tag_presets[name]
            self.tag_preset_cb["values"] = ["None"] + list(self.tag_presets.keys())
            self.tag_preset_var.set("None")
            self._save_settings()
            self.log(f"Preset '{name}' deleted.")

    def _delete_config_preset(self) -> None:
        name = self.config_preset_var.get().strip()
        if name in self.config_presets and messagebox.askyesno("Delete Run Preset", f"Delete run preset '{name}'?"):
            del self.config_presets[name]
            self.config_preset_cb["values"] = ["None"] + list(self.config_presets.keys())
            self.config_preset_var.set("None")
            self._save_settings()
            self.log(f"Run preset '{name}' deleted.")

    def _show_desc_history(self) -> None:
        if not self._group_desc_history:
            messagebox.showinfo("History", "No history yet.")
            return
        win = tk.Toplevel(self.root)
        win.title("History")
        win.geometry("400x250")
        listbox = tk.Listbox(win, selectmode=tk.SINGLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for item in self._group_desc_history:
            listbox.insert(tk.END, item)

        def pick() -> None:
            selection = listbox.curselection()
            if selection:
                self.group_desc_var.set(self._group_desc_history[selection[0]])
            win.destroy()

        buttons = ttk.Frame(win)
        buttons.pack(pady=5)
        ttk.Button(buttons, text="Use selected", command=pick).pack(side=tk.LEFT, padx=4)

    def _clear_desc_history(self) -> None:
        if self._group_desc_history:
            self._group_desc_history.clear()
            self._save_settings()
            self.log("Semantic history cleared.")

    def _open_cluster_name_memory(self) -> None:
        memory_file = get_paths().cache_dir / "cluster_name_memory.json"
        payload = read_json(memory_file, {}, self.log)
        if not isinstance(payload, dict):
            payload = {}

        win = tk.Toplevel(self.root)
        win.title("Remembered Cluster Names")
        win.geometry("620x420")
        ttk.Label(win, text="JSON map: generated folder name -> preferred folder name").pack(anchor=tk.W, padx=10, pady=(10, 4))
        text = tk.Text(win, height=18, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        import json

        text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

        buttons = ttk.Frame(win)
        buttons.pack(fill=tk.X, padx=10, pady=(4, 10))

        def save_memory() -> None:
            try:
                value = json.loads(text.get("1.0", tk.END).strip() or "{}")
                if not isinstance(value, dict):
                    raise ValueError("Top-level JSON value must be an object.")
            except Exception as exc:
                messagebox.showerror("Cluster Names", f"Invalid JSON:\n{exc}")
                return
            cleaned = {safe_filename(str(key)): safe_filename(str(val)) for key, val in value.items() if str(key).strip() and str(val).strip()}
            if write_json(memory_file, cleaned, self.log):
                self.log(f"Cluster name memory saved: {memory_file}")
                win.destroy()

        ttk.Button(buttons, text="Save", command=save_memory).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 6))

    def _clear_cache(self) -> None:
        removed, error = clear_cache(self.log, self.lmdb_cache_dir.get().strip())
        if error:
            self.cache_lbl.config(text=f"error: {error}")
        elif removed:
            self.cache_lbl.config(text=f"cleared: {', '.join(removed)}")
        else:
            self.cache_lbl.config(text="cache empty")

    def _open_target_dir(self) -> None:
        target = self.target_dir.get().strip()
        if target and Path(target).is_dir():
            os.startfile(target)
        else:
            messagebox.showinfo("Open", "Target directory not found.")

    def _undo_last_move(self) -> None:
        target = self.target_dir.get().strip()
        if not target or not Path(target).exists():
            messagebox.showerror("Undo", "Set valid Target first.")
            return
        proceed = messagebox.askyesno("Undo Last Move", "Restore files from the last move journal for this Target?")
        if not proceed:
            return
        result = undo_last_move_run(target, log_callback=self.log)
        if result.get("ok"):
            messagebox.showinfo("Undo", str(result.get("message", "Undo completed.")))
        else:
            messagebox.showwarning("Undo", str(result.get("message", "Undo finished with warnings.")))

    def _open_run_history(self) -> None:
        history = get_run_history(limit=100)
        win = tk.Toplevel(self.root)
        win.title("Run History")
        win.geometry("980x500")

        cols = ("timestamp", "ok", "mode", "processed", "saved", "needs_review", "filtered", "errors", "elapsed")
        tree = ttk.Treeview(win, columns=cols, show="headings", selectmode="browse")
        headings = {
            "timestamp": "Time",
            "ok": "OK",
            "mode": "Mode",
            "processed": "Processed",
            "saved": "Saved",
            "needs_review": "NeedsReview",
            "filtered": "Filtered",
            "errors": "Errors",
            "elapsed": "Sec",
        }
        for key in cols:
            tree.heading(key, text=headings[key])
            tree.column(key, width=90 if key != "timestamp" else 170, anchor=tk.CENTER)
        tree.column("timestamp", anchor=tk.W)

        ysb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(win, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH)
        details = tk.Text(right, width=42, height=22, state=tk.DISABLED)
        details.pack(fill=tk.BOTH, expand=True)
        load_btn = ttk.Button(right, text="Load settings from run")
        load_btn.pack(fill=tk.X, pady=(8, 0))

        rows: list[dict[str, object]] = []
        for item in history:
            rows.append(item)
            tree.insert(
                "",
                tk.END,
                values=(
                    item.get("timestamp", ""),
                    "yes" if item.get("ok") else "no",
                    item.get("mode", ""),
                    item.get("processed", 0),
                    item.get("saved", 0),
                    item.get("needs_review", 0),
                    item.get("filtered", 0),
                    item.get("errors", 0),
                    f"{float(item.get('elapsed', 0.0)):.1f}",
                ),
            )

        def set_details(text: str) -> None:
            details.config(state=tk.NORMAL)
            details.delete("1.0", tk.END)
            details.insert("1.0", text)
            details.config(state=tk.DISABLED)

        def on_select(_event=None) -> None:
            selected = tree.selection()
            if not selected:
                return
            idx = tree.index(selected[0])
            if idx >= len(rows):
                return
            current = rows[idx]
            prev = rows[idx + 1] if idx + 1 < len(rows) else None
            lines = [
                f"Run: {current.get('timestamp', '')}",
                f"Status: {'OK' if current.get('ok') else 'Failed'}",
                f"Source: {current.get('source', '')}",
                f"Target: {current.get('target', '')}",
                "",
                f"Processed: {current.get('processed', 0)}",
                f"Saved: {current.get('saved', 0)}",
                f"Needs Review: {current.get('needs_review', 0)}",
                f"Filtered: {current.get('filtered', 0)}",
                f"Errors: {current.get('errors', 0)}",
                f"Elapsed: {float(current.get('elapsed', 0.0)):.1f}s",
                "",
                "Comparison vs previous run:",
            ]
            if prev:
                delta_saved = int(current.get("saved", 0)) - int(prev.get("saved", 0))
                delta_filtered = int(current.get("filtered", 0)) - int(prev.get("filtered", 0))
                delta_errors = int(current.get("errors", 0)) - int(prev.get("errors", 0))
                delta_elapsed = float(current.get("elapsed", 0.0)) - float(prev.get("elapsed", 0.0))
                lines.extend(
                    [
                        f"Saved: {delta_saved:+d}",
                        f"Filtered: {delta_filtered:+d}",
                        f"Errors: {delta_errors:+d}",
                        f"Elapsed: {delta_elapsed:+.1f}s",
                    ]
                )
            else:
                lines.append("No previous run for comparison.")
            lines.extend(["", f"Message: {current.get('message', '')}"])
            if isinstance(current.get("settings"), dict):
                lines.extend(["", "Settings snapshot: available"])
            set_details("\n".join(lines))

        def load_selected_settings() -> None:
            selected = tree.selection()
            if not selected:
                return
            idx = tree.index(selected[0])
            if idx >= len(rows):
                return
            settings = rows[idx].get("settings")
            if not isinstance(settings, dict):
                messagebox.showinfo("History", "This run has no saved settings snapshot.")
                return
            self._apply_config_payload(settings, include_dirs=True)
            self.log(f"Loaded settings from run: {rows[idx].get('timestamp', '')}")
            win.destroy()

        tree.bind("<<TreeviewSelect>>", on_select)
        load_btn.config(command=load_selected_settings)
        if rows:
            first = tree.get_children()
            if first:
                tree.selection_set(first[0])
                on_select()

    def _do_cancel(self) -> None:
        self.cancel_event.set()
        self.stop_after_batch_event.set()
        if self.training_process is not None and self.training_process.poll() is None:
            if self.training_stop_file is not None:
                try:
                    self.training_stop_file.parent.mkdir(parents=True, exist_ok=True)
                    self.training_stop_file.write_text("cancel", encoding="utf-8")
                    self.log(f"Training stop requested: {self.training_stop_file}")
                except Exception as exc:
                    self.log(f"Warning: could not write training stop file: {exc}")
            try:
                self.training_process.terminate()
            except Exception as exc:
                self.log(f"Warning: could not terminate training process: {exc}")
        self.cancel_btn.config(state=tk.DISABLED)
        self.stop_after_btn.config(state=tk.DISABLED)
        self.log("Hard cancel requested. Stopping immediately...")

    def _stop_after_current_batch(self) -> None:
        self.stop_after_batch_event.set()
        self.stop_after_btn.config(state=tk.DISABLED)
        self.log("Will stop after current batch...")

    def _start(self) -> None:
        source = self.source_dir.get().strip()
        target = self.target_dir.get().strip()
        if not source or not target:
            messagebox.showerror("Error", "Specify both Source and Target.")
            return
        if not Path(source).is_dir():
            messagebox.showerror("Error", f"Source not found:\n{source}")
            return
        active_criteria = any(
            [
                self.sort_ai_human.get(),
                self.sort_content.get(),
                self.sort_style.get(),
                self.sort_grouping.get(),
                self.sort_dedup.get(),
                self.sort_tagger_filter.get(),
                self.gen_metadata.get(),
                self.character_aware_recursive.get(),
            ]
        )
        if not active_criteria:
            messagebox.showwarning("Warning", "Select at least one criterion.")
            return
        if self.move_files.get() and not self.dry_run.get():
            proceed = messagebox.askyesno(
                "Move Mode",
                "Files will be MOVED (not copied).\nOriginal files will no longer exist in Source.\nUndo is available via 'Undo Last Move'.\n\nContinue?",
            )
            if not proceed:
                return

        est_gb = 0.0
        if self.sort_ai_human.get():
            est_gb += MODEL_SIZES_GB["ai"]
        if self.sort_ai_human.get() and self.use_sdxl_vote.get():
            est_gb += MODEL_SIZES_GB["sdxl"]
        if self.sort_content.get() or self.sort_style.get() or self.sort_grouping.get():
            est_gb += MODEL_SIZES_GB["siglip"]
        if self.sort_grouping.get() or self.sort_dedup.get():
            est_gb += MODEL_SIZES_GB["dino"]
        self.log(f"Estimated model memory footprint: ~{format_size(est_gb * 1024**3, precision=1)}")
        if self.dry_run.get():
            self.log("Dry run enabled: no files will be copied, moved, or metadata-written.")

        config = self._collect_run_config()
        self._save_settings()
        self.cancel_event = threading.Event()
        self.stop_after_batch_event = threading.Event()
        config["cancel_event"] = self.cancel_event
        config["stop_after_batch_event"] = self.stop_after_batch_event
        self._run_start_time = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.stop_after_btn.config(state=tk.NORMAL)
        self.open_target_btn.config(state=tk.DISABLED)
        self.pvar.set(0)
        if hasattr(self, "progress_count_label"):
            self.progress_count_label.config(text="0/0")
        self.eta_var.set("ETA: calculating...")
        self.logw.config(state=tk.NORMAL)
        self.logw.delete("1.0", tk.END)
        self.logw.config(state=tk.DISABLED)
        self.worker_thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def _run_worker(self, config: dict[str, object]) -> None:
        result = run_pipeline(config, progress_callback=self._prog, log_callback=self.log)
        self.root.after(0, self._finish, result)

    def _start_metadata_export(self) -> None:
        source = self.source_dir.get().strip()
        if not source:
            messagebox.showerror("Error", "Specify Source first.")
            return
        if not Path(source).is_dir():
            messagebox.showerror("Error", f"Source not found:\n{source}")
            return
        config = self._collect_run_config()
        config["gen_metadata"] = True
        self._save_settings()
        self.cancel_event = threading.Event()
        self.stop_after_batch_event = threading.Event()
        config["cancel_event"] = self.cancel_event
        config["stop_after_batch_event"] = self.stop_after_batch_event
        self._run_start_time = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.metadata_export_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.stop_after_btn.config(state=tk.DISABLED)
        self.open_target_btn.config(state=tk.DISABLED)
        self.pvar.set(0)
        if hasattr(self, "progress_count_label"):
            self.progress_count_label.config(text="0/0")
        self.eta_var.set("Metadata export: calculating...")
        self.logw.config(state=tk.NORMAL)
        self.logw.delete("1.0", tk.END)
        self.logw.config(state=tk.DISABLED)
        self.worker_thread = threading.Thread(target=self._run_metadata_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def _run_metadata_worker(self, config: dict[str, object]) -> None:
        result = run_metadata_export(config, progress_callback=self._prog, log_callback=self.log)
        self.root.after(0, self._finish, result)

    def _start_training(self) -> None:
        source = self.source_dir.get().strip()
        if not source:
            messagebox.showerror("Error", "Specify Dataset/Source first.")
            return
        if not Path(source).is_dir():
            messagebox.showerror("Error", f"Dataset not found:\n{source}")
            return
        self._save_settings()
        self.cancel_event = threading.Event()
        self.stop_after_batch_event = threading.Event()
        self._run_start_time = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        if hasattr(self, "metadata_export_btn"):
            self.metadata_export_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.stop_after_btn.config(state=tk.DISABLED)
        self.open_target_btn.config(state=tk.DISABLED)
        self.pvar.set(0)
        if hasattr(self, "progress_count_label"):
            self.progress_count_label.config(text="0/0")
        self.eta_var.set("DINO training: starting...")
        self.logw.config(state=tk.NORMAL)
        self.logw.delete("1.0", tk.END)
        self.logw.config(state=tk.DISABLED)
        training_config = {
            "source_dir": source,
            "epochs": max(1, self.train_epochs.get()),
            "batch_size": max(1, self.train_batch_size.get()),
            "grad_accum": max(1, self.train_grad_accum.get()),
            "lr": float(self.train_lr.get()),
            "lmdb_mode": self.lmdb_mode.get(),
            "lmdb_cache_dir": self.lmdb_cache_dir.get().strip(),
            "metadata_source": self.train_metadata_source.get(),
            "min_tag_score": float(self.train_min_tag_score.get()),
            "train_max_side": max(256, self.train_max_side.get()),
            "augment": self.train_augment_mode.get(),
            "rebuild_lmdb": self.rebuild_lmdb.get(),
            "no_auto_optimize": self.train_no_auto_optimize.get(),
            "cpu": self.train_cpu.get(),
            "manga": self.train_manga.get(),
            "finetune": self.train_finetune.get(),
            "fresh": self.train_fresh.get(),
        }
        self.training_stop_file = get_paths().cache_dir / "train_dinov_stop.flag"
        try:
            self.training_stop_file.unlink(missing_ok=True)
        except Exception:
            pass
        self.worker_thread = threading.Thread(target=self._run_training_worker, args=(training_config,), daemon=True)
        self.worker_thread.start()

    def _run_training_worker(self, training_config: dict[str, object]) -> None:
        project_root = get_paths().project_root
        cmd = [
            sys.executable,
            str(project_root / "train_dinov.py"),
            "--data_dir",
            str(training_config["source_dir"]),
            "--base_weights_dir",
            str(get_paths().weights_dir),
            "--epochs",
            str(training_config["epochs"]),
            "--batch_size",
            str(training_config["batch_size"]),
            "--grad_accum",
            str(training_config["grad_accum"]),
            "--lr",
            str(training_config["lr"]),
            "--lmdb",
            str(training_config["lmdb_mode"]),
            "--lmdb_dir",
            str(training_config["lmdb_cache_dir"]),
            "--metadata_source",
            str(training_config["metadata_source"]),
            "--min_tag_score",
            str(training_config["min_tag_score"]),
            "--train_max_side",
            str(training_config["train_max_side"]),
            "--augment",
            str(training_config["augment"]),
            "--stop_file",
            str(self.training_stop_file or (get_paths().cache_dir / "train_dinov_stop.flag")),
            "--no_tqdm",
            "--log_every",
            "100",
        ]
        if training_config["rebuild_lmdb"]:
            cmd.append("--rebuild_lmdb")
        if training_config["no_auto_optimize"]:
            cmd.append("--no_auto_optimize")
        if training_config["cpu"]:
            cmd.append("--cpu")
        if training_config["manga"]:
            cmd.append("--manga")
        if training_config["finetune"]:
            cmd.append("--finetune")
        if training_config["fresh"] and not training_config["finetune"]:
            cmd.append("--fresh")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        result: dict[str, object] = {"ok": False, "message": "DINO training did not start."}
        try:
            self.log("Starting DINO training...")
            self.log("Command: " + " ".join(f'"{part}"' if " " in part else part for part in cmd))
            self.training_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )
            assert self.training_process.stdout is not None
            total_epochs = max(1, int(training_config["epochs"]))
            for raw_line in self.training_process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                self.log(line)
                match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                train_match = re.search(
                    r"TRAIN epoch=(\d+)/(\d+) step=(\d+)/(\d+).*?ETA=([^|]+)",
                    line,
                )
                if train_match:
                    epoch = int(train_match.group(1))
                    total_epochs = max(1, int(train_match.group(2)))
                    step = int(train_match.group(3))
                    total_steps = max(1, int(train_match.group(4)))
                    eta_text = train_match.group(5).strip()
                    overall = ((epoch - 1) + step / total_steps) / total_epochs * 100.0
                    self._prog(min(100.0, overall))
                    self.root.after(
                        0,
                        self.eta_var.set,
                        f"DINO training: epoch {epoch}/{total_epochs}, step {step}/{total_steps}, ETA {eta_text}",
                    )
                    self.root.after(0, self.progress_count_label.config, {"text": f"{epoch}/{total_epochs}"})
                if match:
                    epoch = int(match.group(1))
                    total_epochs = max(1, int(match.group(2)))
                    self._prog(min(100.0, epoch / total_epochs * 100.0))
                    self.root.after(0, self.eta_var.set, f"DINO training: epoch {epoch}/{total_epochs}")
                    self.root.after(0, self.progress_count_label.config, {"text": f"{epoch}/{total_epochs}"})
            code = self.training_process.wait()
            if self.cancel_event.is_set():
                result = {"ok": False, "message": "DINO training cancelled."}
            elif code == 0:
                result = {"ok": True, "message": "DINO training finished. Adapter saved in data/weights."}
            else:
                result = {"ok": False, "message": f"DINO training failed with exit code {code}."}
        except Exception as exc:
            result = {"ok": False, "message": f"DINO training failed: {exc}"}
        finally:
            self.training_process = None
            if self.training_stop_file is not None:
                try:
                    self.training_stop_file.unlink(missing_ok=True)
                except Exception:
                    pass
        self.root.after(0, self._finish, result)

    def _finish(self, result: dict[str, object]) -> None:
        if self._run_start_time is not None:
            elapsed = time.time() - self._run_start_time
            mins, secs = divmod(int(elapsed), 60)
            self.log(f"Total time: {mins}m {secs}s")
            self._run_start_time = None
        self.log(str(result.get("message", "")))
        if result.get("undo_available"):
            self.log("Undo is available: use 'Undo Last Move' button if needed.")
        self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, "train_btn"):
            self.train_btn.config(state=tk.NORMAL)
        if hasattr(self, "metadata_export_btn"):
            self.metadata_export_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.stop_after_btn.config(state=tk.DISABLED)
        self.open_target_btn.config(state=tk.NORMAL)
        self.eta_var.set("ETA: --")
        if hasattr(self, "progress_count_label"):
            self.progress_count_label.config(text="0/0")
        self._update_status_label()

    def _open_ai_search(self) -> None:
        target = self.target_dir.get().strip()
        if not target or not Path(target).exists():
            messagebox.showerror("Error", "Please set a valid Target directory first.")
            return

        search_win = tk.Toplevel(self.root)
        search_win.title("Local AI Search")
        search_win.geometry("1000x620")
        search_win.transient(self.root)
        search_win.grab_set()

        ttk.Label(search_win, text="Search within cached SigLIP embeddings", font=("", 10, "bold")).pack(pady=10)
        query_frame = ttk.Frame(search_win)
        query_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(query_frame, text="Search Query:").pack(side=tk.LEFT)
        query_var = tk.StringVar()
        query_entry = ttk.Entry(query_frame, textvariable=query_var, width=54)
        query_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(search_win, text="Example: 'a red apple on a white table'").pack(pady=2)

        top_k_frame = ttk.Frame(search_win)
        top_k_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(top_k_frame, text="Top Results:").pack(side=tk.LEFT)
        top_k_var = tk.IntVar(value=20)
        ttk.Spinbox(top_k_frame, from_=1, to=1000, width=5, textvariable=top_k_var).pack(side=tk.LEFT, padx=10)
        search_button = ttk.Button(top_k_frame, text="Search", width=12)
        search_button.pack(side=tk.LEFT, padx=(10, 0))

        body = ttk.Frame(search_win)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(body, width=360)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))

        ttk.Label(left, text="Results (multi-select with Ctrl/Shift):").pack(anchor=tk.W)
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        y_scroll.pack(side=tk.LEFT, fill=tk.Y)
        listbox.config(yscrollcommand=y_scroll.set)
        ttk.Label(left, text="Thumbnails:").pack(anchor=tk.W)
        thumb_canvas = tk.Canvas(left, height=132, bg="#f8fafc", highlightthickness=1, highlightbackground="#cfd7e6")
        thumb_canvas.pack(fill=tk.X, pady=(4, 0))
        thumb_inner = ttk.Frame(thumb_canvas)
        thumb_window = thumb_canvas.create_window((0, 0), window=thumb_inner, anchor="nw")
        thumb_x = ttk.Scrollbar(left, orient=tk.HORIZONTAL, command=thumb_canvas.xview)
        thumb_x.pack(fill=tk.X)
        thumb_canvas.configure(xscrollcommand=thumb_x.set)

        def refresh_thumb_scroll(_event=None) -> None:
            thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all"))
            thumb_canvas.itemconfigure(thumb_window, height=thumb_canvas.winfo_height())

        thumb_inner.bind("<Configure>", refresh_thumb_scroll)

        ttk.Label(right, text="Preview").pack(anchor=tk.W)
        preview_label = ttk.Label(right, text="No image selected")
        preview_label.pack(fill=tk.BOTH, expand=True, pady=(6, 6))
        preview_info = ttk.Label(right, text="", justify=tk.LEFT, foreground="gray")
        preview_info.pack(fill=tk.X)

        controls = ttk.Frame(search_win)
        controls.pack(fill=tk.X, padx=20, pady=(0, 10))
        result_lbl = ttk.Label(controls, text="", foreground="blue")
        result_lbl.pack(side=tk.LEFT)
        copy_selected_btn = ttk.Button(controls, text="Copy Selected", state=tk.DISABLED)
        copy_selected_btn.pack(side=tk.RIGHT)
        copy_all_btn = ttk.Button(controls, text="Copy All", state=tk.DISABLED)
        copy_all_btn.pack(side=tk.RIGHT, padx=(0, 6))

        def perform_search() -> None:
            query = query_var.get().strip()
            if not query:
                return
            search_button.config(state=tk.DISABLED)
            copy_selected_btn.config(state=tk.DISABLED)
            copy_all_btn.config(state=tk.DISABLED)
            listbox.delete(0, tk.END)
            preview_label.config(image="", text="No image selected")
            preview_info.config(text="")
            self._search_results = []
            self._search_preview_image = None
            self._search_thumb_images = []
            for child in thumb_inner.winfo_children():
                child.destroy()
            result_lbl.config(text="Loading model and encoding query...")

            def worker() -> None:
                result = run_ai_search(
                    {
                        "query": query,
                        "target_dir": target,
                        "top_k": top_k_var.get(),
                        "copy_results": False,
                        "precision": self.precision_var.get(),
                        "optimize_models": self.optimize_models.get(),
                        "tagger_engine_var": self.tagger_engine_var.get(),
                        "use_dino_adapter": self.use_dino_adapter.get(),
                        "group_sequences": self.group_sequences.get(),
                        "camie_threshold": self.camie_threshold.get(),
                        "florence_mode": self.florence_mode.get(),
                        "florence_char": self.florence_char.get(),
                        "florence_profile": self.florence_profile.get(),
                        "florence_batch_size": self.florence_batch_size.get(),
                        "florence_max_side": self.florence_max_side.get(),
                        "auto_vram_batch": self.auto_vram_batch.get(),
                        "unload_inactive_models": self.unload_inactive_models.get(),
                        "vram_profile": self.vram_profile.get(),
                        "vram_limit_gb": self.vram_limit_gb.get(),
                    },
                    log_callback=self.log,
                )

                def apply_result() -> None:
                    if result.get("ok"):
                        entries = result.get("results", [])
                        self._search_results = entries if isinstance(entries, list) else []
                        for item in self._search_results:
                            rank = int(item.get("rank", 0))
                            score = float(item.get("score", 0.0))
                            path = Path(str(item.get("path", "")))
                            listbox.insert(tk.END, f"{rank:03d} | {score:.3f} | {path.name}")
                        build_thumbnails()
                        result_lbl.config(text=f"Found {len(self._search_results)} matches", foreground="green")
                        if self._search_results:
                            copy_selected_btn.config(state=tk.NORMAL)
                            copy_all_btn.config(state=tk.NORMAL)
                    else:
                        result_lbl.config(text=str(result.get("message", "Search failed")), foreground="red")
                    search_button.config(state=tk.NORMAL)

                self.root.after(0, apply_result)

            threading.Thread(target=worker, daemon=True).start()

        def select_result(index: int) -> None:
            if index >= len(self._search_results):
                return
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            listbox.see(index)
            show_preview()

        def build_thumbnails() -> None:
            self._search_thumb_images = []
            for child in thumb_inner.winfo_children():
                child.destroy()
            for idx, item in enumerate(self._search_results):
                path = Path(str(item.get("path", "")))
                rank = int(item.get("rank", idx + 1))
                score = float(item.get("score", 0.0))
                tile = ttk.Frame(thumb_inner, padding=3)
                tile.grid(row=0, column=idx, padx=3, pady=3, sticky=tk.N)
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        image.thumbnail((86, 86), Image.Resampling.BILINEAR)
                        photo = ImageTk.PhotoImage(image)
                except Exception:
                    photo = None
                if photo is not None:
                    self._search_thumb_images.append(photo)
                    btn = ttk.Button(tile, image=photo, command=lambda i=idx: select_result(i))
                    btn.pack()
                else:
                    ttk.Button(tile, text="missing", width=10, command=lambda i=idx: select_result(i)).pack()
                ttk.Label(tile, text=f"{rank:03d}  {score:.2f}").pack()
            refresh_thumb_scroll()

        def show_preview(_event=None) -> None:
            selected = listbox.curselection()
            if not selected:
                return
            idx = selected[0]
            if idx >= len(self._search_results):
                return
            item = self._search_results[idx]
            path = Path(str(item.get("path", "")))
            score = float(item.get("score", 0.0))
            if not path.exists():
                preview_label.config(image="", text="File missing")
                preview_info.config(text=str(path))
                return
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((340, 340), Image.Resampling.BILINEAR)
                    photo = ImageTk.PhotoImage(image)
                self._search_preview_image = photo
                preview_label.config(image=photo, text="")
                preview_info.config(text=f"{path.name}\nscore={score:.3f}\n{path}")
            except Exception as exc:
                preview_label.config(image="", text=f"Preview failed: {exc}")
                preview_info.config(text=str(path))

        def copy_selection(copy_all: bool = False) -> None:
            query = query_var.get().strip()
            if not query:
                return
            if not self._search_results:
                return
            indices = list(range(len(self._search_results))) if copy_all else list(listbox.curselection())
            if not indices:
                messagebox.showinfo("Copy", "Select at least one result.")
                return
            result_dir = Path(target) / "AI_Search_Results" / safe_filename(query)[:50]
            result_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for idx in indices:
                if idx >= len(self._search_results):
                    continue
                item = self._search_results[idx]
                path = Path(str(item.get("path", "")))
                if not path.exists():
                    continue
                rank = int(item.get("rank", idx + 1))
                score = float(item.get("score", 0.0))
                dst = result_dir / f"{rank:03d}_{score:.2f}{path.suffix}"
                counter = 1
                while dst.exists():
                    dst = result_dir / f"{rank:03d}_{score:.2f}_{counter}{path.suffix}"
                    counter += 1
                shutil.copy2(path, dst)
                copied += 1
            result_lbl.config(text=f"Copied {copied} files to {result_dir}", foreground="green")

        listbox.bind("<<ListboxSelect>>", show_preview)
        query_entry.bind("<Return>", lambda _event: perform_search())
        search_button.config(command=perform_search)
        copy_selected_btn.config(command=lambda: copy_selection(False))
        copy_all_btn.config(command=lambda: copy_selection(True))

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()
