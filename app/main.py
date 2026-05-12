from __future__ import annotations

import argparse
import os
import tkinter as tk

from .utils import install_windows_dpi_awareness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image Sorter GUI")
    parser.add_argument("--engine", type=str, choices=["camie", "wd"], help="Force tagger engine (camie or wd)")
    parser.add_argument("--skip-model-check", action="store_true", help="Skip automatic model presence check at startup")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = build_parser()
    args = parser.parse_args(argv)

    install_windows_dpi_awareness()
    if not args.skip_model_check:
        from scripts.download_models import check_and_download_all

        check_and_download_all()

    from .ui import ImageSorterUI

    try:
        from tkinterdnd2 import TkinterDnD

        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    ui = ImageSorterUI(root)
    if args.engine:
        ui.tagger_engine_var.set("Camie-Tagger-v2" if args.engine == "camie" else "WD EVA02 v3")
    ui._update_status_label()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
