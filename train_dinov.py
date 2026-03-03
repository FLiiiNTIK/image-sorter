"""
train_dinov.py — DINOv2 Adapter Training (multi-scale)

v2.0 — Compatible with sort_images_app.py v2.0:
  - Multi-scale DINOv2: CLS + spatial pooling = 1536d input
  - Adapter: hidden_dim=512, dropout=0.3 (matches main app)
  - OneCycleLR scheduler with warmup for better convergence
  - Gradient accumulation for effective large-batch training
  - Enhanced augmentation (RandAugment + RandomErasing)
  - EMA (Exponential Moving Average) for stability
  - Cosine similarity monitoring on validation
"""

import os
import sys
import json
import glob
import copy
import logging
import argparse
import time
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from tqdm import tqdm

# ─────────────────────────── Default Configuration ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    "data_dir": os.path.join(SCRIPT_DIR, "training_data"),
    "base_weights_dir": os.path.join(SCRIPT_DIR, "weights"),
    "cache_file": "text_emb_cache.pt",
    "dino_path": os.path.join(SCRIPT_DIR, "dinov2-base"),
    "text_model": os.path.join(SCRIPT_DIR, "all-MiniLM-L6-v2"),
    # Adapter dims — MUST match sort_images_app.py DINOv2Adapter
    "input_dim": 1536,       # CLS(768) + spatial(768) = 1536 (multi-scale)
    "output_dim": 384,
    "hidden_dim": 512,       # Matching sort_images_app.py
    "dropout": 0.3,          # Matching sort_images_app.py
    # Training
    "batch_size": 32,
    "grad_accum_steps": 2,   # Effective batch = batch_size x grad_accum
    "epochs": 30,            # Early stopping will halt earlier
    "lr": 3e-4,              # OneCycleLR max_lr
    "weight_decay": 0.05,
    "num_workers": 4,
    # Loss
    "cosine_loss_weight": 0.7,
    "mse_loss_weight": 0.3,
    "l2_lambda": 1e-3,
    # Regularization
    "val_split": 0.15,
    "early_stopping_patience": 5,
    "ema_decay": 0.999,      # EMA smoothing factor
    "label_noise": 0.02,     # Gaussian noise on target embeddings
}

# ──────────────────────────── Logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ───────────────────────────────── Dataset ───────────────────────────────────────
class ImageTextDataset(Dataset):
    """
    Loads (image, text description) pairs from JSON files.
    Caches text embeddings to disk.
    """

    IMG_KEYS = ("image_path", "image", "file_path", "path", "img", "filename", "source")
    TXT_KEYS = ("description", "caption", "text", "label", "annotation")
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

    def __init__(self, data_dir: str, text_model_name: str, cache_file: str, device="cpu"):
        self.data_dir = data_dir
        # Enhanced augmentation: RandAugment + RandomErasing
        self.transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=6),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        self.transform = self.transform_train
        self.is_train = True

        log.info(f"Scanning JSON files in: {data_dir}")
        self.samples = self._scan(data_dir)
        log.info(f"Found {len(self.samples)} valid (image + text) pairs.")

        if not self.samples:
            return

        self.cached_embeddings = self._load_or_build_cache(
            text_model_name, cache_file, device
        )

    def _scan(self, data_dir: str) -> list:
        """Recursively find JSON files and collect valid pairs."""
        samples = []
        json_files = glob.glob(
            os.path.join(data_dir, "**", "*.json"), recursive=True
        )
        log.info(f"JSON files found: {len(json_files)}")

        skipped = 0
        found = 0
        for j_path in json_files:
            try:
                with open(j_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log.warning(f"Could not read {j_path}: {e}")
                continue

            # ── Format WD: metadata.json from WD Tagger ──
            # {"folder": ..., "total_images": N, "sampled": N, "images": {fname: {general_tags, ...}}}
            if (
                isinstance(data, dict)
                and "images" in data
                and isinstance(data.get("images"), dict)
                and "total_images" in data
            ):
                parent = os.path.dirname(j_path)
                images_section = data["images"]

                for fname, info in images_section.items():
                    if not isinstance(info, dict):
                        skipped += 1
                        continue

                    # Build description from tags
                    tags = []
                    for key in ("character_tags", "general_tags"):
                        tag_list = info.get(key, [])
                        if isinstance(tag_list, list):
                            tags.extend(tag_list)

                    if not tags:
                        # Fallback: use keys from all_scores
                        scores = info.get("all_scores", {})
                        if isinstance(scores, dict):
                            sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                            tags = [t for t, s in sorted_tags[:15] if s > 0.1]

                    if not tags:
                        skipped += 1
                        continue

                    desc = ", ".join(tags[:20])  # Cap at 20 tags

                    img_path = os.path.join(parent, fname)
                    img_path = os.path.normpath(img_path)
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue

                    samples.append({"image_path": img_path, "description": desc})
                    found += 1
                continue

            # ── Format 0: metadata.json from sort_images_app (legacy, with description) ──
            if (
                isinstance(data, dict)
                and "sample_files" in data
                and "description" in data
            ):
                parent = os.path.dirname(j_path)
                sample_files = data["sample_files"]
                descriptions = [d.strip() for d in data["description"].split(";") if d.strip()]
                folder_desc = data.get("description", "")

                all_imgs = []
                for ext in self.IMG_EXTS:
                    all_imgs.extend(glob.glob(os.path.join(parent, f"*{ext}")))

                if not all_imgs:
                    all_imgs = [os.path.join(parent, fn) for fn in sample_files]

                caption_map = {}
                if len(descriptions) == len(sample_files):
                    for fn, cap in zip(sample_files, descriptions):
                        caption_map[fn] = cap
                elif descriptions:
                    for fn in sample_files:
                        caption_map[fn] = folder_desc

                for img_path in all_imgs:
                    img_path = os.path.normpath(img_path)
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    fname = os.path.basename(img_path)
                    desc = caption_map.get(fname, folder_desc)
                    if not desc:
                        skipped += 1
                        continue
                    samples.append({"image_path": img_path, "description": desc})
                    found += 1
                continue

            # ── Format 1: {filename: {caption, source}} ──
            if (
                isinstance(data, dict)
                and data
                and all(isinstance(v, dict) for v in data.values())
            ):
                for filename, record in data.items():
                    description = self._extract_key(record, self.TXT_KEYS)
                    img_path = record.get("source") or record.get("image_path") or record.get("path")

                    if not img_path:
                        parent = os.path.dirname(j_path)
                        candidate = os.path.join(parent, filename)
                        if os.path.exists(candidate):
                            img_path = candidate

                    if not img_path or not description:
                        skipped += 1
                        continue

                    img_path = os.path.normpath(img_path)
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue

                    samples.append({"image_path": img_path, "description": description})
                    found += 1
                continue

            # ── Formats 2 & 3: list or single object ──
            records = data if isinstance(data, list) else [data]
            for record in records:
                if not isinstance(record, dict):
                    skipped += 1
                    continue

                img_path = self._extract_key(record, self.IMG_KEYS)
                description = self._extract_key(record, self.TXT_KEYS)

                if not img_path:
                    base = os.path.splitext(j_path)[0]
                    for ext in self.IMG_EXTS:
                        if os.path.exists(base + ext):
                            img_path = base + ext
                            break

                if not img_path or not description:
                    skipped += 1
                    continue

                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(
                        os.path.join(os.path.dirname(j_path), img_path)
                    )

                img_path = os.path.normpath(img_path)
                if not os.path.exists(img_path):
                    skipped += 1
                    continue

                samples.append({"image_path": img_path, "description": description})
                found += 1

        log.info(f"Records added: {found}")
        if skipped:
            log.warning(f"Skipped (missing file or description): {skipped}")
        return samples

    @staticmethod
    def _extract_key(d: dict, keys: tuple):
        for k in keys:
            if k in d and d[k]:
                return d[k]
        return None

    def _load_or_build_cache(self, model_name: str, cache_file: str, device) -> dict:
        """Load cache from disk if valid, otherwise rebuild."""
        all_descriptions = [s["description"] for s in self.samples]
        data_hash = hashlib.md5(
            json.dumps(sorted(set(all_descriptions)), ensure_ascii=False).encode()
        ).hexdigest()

        if os.path.exists(cache_file):
            log.info(f"Cache file found: {cache_file}. Checking validity...")
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=False)
                if saved.get("hash") == data_hash:
                    log.info("Cache is valid. Loading...")
                    return saved["embeddings"]
                else:
                    log.info("Cache is stale. Rebuilding...")
            except Exception as e:
                log.warning(f"Error reading cache: {e}. Rebuilding...")

        embeddings = self._build_cache(model_name, all_descriptions, device)
        torch.save({"hash": data_hash, "embeddings": embeddings}, cache_file)
        log.info(f"Cache saved: {cache_file}")
        return embeddings

    @staticmethod
    def _build_cache(model_name: str, descriptions: list, device) -> dict:
        log.info(f"Loading SentenceTransformer ({model_name})...")
        text_model = SentenceTransformer(model_name, device=str(device))
        text_model.eval()

        unique = list(set(descriptions))
        log.info(f"Computing embeddings for {len(unique)} unique descriptions...")
        embeddings = {}

        batch_size = 256
        for i in tqdm(range(0, len(unique), batch_size), desc="Embedding cache"):
            batch = unique[i : i + batch_size]
            with torch.no_grad():
                vecs = text_model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                for desc, vec in zip(batch, vecs):
                    embeddings[desc] = vec.cpu()

        del text_model
        torch.cuda.empty_cache()
        return embeddings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image = self.transform(image)
        except (UnidentifiedImageError, OSError):
            image = torch.zeros((3, 224, 224))

        text_emb = self.cached_embeddings[sample["description"]]
        return image, text_emb

    def set_train(self):
        self.is_train = True
        self.transform = self.transform_train

    def set_val(self):
        self.is_train = False
        self.transform = self.transform_val


# ──────────────────────────────── Adapter MLP ────────────────────────────────────
class DINOv2Adapter(nn.Module):
    """
    MLP adapter: projects multi-scale DINOv2 (1536d) -> text embedding space.
    Architecture MUST match sort_images_app.py.
    """

    def __init__(self, input_dim: int = 1536, output_dim: int = 384,
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


# ─────────────────────────── Combined Loss ────────────────────────────────
class CombinedLoss(nn.Module):
    """CosineEmbeddingLoss + MSELoss + L2 regularization."""

    def __init__(self, cosine_w: float = 0.7, mse_w: float = 0.3, l2_lambda: float = 1e-3):
        super().__init__()
        self.cosine_w = cosine_w
        self.mse_w = mse_w
        self.l2_lambda = l2_lambda
        self.cosine = nn.CosineEmbeddingLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = F.normalize(target, p=2, dim=1)
        pred_norm = F.normalize(pred, p=2, dim=1)

        ones = torch.ones(pred.size(0), device=pred.device)
        cos_loss = self.cosine(pred_norm, target_norm, ones)
        mse_loss = self.mse(pred_norm, target_norm)

        l2_reg = torch.mean(pred.pow(2))

        return self.cosine_w * cos_loss + self.mse_w * mse_loss + self.l2_lambda * l2_reg


# ──────────────────────────── EMA ────────────────────────────────
class EMA:
    """Exponential Moving Average of model weights for better generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA params. Returns backup for restore."""
        backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        """Restore model params from backup."""
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])


# ──────────────────────────── Multi-scale DINOv2 extraction ──────────────────────
def extract_multiscale_dino(dino_model, pixel_values):
    """
    Extract CLS + spatial pooling from DINOv2.
    Returns (batch, 1536) — matches sort_images_app.py.
    """
    hs = dino_model(pixel_values=pixel_values).last_hidden_state
    cls_tok = hs[:, 0, :]             # (B, 768)
    spatial = hs[:, 1:, :].mean(dim=1)  # (B, 768)
    return torch.cat([cls_tok, spatial], dim=-1)  # (B, 1536)


# ──────────────────────────── Checkpoint helpers ────────────────────────────────
def save_checkpoint(path: str, epoch: int, adapter: nn.Module,
                    optimizer, scheduler, scaler, best_loss: float,
                    ema: EMA, cfg: dict):
    torch.save({
        "epoch": epoch,
        "adapter_state": adapter.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "ema_shadow": ema.shadow if ema else None,
        "best_loss": best_loss,
        "config": cfg,
    }, path)


def load_checkpoint(path: str, adapter: nn.Module, optimizer,
                    scheduler, scaler, ema, device):
    log.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    if ema and ckpt.get("ema_shadow"):
        ema.shadow = ckpt["ema_shadow"]
    return ckpt["epoch"] + 1, ckpt.get("best_loss", float("inf")), ckpt.get("config", {})


# ──────────────────────────────── Main ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2 (multi-scale) adapter -> Text embeddings")
    p.add_argument("--data_dir",    default=DEFAULT_CONFIG["data_dir"],
                   help="Directory with training data (JSON + images)")
    p.add_argument("--base_weights_dir", default=DEFAULT_CONFIG["base_weights_dir"],
                   help="Directory to save adapter weights")
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--grad_accum",  type=int,   default=DEFAULT_CONFIG["grad_accum_steps"],
                   help="Gradient accumulation steps")
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--weight_decay",type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--hidden_dim",  type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--dropout",     type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--text_model",  default=DEFAULT_CONFIG["text_model"],
                   help="Path to SentenceTransformer model for text embeddings")
    p.add_argument("--dino_path",   default=DEFAULT_CONFIG["dino_path"],
                   help="Path to local DINOv2 model")
    p.add_argument("--num_workers", type=int,   default=DEFAULT_CONFIG["num_workers"])
    p.add_argument("--cpu",         action="store_true",
                   help="Force CPU mode (very slow)")
    p.add_argument("--resume",      default=None,
                   help="Path to .pth checkpoint to resume training")
    p.add_argument("--no_ema",      action="store_true", help="Disable EMA")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──
    if args.cpu:
        device = torch.device("cpu")
        log.warning("CPU mode (--cpu). Training will be very slow!")
    elif not torch.cuda.is_available():
        log.error(
            "CUDA not available! Install PyTorch with CUDA support:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "To run on CPU, add the --cpu flag"
        )
        sys.exit(1)
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"CUDA: {gpu_name} | VRAM: {vram:.1f} GB")

    # ── Output directories ──
    base_dir = args.base_weights_dir
    os.makedirs(base_dir, exist_ok=True)
    cache_file = os.path.join(base_dir, DEFAULT_CONFIG["cache_file"])

    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Weights for this run will be saved to: {save_dir}")

    # ── Dataset ──
    dataset = ImageTextDataset(
        data_dir=args.data_dir,
        text_model_name=args.text_model,
        cache_file=cache_file,
        device=device,
    )
    if len(dataset) == 0:
        log.error("No training data found! Check JSON format and image paths.")
        sys.exit(1)

    # Detect output_dim from cached embeddings
    sample_emb = next(iter(dataset.cached_embeddings.values()))
    output_dim = sample_emb.shape[0]
    log.info(f"Text embedding dimension (output_dim): {output_dim}")

    safe_workers = 0 if os.name == "nt" else args.num_workers

    # Train/Val split
    import random as _rng
    val_split = DEFAULT_CONFIG["val_split"]
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    all_indices = list(range(n_total))
    _rng.seed(42)
    _rng.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    log.info(f"Split: train={n_train}, val={n_val} ({val_split:.0%})")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=None,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=None,
    )

    # ── Backbone: DINOv2 (frozen) ──
    log.info(f"Loading DINOv2 from: {args.dino_path}")
    dinov2 = AutoModel.from_pretrained(args.dino_path, local_files_only=True)
    dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False
    total_params = sum(p.numel() for p in dinov2.parameters())
    log.info(f"DINOv2 loaded. Frozen params: {total_params:,}")
    log.info(f"Multi-scale output: CLS(768) + spatial(768) = 1536d")

    # ── Adapter ──
    input_dim = DEFAULT_CONFIG["input_dim"]  # 1536
    adapter = DINOv2Adapter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    log.info(f"Adapter: {input_dim}d -> {args.hidden_dim} -> {args.hidden_dim // 2} -> {output_dim}d")
    log.info(f"Trainable parameters: {trainable:,}")

    # ── Loss / Optimizer / Scheduler ──
    criterion = CombinedLoss(
        cosine_w=DEFAULT_CONFIG["cosine_loss_weight"],
        mse_w=DEFAULT_CONFIG["mse_loss_weight"],
        l2_lambda=DEFAULT_CONFIG["l2_lambda"],
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # OneCycleLR: warmup -> peak -> cosine decay
    # Use ceiling division + buffer to avoid "stepped too many times" error
    steps_per_epoch = -(-len(train_loader) // args.grad_accum)  # ceiling div
    total_steps = steps_per_epoch * args.epochs + 1  # +1 safety buffer
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, total_steps),
        pct_start=0.1,       # 10% warmup
        anneal_strategy='cos',
        div_factor=10,       # start_lr = max_lr / 10
        final_div_factor=100, # end_lr = max_lr / 1000
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(str(device) == "cuda"))

    # ── EMA ──
    ema = None if args.no_ema else EMA(adapter, decay=DEFAULT_CONFIG["ema_decay"])
    if ema:
        log.info(f"EMA enabled (decay={DEFAULT_CONFIG['ema_decay']})")

    # ── Resume ──
    start_epoch = 0
    best_loss = float("inf")

    # ── Persistent best_loss across runs (validated by dataset hash) ──
    best_loss_file = os.path.join(base_dir, "best_val_loss.json")
    dataset_hash = hashlib.md5(
        json.dumps(
            [(s["image_path"], s["description"]) for s in sorted(dataset.samples, key=lambda x: x["image_path"])],
            ensure_ascii=False
        ).encode()
    ).hexdigest()

    if os.path.exists(best_loss_file):
        try:
            with open(best_loss_file, "r") as f:
                saved = json.load(f)
            if saved.get("dataset_hash") == dataset_hash:
                best_loss = saved["best_val_loss"]
                log.info(f"Loaded previous best_val_loss={best_loss:.4f} (dataset unchanged)")
            else:
                log.info("Dataset changed — previous best_val_loss not applicable")
        except Exception:
            pass

    if args.resume:
        if not os.path.exists(args.resume):
            log.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)
        start_epoch, best_loss, _ = load_checkpoint(
            args.resume, adapter, optimizer, scheduler, scaler, ema, device
        )
        log.info(f"Resuming from epoch {start_epoch} | best_loss={best_loss:.4f}")
    else:
        existing = sorted(glob.glob(os.path.join(base_dir, "**", "best_adapter.pth"), recursive=True))
        if existing:
            latest = existing[-1]
            log.info(f"Found existing adapter: {latest}. Load it? (y/n): ")
            answer = input().strip().lower()
            if answer == "y":
                try:
                    start_epoch, best_loss, _ = load_checkpoint(
                        latest, adapter, optimizer, scheduler, scaler, ema, device
                    )
                    log.info(f"Resumed from epoch {start_epoch} | best_loss={best_loss:.4f}")
                except Exception:
                    ckpt = torch.load(latest, map_location=device, weights_only=False)
                    sd = ckpt.get("adapter", ckpt) if isinstance(ckpt, dict) else ckpt

                    # Check dimension compatibility
                    w0 = sd.get("net.0.weight")
                    if w0 is not None and w0.shape[1] != input_dim:
                        log.warning(
                            f"Existing adapter has input_dim={w0.shape[1]}, "
                            f"but current = {input_dim}. Skipping load."
                        )
                    else:
                        adapter.load_state_dict(sd)
                        log.info("Loaded adapter weights only. Training from epoch 0.")
                    start_epoch = 0

    # ── Training loop ──
    patience = DEFAULT_CONFIG["early_stopping_patience"]
    no_improve = 0
    grad_accum = args.grad_accum
    label_noise = DEFAULT_CONFIG["label_noise"]

    log.info(f"{'='*55}")
    log.info(f"Training: epochs {start_epoch + 1}->{args.epochs} | "
             f"batch={args.batch_size} x accum={grad_accum} = effective {args.batch_size * grad_accum} | "
             f"lr={args.lr} | patience={patience}")
    log.info(f"{'='*55}")

    for epoch in range(start_epoch, args.epochs):
        # ── Train phase ──
        adapter.train()
        dataset.set_train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"[{epoch+1:3d}/{args.epochs}] train", dynamic_ncols=True)

        for step, (images, text_embs) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            text_embs = text_embs.to(device, non_blocking=True)

            # Label noise — prevents memorizing exact targets
            if label_noise > 0:
                text_embs = text_embs + torch.randn_like(text_embs) * label_noise

            with torch.amp.autocast("cuda", enabled=(str(device) == "cuda")):
                with torch.no_grad():
                    dino_features = extract_multiscale_dino(dinov2, images)
                pred_embs = adapter(dino_features)
                loss = criterion(pred_embs, text_embs) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if ema:
                    ema.update(adapter)

            total_loss += loss.item() * grad_accum
            n_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg_train_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # ── Val phase (with EMA if enabled) ──
        adapter.eval()
        dataset.set_val()

        ema_backup = None
        if ema:
            ema_backup = ema.apply(adapter)

        val_loss = 0.0
        val_cos_sim = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, text_embs in val_loader:
                images = images.to(device, non_blocking=True)
                text_embs = text_embs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(str(device) == "cuda")):
                    dino_features = extract_multiscale_dino(dinov2, images)
                    pred_embs = adapter(dino_features)
                    loss = criterion(pred_embs, text_embs)

                    # Cosine similarity metric
                    cos_sim = F.cosine_similarity(
                        F.normalize(pred_embs.float(), dim=1),
                        F.normalize(text_embs.float(), dim=1),
                        dim=1
                    ).mean()

                val_loss += loss.item()
                val_cos_sim += cos_sim.item()
                n_val_batches += 1

        if ema and ema_backup:
            ema.restore(adapter, ema_backup)

        avg_val_loss = val_loss / max(n_val_batches, 1)
        avg_cos_sim = val_cos_sim / max(n_val_batches, 1)

        log.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"val_cos_sim={avg_cos_sim:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        # Save checkpoint every epoch
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
        save_checkpoint(ckpt_path, epoch, adapter, optimizer, scheduler,
                        scaler, best_loss, ema, vars(args))

        # Best model (by val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve = 0

            # Save EMA weights as the best adapter
            if ema:
                ema_backup = ema.apply(adapter)
                save_data = {"adapter": adapter.state_dict()}
                ema.restore(adapter, ema_backup)
            else:
                save_data = {"adapter": adapter.state_dict()}

            best_path = os.path.join(save_dir, "best_adapter.pth")
            global_best_path = os.path.join(base_dir, "best_adapter.pth")
            torch.save(save_data, best_path)
            torch.save(save_data, global_best_path)

            # Persist best_val_loss for future runs
            with open(best_loss_file, "w") as f:
                json.dump({"best_val_loss": best_loss, "dataset_hash": dataset_hash}, f)

            log.info(f"  * New best val_loss! cos_sim={avg_cos_sim:.4f} | -> {best_path}")
        else:
            no_improve += 1
            log.info(f"  ! val_loss did not improve ({no_improve}/{patience})")
            if no_improve >= patience:
                log.info(f"  X Early stopping! val_loss stagnated for {patience} epochs.")
                break

    # Final save
    if ema:
        ema.apply(adapter)
    final_path = os.path.join(save_dir, "adapter_final.pth")
    torch.save({"adapter": adapter.state_dict()}, final_path)
    log.info(f"Training complete! Final model: {final_path}")
    log.info(f"Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
