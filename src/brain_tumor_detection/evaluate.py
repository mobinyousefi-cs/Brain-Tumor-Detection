#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: evaluate.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Evaluation script and utilities for a trained brain tumor detection model.

Usage:
python -m brain_tumor_detection.evaluate \
  --data-dir data/brain_mri \
  --checkpoint runs/exp1/best_model.pt

Notes:
- Reports accuracy, classification report, and confusion matrix.
- Optionally saves a confusion matrix plot as PNG.
==========================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import datasets, transforms

from .config import DatasetConfig, ModelConfig
from .model import create_model
from .utils import get_device, load_checkpoint, set_seed


def _build_eval_loader(dataset_cfg: DatasetConfig, batch_size: int, num_workers: int) -> Tuple[DataLoader, Dict[int, str]]:
    """Build test loader (no splitting; assumes full folder is test set)."""

    transform = transforms.Compose(
        [
            transforms.Resize((dataset_cfg.img_size, dataset_cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(root=str(dataset_cfg.data_dir), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return loader, idx_to_class


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred)."""

    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Test", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    return np.array(all_targets), np.array(all_preds)


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path | None = None,
) -> None:
    """Plot and optionally save a confusion matrix."""

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path is not None:
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")


def evaluate_main(args: argparse.Namespace) -> None:
    """Main evaluation routine used by the CLI."""

    base_dir = Path(args.base_dir).resolve()

    dataset_cfg = DatasetConfig(
        data_dir=Path(args.data_dir).resolve() if args.data_dir else base_dir / "data" / "brain_mri",
        img_size=args.img_size or 224,
        seed=42,
    )

    device = get_device(args.device or "auto")
    print(f"Using device: {device}")

    set_seed(dataset_cfg.seed)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    # Restore model configuration if available, otherwise fall back to defaults
    model_cfg_dict = checkpoint.get("model_cfg", {})
    model_cfg = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()

    model = create_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader, idx_to_class = _build_eval_loader(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    y_true, y_pred = _predict(model, loader, device)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (raw counts):")
    print(cm)

    if args.cm_path is not None:
        _plot_confusion_matrix(cm, class_names, save_path=Path(args.cm_path))


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained brain tumor detection model.",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for project paths (default: current directory)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to ImageFolder-compatible test dataset directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input image size (overrides config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Computation device.",
    )
    parser.add_argument(
        "--cm-path",
        type=str,
        default=None,
        help="Optional path to save confusion matrix plot as PNG.",
    )

    return parser


def main() -> None:
    """CLI entry point (``python -m brain_tumor_detection.evaluate``)."""

    parser = build_argparser()
    args = parser.parse_args()
    evaluate_main(args)


if __name__ == "__main__":
    main()
