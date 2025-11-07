#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Training loop and CLI entry point for fine-tuning a ResNet-based classifier
on brain MRI images.

Usage:
python -m brain_tumor_detection.train --data-dir data/brain_mri --output-dir runs/exp1

Notes:
- Designed to be simple and readable while following best practices.
- Early stopping is implemented based on validation accuracy.
==========================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import DatasetConfig, ModelConfig, TrainingConfig, get_default_config
from .data import Dataloaders, create_dataloaders
from .model import create_model
from .utils import (
    accuracy_from_logits,
    config_to_dict,
    ensure_dir,
    get_device,
    save_checkpoint,
    set_seed,
)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch and return (loss, accuracy)."""

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits, targets)
        running_loss += loss.item()
        running_acc += acc
        num_batches += 1

    return running_loss / max(num_batches, 1), running_acc / max(num_batches, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a validation/test loader."""

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            acc = accuracy_from_logits(logits, targets)
            running_loss += loss.item()
            running_acc += acc
            num_batches += 1

    return running_loss / max(num_batches, 1), running_acc / max(num_batches, 1)


def train_main(args: argparse.Namespace) -> None:
    """Main training routine used by the CLI."""

    base_dir = Path(args.base_dir).resolve()
    dataset_cfg, model_cfg, train_cfg = get_default_config(base_dir)

    # Override defaults from CLI
    if args.data_dir is not None:
        dataset_cfg.data_dir = Path(args.data_dir).resolve()
    if args.output_dir is not None:
        train_cfg.output_dir = Path(args.output_dir).resolve()
    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.img_size is not None:
        dataset_cfg.img_size = args.img_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.device is not None:
        train_cfg.device = args.device

    set_seed(dataset_cfg.seed)

    device = get_device(train_cfg.device)
    print(f"Using device: {device}")

    loaders, idx_to_class = create_dataloaders(
        dataset_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    model = create_model(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    output_dir = ensure_dir(train_cfg.output_dir)
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = output_dir / "best_model.pt"

    print("Class index mapping:")
    for idx, cls_name in idx_to_class.items():
        print(f"  {idx}: {cls_name}")

    for epoch in range(1, train_cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{train_cfg.num_epochs}")

        train_loss, train_acc = _train_one_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = _evaluate(model, loaders.val, criterion, device)

        scheduler.step(val_acc)

        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "dataset_cfg": config_to_dict(dataset_cfg),
                "model_cfg": config_to_dict(model_cfg),
                "training_cfg": config_to_dict(train_cfg),
                "idx_to_class": idx_to_class,
            }
            save_checkpoint(state, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path} (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(
                f"No improvement in val_acc. Patience: "
                f"{patience_counter}/{train_cfg.early_stopping_patience}"
            )

        if patience_counter >= train_cfg.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Train a brain tumor detection model using ResNet-18.",
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
        help="Path to ImageFolder-compatible dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save logs and model checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for train/val/test (overrides config).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input image size (overrides config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (L2 regularization, overrides config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Computation device (overrides config).",
    )

    return parser


def main() -> None:
    """CLI entry point (``python -m brain_tumor_detection.train``)."""

    parser = build_argparser()
    args = parser.parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
