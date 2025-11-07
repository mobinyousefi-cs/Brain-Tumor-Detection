#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Dataset, transforms, and DataLoader utilities built on top of
``torchvision.datasets.ImageFolder``.

Usage:
from pathlib import Path
from torch.utils.data import DataLoader
from brain_tumor_detection.config import DatasetConfig
from brain_tumor_detection.data import create_dataloaders

loaders, class_names = create_dataloaders(dataset_cfg)

Notes:
- Designed for ImageFolder-compatible layouts: data_dir/class_name/*.jpg
- The Kaggle brain MRI dataset can be used directly after extraction.
==========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import DatasetConfig
from .utils import set_seed


@dataclass
class Dataloaders:
    """Container for train/val/test DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return training and evaluation transforms."""

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(
    dataset_cfg: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[Dataloaders, Dict[int, str]]:
    """Create train/val/test dataloaders and class index mapping.

    Parameters
    ----------
    dataset_cfg:
        DatasetConfig instance specifying data directory and splits.
    batch_size:
        Batch size used for all splits.
    num_workers:
        Number of workers for DataLoader.
    """

    set_seed(dataset_cfg.seed)

    train_transform, eval_transform = _build_transforms(dataset_cfg.img_size)

    # Use ImageFolder to automatically infer class labels from sub-directories
    full_dataset = datasets.ImageFolder(
        root=str(dataset_cfg.data_dir),
        transform=train_transform,
    )

    num_samples = len(full_dataset)
    train_len = int(num_samples * dataset_cfg.train_split)
    val_len = int(num_samples * dataset_cfg.val_split)
    test_len = num_samples - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(dataset_cfg.seed),
    )

    # Validation and test use evaluation transforms
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Map class index to class name for later reporting
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

    loaders = Dataloaders(train=train_loader, val=val_loader, test=test_loader)
    return loaders, idx_to_class
