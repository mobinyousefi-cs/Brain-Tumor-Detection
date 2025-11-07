#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Central configuration objects and helper function for dataset, model, and
training hyperparameters.

Usage:
from pathlib import Path
from brain_tumor_detection.config import get_default_config

dataset_cfg, model_cfg, train_cfg = get_default_config(base_dir=Path("."))

Notes:
- The defaults are chosen to be reasonable for most GPU setups.
- Override fields as needed or expose them via CLI arguments.
==========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DatasetConfig:
    """Configuration for dataset and dataloaders."""

    data_dir: Path
    img_size: int = 224
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for the classification model."""

    arch: Literal["resnet18"] = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Configuration for training loop and optimization."""

    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "auto"  # "cpu", "cuda" or "auto"
    early_stopping_patience: int = 5
    output_dir: Path = Path("runs")


def get_default_config(base_dir: Path | str = ".") -> tuple[DatasetConfig, ModelConfig, TrainingConfig]:
    """Return default dataset, model, and training configs.

    Parameters
    ----------
    base_dir:
        Base directory for resolving relative paths. Typically the project
        root or current working directory.
    """

    base_path = Path(base_dir).resolve()

    dataset_cfg = DatasetConfig(data_dir=base_path / "data" / "brain_mri")
    model_cfg = ModelConfig()
    training_cfg = TrainingConfig(output_dir=base_path / "runs")

    return dataset_cfg, model_cfg, training_cfg
