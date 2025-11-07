#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: utils.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Utility helpers for seeding, device management, metrics, and checkpoint I/O.

Usage:
from brain_tumor_detection.utils import set_seed, get_device

Notes:
- Keep these functions generic so they can be reused across projects.
==========================================================================
"""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Return a torch.device based on the given string.

    Parameters
    ----------
    device:
        - "cpu" to force CPU
        - "cuda" to force GPU (if available)
        - "auto" to prefer GPU when available (default)
    """

    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy given model logits and integer class targets."""

    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Save training state to disk.

    The ``state`` dict typically includes:

    - epoch
    - model_state_dict
    - optimizer_state_dict
    - best_val_acc
    - config objects (converted to plain dicts)
    """

    checkpoint_path = checkpoint_path.resolve()
    ensure_dir(checkpoint_path.parent)
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a checkpoint from disk."""

    return torch.load(checkpoint_path, map_location=map_location)


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert a dataclass config to a plain dictionary if needed."""

    try:
        return asdict(config)
    except TypeError:
        # Not a dataclass; assume it is already a mapping-like object.
        return dict(config)
