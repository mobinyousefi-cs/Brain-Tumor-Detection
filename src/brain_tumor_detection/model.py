#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Model creation utilities using transfer learning with torchvision's ResNet-18.

Usage:
from brain_tumor_detection.config import ModelConfig
from brain_tumor_detection.model import create_model

model = create_model(ModelConfig())

Notes:
- Currently supports only ResNet-18, but can be extended to other backbones.
==========================================================================
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

from .config import ModelConfig


class ResNet18Classifier(nn.Module):
    """ResNet-18 based classifier for binary brain tumor detection."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        # Load torchvision's ResNet-18
        # For newer torchvision versions, use 'weights' instead of 'pretrained'.
        if pretrained:
            try:
                # New API (torchvision >= 0.13)
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                backbone = models.resnet18(weights=weights)
            except AttributeError:  # fallback for older versions
                backbone = models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: Any) -> Any:  # type: ignore[override]
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_model(cfg: ModelConfig) -> nn.Module:
    """Create a classification model according to the given configuration."""

    if cfg.arch == "resnet18":
        return ResNet18Classifier(
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
        )

    raise ValueError(f"Unsupported architecture: {cfg.arch}")
