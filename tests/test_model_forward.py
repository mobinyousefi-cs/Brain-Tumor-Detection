#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: tests/test_model_forward.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Basic forward-pass test to ensure the model produces outputs of the expected
shape for a dummy batch.

Usage:
pytest tests/test_model_forward.py

Notes:
- This test does not require the dataset; it uses random tensors.
==========================================================================
"""
from __future__ import annotations

import torch

from brain_tumor_detection.config import ModelConfig
from brain_tumor_detection.model import create_model


def test_model_forward_shape() -> None:
    cfg = ModelConfig(num_classes=2, pretrained=False)
    model = create_model(cfg)

    dummy_inputs = torch.randn(4, 3, 224, 224)
    outputs = model(dummy_inputs)

    assert outputs.shape == (4, 2)
