#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Package initialization, version definition, and convenience exports.

Usage:
from brain_tumor_detection import create_default_model, get_default_config

Notes:
- Keep this file lightweight to avoid slow imports.
==========================================================================
"""

from .config import DatasetConfig, ModelConfig, TrainingConfig, get_default_config
from .model import create_model as create_default_model

__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "get_default_config",
    "create_default_model",
]

__version__ = "0.1.0"
