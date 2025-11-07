#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Brain Tumor Detection with Data Science
File: tests/test_imports.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=

Description:
Simple smoke tests to ensure that core modules can be imported without errors.

Usage:
pytest tests/test_imports.py

Notes:
- Extend with additional import checks as the project grows.
==========================================================================
"""
from __future__ import annotations

import importlib


def test_import_package() -> None:
    importlib.import_module("brain_tumor_detection")


def test_import_submodules() -> None:
    for name in [
        "brain_tumor_detection.config",
        "brain_tumor_detection.data",
        "brain_tumor_detection.model",
        "brain_tumor_detection.train",
        "brain_tumor_detection.evaluate",
        "brain_tumor_detection.utils",
    ]:
        importlib.import_module(name)
