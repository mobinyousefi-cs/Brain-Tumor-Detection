# Brain Tumor Detection with Data Science

A clean, reproducible deep learning pipeline for classifying brain MRI scans into **tumor** vs **no tumor** using **PyTorch** and **transfer learning (ResNet-18)**.

> Author: **Mobin Yousefi**  \
> GitHub: [mobinyousefi-cs](https://github.com/mobinyousefi-cs)

---

## 1. Project Overview

This repository implements an end‑to‑end workflow for detecting brain tumors from MRI images:

- **Data loading & preprocessing** using `torchvision.datasets.ImageFolder`
- **Config‑driven training** (no hard‑coded paths or magic numbers)
- **Transfer learning** with a pre‑trained **ResNet‑18**
- **Training & evaluation loops** with accuracy tracking
- **Confusion matrix & classification report** (via `scikit-learn`)
- **Reproducible experiments** (fixed seeds, clear project structure)

The project is designed to be:

- **Beginner‑friendly**: simple CLI and clear defaults
- **Research‑ready**: easily extendable to other architectures & datasets
- **Production‑friendly**: testable, modular, and packaged as a Python project

Dataset used (example):  
➡️ [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

You can also plug in any **binary classification MRI dataset** following the same folder structure.

---

## 2. Project Structure

```text
brain-tumor-detection/
├─ .github/
│  └─ workflows/
│     └─ ci.yml              # CI: Ruff + Black + pytest
├─ src/
│  └─ brain_tumor_detection/
│     ├─ __init__.py         # Package exports and version
│     ├─ config.py           # Central configuration (paths, hyperparameters)
│     ├─ data.py             # Dataset, transforms, DataLoader utilities
│     ├─ model.py            # Model creation (ResNet‑18 transfer learning)
│     ├─ utils.py            # Helper functions (seeding, metrics, I/O)
│     ├─ train.py            # Training loop & CLI entry point
│     └─ evaluate.py         # Evaluation utilities & CLI
├─ tests/
│  ├─ __init__.py
│  ├─ test_imports.py        # Smoke tests for imports
│  └─ test_model_forward.py  # Simple forward pass test
├─ .editorconfig
├─ .gitignore
├─ LICENSE                   # MIT License
├─ pyproject.toml            # Project metadata and dependencies
└─ README.md                 # You are here
```

You can install the package and run training/evaluation via the CLI modules:

- `python -m brain_tumor_detection.train`
- `python -m brain_tumor_detection.evaluate`

---

## 3. Installation

### 3.1. Clone the repo

```bash
git clone https://github.com/mobinyousefi-cs/brain-tumor-detection.git
cd brain-tumor-detection
```

### 3.2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# On Linux / macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

### 3.3. Install the project (editable mode)

```bash
pip install --upgrade pip
pip install -e .[dev]
```

This installs:

- **Runtime deps**: `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `scikit-learn`, `matplotlib`, `tqdm`, etc.
- **Dev tools**: `pytest`, `black`, `ruff`, `mypy` (via the `dev` extra).

---

## 4. Dataset Preparation

1. Download the dataset from Kaggle:
   - [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

2. Unzip it into the `data/` directory (you can choose a different path, but then update your CLI arguments):

```text
brain-tumor-detection/
└─ data/
   └─ brain_mri/
      ├─ yes/
      │  ├─ Y1.jpg
      │  ├─ ...
      └─ no/
         ├─ N1.jpg
         ├─ ...
```

3. The code treats `data_dir` as an **ImageFolder‑compatible root directory**, where sub‑directories correspond to class names.

> ✅ For a different dataset, just follow the same `data_dir/class_name/*.jpg` pattern.

---

## 5. Quickstart: Train a Model

After installing the package and preparing the dataset, you can start training with a single command.

### 5.1. Basic training run

```bash
python -m brain_tumor_detection.train \
  --data-dir data/brain_mri \
  --output-dir runs/exp1
```

Key options (see `python -m brain_tumor_detection.train --help`):

- `--data-dir`: Root folder containing class sub‑folders (default: `data/brain_mri`)
- `--output-dir`: Where to save logs and model checkpoints (default: `runs/default`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Mini‑batch size (default: 32)
- `--img-size`: Input image size (default: 224)
- `--lr`: Learning rate (default: 3e-4)
- `--weight-decay`: L2 regularization (default: 1e-4)
- `--num-workers`: Dataloader workers (default: 4)
- `--device`: `cpu` or `cuda` (auto‑detected by default)

The script will:

1. Split the dataset into **train/val/test** sets.
2. Apply data augmentations on the training set.
3. Fine‑tune a pre‑trained **ResNet‑18**.
4. Save the best checkpoint (based on validation accuracy) in `output-dir`.

---

## 6. Evaluation

Once you have a trained model checkpoint, you can evaluate it on the test set.

```bash
python -m brain_tumor_detection.evaluate \
  --data-dir data/brain_mri \
  --checkpoint runs/exp1/best_model.pt
```

The evaluation script reports:

- **Test accuracy**
- **Classification report** (precision, recall, F1-score)
- **Confusion matrix**

Optionally, it can export plots (PNG) into the specified output directory.

---

## 7. Running Tests & Linters

Quality checks are configured via **pytest**, **Black**, and **Ruff**.

```bash
# Run unit tests
pytest

# Check formatting
black --check src tests

# Lint code
ruff check src tests
```

CI is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs on each push and pull request.

---

## 8. Extending the Project

Some ideas to take this project further:

- Swap **ResNet‑18** for **EfficientNet**, **DenseNet**, or **Vision Transformers**.
- Add **mixed precision training** (`torch.cuda.amp`) for faster training on GPU.
- Implement **k‑fold cross‑validation** for more robust evaluation.
- Add **Grad‑CAM** visualizations to inspect model attention.
- Extend to **multi‑class classification** (e.g., multiple tumor types).

Because the code is modular and config‑driven, most of these extensions require minimal changes.

---

## 9. Reproducibility

To improve experiment reproducibility, the code:

- Sets **random seeds** for `random`, `numpy`, and `torch`.
- Logs the configuration used for each run.
- Uses explicit **train/val/test** splits.

Nevertheless, full bit‑wise reproducibility is not guaranteed across different hardware and PyTorch versions.

---

## 10. License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

If you use this repository as a starting point for research or coursework, a short citation or GitHub link back to **mobinyousefi-cs** is always appreciated 🙌.

