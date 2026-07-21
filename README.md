# CBAM-Enhanced ResNet18 and U-Net for Steel Surface Defect Detection: Joint Classification and Segmentation Using the Severstal Dataset

A deep learning project for automated detection and segmentation of surface defects in steel strips using the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) dataset. Two tasks are implemented: **multi-label classification** and **pixel-level segmentation**, each with a baseline model and a CBAM-enhanced variant.

---

## Tasks

| Task | Baseline | Enhanced |
|---|---|---|
| Multi-label Classification | ResNet18 | CBAM-ResNet18 |
| Pixel-level Segmentation | U-Net (ResNet18 encoder) | CBAM U-Net |

---

## Key Results

### Classification (3-seed mean ± std)

| Model | Macro F1 | ROC AUC |
|---|---|---|
| Baseline ResNet18 | 0.8445 ± 0.0159 | 0.9881 ± 0.0028 |
| CBAM-ResNet18 | 0.8526 ± 0.0232 | 0.9889 ± 0.0026 |

### Segmentation (epoch 40)

| Model | Dice | IoU |
|---|---|---|
| Baseline U-Net | 0.8372 | 0.8114 |
| CBAM U-Net | 0.8368 | 0.8116 |

---

## Project Structure

```
SteelDefectDetection/
├── src/                                     # Source code modules
│   ├── models/                              # PyTorch model definitions
│   │   ├── cbam.py                          # Custom CBAM module (pure PyTorch)
│   │   ├── cbam_resnet18.py                 # CBAM-ResNet18 classifier
│   │   ├── resnet18_classifier.py           # Baseline ResNet18 classifier
│   │   ├── unet_cbam.py                     # CBAM U-Net segmentation model
│   │   ├── unet_resnet18.py                 # Baseline U-Net segmentation model
│   │   └── unet_blocks.py                   # U-Net building blocks
│   ├── datasets/                            # Datasets, splits & transforms
│   ├── losses/                              # Custom loss functions (Focal, Dice)
│   ├── metrics/                             # Metric calculation utilities
│   ├── train/                               # Trainer & callback classes
│   └── utils/                               # Helper utilities (RLE, seed)
├── data/                                    # Dataset splits & master labels
├── logs/                                    # Raw training CSV logs (per seed)
├── checkpoints/                             # Trained model weights (.pth)
├── results/                                 # Derived tables, metrics & audit reports
│   ├── attention/                           # Attention map visualisations
│   ├── gradcam/                             # Grad-CAM classification maps
│   ├── explainability/                      # Segmentation TP/FP/FN overlays
│   └── error_analysis/                      # Worst FP/FN failure analysis
├── figures/                                 # Publication-ready report figures
├── dataset_audit.py                         # Dataset audit & master label builder
├── train_classifier.py                      # Classification training CLI
├── train_segmentation.py                    # Segmentation training CLI
├── run_all_seeds.py                         # Multi-seed training automation script
├── regenerate_all_results.py                # Master result regeneration script
├── verify_results.py                        # Automated project integrity verification
├── generate_report_figures.py               # Generates all report figures (fig4, fig6-fig9)
├── generate_training_plots.py               # Auxiliary PR/ROC/Confusion matrix plots
├── collect_runtime_analysis.py              # Inference speed & parameter benchmarking
├── generate_gradcam_visualizations.py       # Grad-CAM generator
├── generate_cbam_attention_visualizations.py  # CBAM attention map generator
├── generate_segmentation_explainability.py   # Segmentation error overlay generator
├── generate_classification_error_analysis.py# Error analysis plot generator
├── environment.yml                          # Conda environment definition
├── README.md                                # Project overview
└── comands/how_to_run.md                    # Detailed CLI command reference
```

---

## Setup

```powershell
conda env create -f environment.yml
conda activate env-project
```

---

## Quick Start

```powershell
# 1. Prepare dataset
python dataset_audit.py
python src/datasets/split_dataset.py

# 2. Train models (single seed 42)
python train_classifier.py --model baseline --seed 42
python train_classifier.py --model cbam --seed 42
python train_segmentation.py --model baseline --seed 42
python train_segmentation.py --model cbam --seed 42

# 3. Regenerate all result tables & audit metrics
python regenerate_all_results.py

# 4. Generate report figures
python generate_report_figures.py

# 5. Verify integrity (158 automated checks)
python verify_results.py
```

> See [`comands/how_to_run.md`](comands/how_to_run.md) for the complete step-by-step pipeline reference.

---

## Report Figures

| Figure File | Short Caption |
|---|---|
| `figures/fig4_class_distribution.png` | Severstal training set class distribution showing Class 2 imbalance. |
| `figures/fig6_macro_f1_comparison.png` | Macro F1 score across seeds: Val-Loss vs Peak-F1 checkpoint selection. |
| `figures/fig7_dice_vs_epoch.png` | Segmentation Dice score vs epoch for Baseline vs CBAM U-Net. |
| `figures/fig8_classification_train_val_loss.png` | **Fig. 8** — Train vs. validation loss: Classification (Seed 42). |
| `figures/fig9_segmentation_train_val_loss.png` | **Fig. 9** — Train vs. validation loss: Segmentation (40 epochs). |

---

## Hardware

Trained on **NVIDIA RTX 3050 (4 GB VRAM)**.
- **Classification:** ~6–7.5 min per run (11–14 epochs).
- **Segmentation:** ~45–50 min per run (40 epochs).
