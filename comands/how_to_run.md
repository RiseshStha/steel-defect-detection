# Project Running Reference (Definitive Guide)

This document contains the tested, exact CLI commands for running all parts of the steel defect detection pipeline on your local system.

---

## 1. Preprocess / Prepare the Dataset
Prepares positive annotations, audits the training images on disk, and splits the data into stratified 70/15/15 train/validation/test sets.

### Commands
```powershell
# Step A: Audit dataset and build master labels CSV
python dataset_audit.py

# Step B: Perform stratified train/validation/test split
python src/datasets/split_dataset.py
```

### Outputs
- `data/processed/master_labels.csv` (complete image-level multi-label matrix)
- `data/processed/train_split.csv` (70% training set split)
- `data/processed/val_split.csv` (15% validation set split)
- `data/processed/test_split.csv` (15% test set split)

### Runtime
- **dataset_audit.py**: ~5 seconds
- **split_dataset.py**: ~2 seconds

---

## 2. Train Baseline ResNet18 Classifier
Trains the baseline ResNet18 classifier on the multi-label classification task.

### Command
```powershell
python train_classifier.py --model baseline --seed 42
```

### Outputs
- **Checkpoint**: `checkpoints/baseline_resnet18_seed42_best.pth` (retains a copy at `checkpoints/baseline_resnet18_best.pth`)
- **Log**: `logs/baseline_seed42.csv` (retains a copy at `logs/baseline_training.csv`)
- **Metrics Result**: `results/classification_baseline_seed42.csv`

### Runtime
- **Total Time**: ~7.5 minutes (450 seconds) on RTX 3050 (4GB VRAM) for 11 epochs (~41.0 seconds per epoch).

---

## 3. Train CBAM-ResNet18 Classifier
Trains the ResNet18 classifier equipped with CBAM attention blocks.

### Command
```powershell
python train_classifier.py --model cbam --seed 42
```

### Outputs
- **Checkpoint**: `checkpoints/cbam_resnet18_seed42_best.pth` (retains a copy at `checkpoints/cbam_resnet18_best.pth`)
- **Log**: `logs/cbam_seed42.csv` (retains a copy at `logs/cbam_training.csv`)
- **Metrics Result**: `results/classification_cbam_seed42.csv`

### Runtime
- **Total Time**: ~6.2 minutes (373 seconds) on RTX 3050 (4GB VRAM) for 9 epochs (~41.5 seconds per epoch).

---

## 4. Train Baseline U-Net Segmentation Model
Trains the baseline U-Net (with a ResNet18 encoder) on the pixel-level defect segmentation task.

### Command
```powershell
python train_segmentation.py --model baseline --seed 42
```

### Outputs
- **Checkpoint**: `checkpoints/baseline_unet_best.pth`
- **Log**: `logs/baseline_history.csv`

### Runtime
- **Total Time**: ~45-50 minutes on RTX 3050 (4GB VRAM) for 40 epochs (~70-75 seconds per epoch).

---

## 5. Train CBAM U-Net Segmentation Model
Trains the U-Net segmentation model using a CBAM-integrated ResNet18 encoder.

### Command
```powershell
python train_segmentation.py --model cbam --seed 42
```

### Outputs
- **Checkpoint**: `checkpoints/cbam_unet_best.pth`
- **Log**: `logs/cbam_history.csv`

### Runtime
- **Total Time**: ~45-50 minutes on RTX 3050 (4GB VRAM) for 40 epochs (~70-75 seconds per epoch).

---

## 6. Run Master Results Regeneration
Single source-of-truth script that re-computes all derived metrics, comparison tables, and 3-run mean ± std summaries directly from raw logs.

### Command
```powershell
python regenerate_all_results.py
```

### Outputs
- `results/classification_results.csv` and `results/classification_results.md`
- `results/classification_results_3run.csv` and `results/classification_results_3run.md`
- `results/classification_results_3run_raw.csv`
- `results/classification_comparison_table.csv`
- `results/segmentation_comparison_table.csv`
- `results/comparison_tables.md`
- `results/checkpoint_selection_audit.md`

### Runtime
- **Total Time**: ~3 seconds.

---

## 7. Run Hardware and Speed Benchmarking
Measures inference execution speed, parameter counts, checkpoint file sizes, and image throughput on the GPU.

### Command
```powershell
python collect_runtime_analysis.py
```

### Outputs
- `results/runtime/runtime_analysis.csv`
- `results/runtime/runtime_analysis.md`

### Runtime
- **Total Time**: ~1.5 minutes (runs live GPU inference benchmarks).

---

## 8. Generate Report Figures
Generates all 5 publication-ready figures for the final report (`fig4`, `fig6`, `fig7`, `fig8`, `fig9`).

### Command
```powershell
python generate_report_figures.py
```

### Outputs
- `figures/fig4_class_distribution.png` (Class distribution bar chart)
- `figures/fig6_macro_f1_comparison.png` (Macro F1 seed comparison)
- `figures/fig7_dice_vs_epoch.png` (Segmentation Dice score vs epoch)
- `figures/fig8_classification_train_val_loss.png` (**Fig. 8** — Train vs. val loss: Classification)
- `figures/fig9_segmentation_train_val_loss.png` (**Fig. 9** — Train vs. val loss: Segmentation)

### Runtime
- **Total Time**: ~5 seconds.

---

## 9. Generate Grad-CAM Visualizations
Generates visual Grad-CAM class activation maps for 6 defective validation images to explain classifier predictions.

### Command
```powershell
python generate_gradcam_visualizations.py --baseline-checkpoint checkpoints/baseline_resnet18_best.pth --cbam-checkpoint checkpoints/cbam_resnet18_best.pth --output-dir results/gradcam/classification --num-examples 6
```

### Outputs
- `results/gradcam/classification/gradcam_manifest.csv` (visualized examples index)
- `results/gradcam/classification/example_01_gradcam.png` through `example_06_gradcam.png`

### Runtime
- **Total Time**: ~12 seconds.

---

## 10. Generate CBAM Attention Map Visualizations
Extracts and generates visualizations of CBAM's channel and spatial attention maps for validation examples.

### Command
```powershell
python generate_cbam_attention_visualizations.py --cbam-checkpoint checkpoints/cbam_resnet18_best.pth --output-dir results/attention/cbam --num-examples 6
```

### Outputs
- `results/attention/cbam/cbam_attention_manifest.csv` (attention statistics and index)
- `results/attention/cbam/cbam_attention_explanation.md` (explanation of features)
- `results/attention/cbam/example_01_cbam_attention.png` through `example_06_cbam_attention.png`

### Runtime
- **Total Time**: ~12 seconds.

---

## 11. Generate Segmentation Explainability Maps
Creates pixel-level visual error overlays showing True Positives (green), False Negatives (red), and False Positives (orange) side-by-side to compare U-Net models.

### Command
```powershell
python generate_segmentation_explainability.py --baseline-checkpoint checkpoints/baseline_unet_best.pth --cbam-checkpoint checkpoints/cbam_unet_best.pth --output-dir results/explainability/segmentation --num-examples 6
```

### Outputs
- `results/explainability/segmentation/segmentation_explainability_manifest.csv`
- `results/explainability/segmentation/segmentation_explainability_explanation.md`
- `results/explainability/segmentation/example_01_comparison.png`, `example_01_baseline.png`, `example_01_cbam.png` (up to example 06)

### Runtime
- **Total Time**: ~15 seconds.

---

## 12. Run Error Analysis
Extracts the worst failure cases (highest confidence False Positives and lowest confidence False Negatives) for each class across baseline and CBAM classifiers.

### Command
```powershell
python generate_classification_error_analysis.py --val-csv data/processed/val_split.csv --image-dir data/train_images --output-dir results/error_analysis/classification --baseline-checkpoint checkpoints/baseline_resnet18_best.pth --cbam-checkpoint checkpoints/cbam_resnet18_best.pth --threshold 0.5 --examples-per-class 2
```

### Outputs
- `results/error_analysis/classification/classification_error_manifest.csv` (list of 32 error examples)
- `results/error_analysis/classification/baseline_class1_false_negative_1.png` to `cbam_class4_false_positive_2.png` (32 comparison plots)

### Runtime
- **Total Time**: ~15 seconds.

---

## 13. Run Automated Project Integrity Verification
Executes numerical verification across the entire project, re-deriving all log results, checking error analysis manifests, and performing 158 automated pass/fail checks.

### Command
```powershell
python verify_results.py
```

### Outputs
- `results/verification_report.md` (detailed 158-check pass/fail audit report)

### Runtime
- **Total Time**: ~5 seconds.
