# Checkpoint Selection Audit: Baseline ResNet18

Compares two checkpoint-selection criteria across 3 seeds.

## Per-Seed Summary

| Seed | Val-Loss Epoch | Val-Loss F1 | Peak-F1 Epoch | Peak F1 | Gain |
|---:|---:|---:|---:|---:|---:|
| 42 | 7 | 0.8341 | 11 | 0.8598 | +0.0257 |
| 0 | 8 | 0.8367 | 12 | 0.8629 | +0.0263 |
| 7 | 8 | 0.8628 | 13 | 0.8671 | +0.0043 |

## Aggregate Comparison

| Criterion | Mean Macro F1 | Std |
|---|---:|---:|
| Lowest Val Loss | 0.8445 | 0.0159 |
| Peak Macro F1   | 0.8633 | 0.0037 |

CBAM peak-F1 exceeds Baseline by +0.0104 macro F1 (0.8737 vs 0.8633).
Baseline peak-F1 std (0.0037) is about 1.1x smaller than CBAM (0.0041).
Baseline peak-F1 spans 0.0073 vs CBAM 0.0082.

