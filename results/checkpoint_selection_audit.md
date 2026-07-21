# Checkpoint Selection Audit: CBAM-ResNet18

Compares two checkpoint-selection criteria across 3 seeds.

## Per-Seed Summary

| Seed | Val-Loss Epoch | Val-Loss F1 | Peak-F1 Epoch | Peak F1 | Gain |
|---:|---:|---:|---:|---:|---:|
| 42 | 9 | 0.8777 | 9 | 0.8777 | +0.0000 |
| 0 | 8 | 0.8318 | 13 | 0.8695 | +0.0377 |
| 7 | 12 | 0.8484 | 16 | 0.8740 | +0.0257 |

## Aggregate Comparison

| Criterion | Mean Macro F1 | Std |
|---|---:|---:|
| Lowest Val Loss | 0.8526 | 0.0232 |
| Peak Macro F1   | 0.8737 | 0.0041 |

Using peak-F1 checkpoints, mean improves by +0.0211 and std falls by ~82%.
