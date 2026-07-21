# Classification Results: 3-Run Mean +/- Std

Best epoch for each seed is selected by lowest validation loss, matching the checkpoint saver.

## Mean +/- Std Comparison

| Model | Runs | Precision | Recall | Macro F1 | Micro F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ResNet18 | 3 | 0.8779 +/- 0.0383 | 0.8236 +/- 0.0223 | 0.8445 +/- 0.0159 | 0.9021 +/- 0.0087 | 0.9881 +/- 0.0028 |
| CBAM-ResNet18 | 3 | 0.8893 +/- 0.0352 | 0.8245 +/- 0.0409 | 0.8526 +/- 0.0232 | 0.9047 +/- 0.0102 | 0.9889 +/- 0.0026 |

## Macro F1 Difference Check

| Comparison | Value |
|---|---:|
| CBAM mean - Baseline mean | 0.0081 |
| Combined std (Baseline std + CBAM std) | 0.0391 |
| Absolute difference larger than combined std? | No |

## Per-Seed Raw Values

| Model | Seed | Best Epoch | Precision | Recall | Macro F1 | Micro F1 | ROC AUC | Val Loss | Epochs Completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline ResNet18 | 42 | 7 | 0.8953 | 0.8000 | 0.8341 | 0.8989 | 0.9860 | 0.0052 | 12 |
| Baseline ResNet18 | 0 | 8 | 0.8340 | 0.8442 | 0.8367 | 0.8955 | 0.9870 | 0.0053 | 13 |
| Baseline ResNet18 | 7 | 8 | 0.9044 | 0.8267 | 0.8628 | 0.9120 | 0.9913 | 0.0047 | 13 |
| CBAM-ResNet18 | 42 | 9 | 0.9156 | 0.8447 | 0.8777 | 0.9128 | 0.9915 | 0.0048 | 14 |
| CBAM-ResNet18 | 0 | 8 | 0.9029 | 0.7774 | 0.8318 | 0.8933 | 0.9864 | 0.0050 | 13 |
| CBAM-ResNet18 | 7 | 12 | 0.8494 | 0.8513 | 0.8484 | 0.9080 | 0.9889 | 0.0052 | 17 |
