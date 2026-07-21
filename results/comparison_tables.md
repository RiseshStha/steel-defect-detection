# Model Comparison Tables

Classification values are taken from the best validation-loss checkpoint summary.
Segmentation values are taken from the final recorded validation epoch, with best Dice retained as a diagnostic.

## Classification

| Metric | Baseline | CBAM |
|---|---:|---:|
| Best Epoch | 7 | 9 |
| Precision | 0.8953 | 0.9156 |
| Recall | 0.8000 | 0.8447 |
| Macro F1 | 0.8341 | 0.8777 |
| Micro F1 | 0.8989 | 0.9128 |
| ROC AUC | 0.9860 | 0.9915 |
| Training Time (min) | 7.66 | 10.70 |

## Segmentation

| Metric | Baseline | CBAM |
|---|---:|---:|
| Final Epoch | 40 | 40 |
| Validation Loss | 0.1093 | 0.1089 |
| Dice | 0.8372 | 0.8368 |
| IoU | 0.8114 | 0.8116 |
| Pixel Accuracy | 0.9896 | 0.9905 |
| Best Dice Epoch | 40 | 38 |
| Best Dice | 0.8372 | 0.8393 |
