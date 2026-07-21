# Classification Results

Best epoch is selected using the same criterion as the checkpoint saver: lowest validation loss.

| Model | Best Epoch | Precision | Recall | Macro F1 | Micro F1 | ROC AUC | Training Time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline ResNet18 | 7 | 0.8953 | 0.8000 | 0.8341 | 0.8989 | 0.9860 | 7.66 |
| CBAM-ResNet18 | 9 | 0.9156 | 0.8447 | 0.8777 | 0.9128 | 0.9915 | 10.70 |

Peak Macro F1 is retained as a secondary diagnostic because the saved checkpoint is loss-selected.

| Model | Peak Macro F1 Epoch | Peak Macro F1 | Epochs Completed |
|---|---:|---:|---:|
| Baseline ResNet18 | 11 | 0.8598 | 12 |
| CBAM-ResNet18 | 9 | 0.8777 | 14 |
