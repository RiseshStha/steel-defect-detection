# Segmentation Explainability Notes

The error map compares the binary union of all four defect channels against the predicted union mask.

Green pixels are true positives, where the model prediction overlaps the ground truth. Red pixels are false negatives, where a ground-truth defect was missed. Orange pixels are false positives, where the model predicted a defect outside the annotated mask.

The difference map is computed as prediction minus ground truth for each class and clipped to the range [-1, 1]. Blue regions indicate missed defect pixels, while red regions indicate over-predicted defect pixels.

These maps are intended to support qualitative discussion of segmentation behavior, including boundary errors, missed fine scratches, and over-segmentation of steel texture.
