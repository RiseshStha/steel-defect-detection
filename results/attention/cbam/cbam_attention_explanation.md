# CBAM Attention Visualization Notes

Channel attention assigns a learned weight to each high-level feature channel after the ResNet18 encoder. Higher channel weights indicate feature detectors that CBAM considered more relevant for the current steel image.

Spatial attention then compresses the channel-refined feature map using average and max pooling across channels. The resulting heatmap highlights image regions that the model emphasizes before classification.

In the generated figures, warmer spatial-attention regions indicate stronger CBAM focus. When these regions coincide with visible scratches, stains, edge defects, or texture disruptions, the visualization supports the interpretation that CBAM is directing the classifier toward defect-relevant image areas rather than only background steel texture.

The channel bar plot should be interpreted as feature importance within the latent encoder representation, not as direct pixel-level localization. The spatial overlay provides the localization cue.
