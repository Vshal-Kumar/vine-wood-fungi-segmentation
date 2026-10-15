# Synthetic mask threshold comparison

This is a **diagnostic experiment only**. It did not modify the processed masks or the configured training threshold.

- Synthetic masks inspected: 427
- Current preprocessing rule: `fixed grayscale >= 128`
- Per-mask Otsu thresholds (min/median/max): [104, 114, 119]
- Pooled-dataset Otsu threshold: 114
- Pixels changed using pooled Otsu vs current rule: 62,911 / 27,983,872 (0.225%)
- Pixels changed using per-mask Otsu vs current rule: 67,448 / 27,983,872 (0.241%)
- Foreground pixels: current=1,795,699; pooled Otsu=1,858,610; per-mask Otsu=1,863,147.
- Visual comparison: `reports/visualizations/otsu_mask_threshold_comparison.png`

## Interpretation

Otsu is a histogram-based binarizer, not an edge detector and not a measure of annotation correctness. It chooses a threshold that separates grayscale intensities; the dataset README does not define what each synthetic gray level means. Here Otsu thresholds are consistently below the current midpoint and would label more faint boundary pixels as fungus. That may thicken/expand boundaries rather than improve ground truth.

Keep the current midpoint labels as the baseline for now. If comparing Otsu, make it a separate training-data variant, train otherwise identical models, select using validation only, and inspect thin structures/boundaries. Do not replace or tune labels using the held-out test set.

Edge methods such as Sobel/Canny should likewise not replace masks. A future experiment can add a derived edge input channel or boundary-aware training loss, but that requires a model-specific implementation and validation comparison. No edge filtering was applied to images or masks in this preparation run.
