# Unsupervised Segmentation of Lower-Limb Veins in Clinical Near-Infrared Images with Morphology Visualization

Training-free pipeline for unsupervised segmentation and morphology visualization (boundaries and centerlines) of superficial veins in clinical near-infrared (NIR) images of the lower limbs.

The method uses classical image processing only. It requires no training data and no GPU.

## Contents

```
.
├── vein_pipeline.py     # segmentation, morphology extraction, and metrics
├── requirements.txt    
└── README.md
```

## Method Overview

1. **Preprocessing:** grayscale conversion, global histogram equalization, and two CLAHE passes (clip limit 8.0, 16×16 tiles) with intensity remapping.
2. **Filtering:** bilateral filter (diameter 7, σ_color 250, σ_space 250), median blur, and Gaussian blur.
3. **Thresholding and contours:** two binary thresholds, contour detection, and a contour-guided blend followed by a median blur.
4. **Vein mask:** pixels in the range 0–5 after the final blur are kept, since veins appear dark in NIR.
5. **Morphology:**
   - Centerlines by morphological skeletonization (cross-shaped 3×3 element).
   - Boundaries by Canny edge detection (thresholds 10 and 100).
6. **Evaluation:** Dice, sensitivity, specificity, clDice, HD95, and ASSD against manually annotated masks.

## Requirements

- Python 3.10.18
- OpenCV (`opencv-python`)
- NumPy
- SciPy
- scikit-image

Install with:

```bash
pip install -r requirements.txt
```

## Data Format

Place the images and masks in separate folders:

```
data/
├── images/
│   ├── raw_1.png
│   └── ...
└── masks/
    ├── raw_1_mask.png
    └── ...
```

- **Images:** `raw_<N>.png`, 8-bit NIR, BGR or grayscale.
- **Masks:** `raw_<N>_mask.png`, with the same dimensions as the image.
- Images without a matching mask are skipped. Masks are not resized; a shape mismatch raises an error.

The clinical dataset is not publicly available due to patient privacy restrictions. See the paper's data availability statement for access conditions.

## Usage

### Segmentation and metrics

```bash
python vein_pipeline.py --images data/images --masks data/masks --out metrics.csv
```

This prints per-image scores, writes `metrics.csv`, and reports the mean, standard deviation, and 95% confidence interval for each metric.

### Per-stage timing

```bash
python fps_benchmark.py --images data/images --masks data/masks --repeats 3
```

This reports FPS for segmentation only, morphology only, and the combined pipeline. Image loading and file saving are excluded.

## Metrics

| Metric | Description |
|---|---|
| Dice | Overlap between predicted and ground-truth vein masks |
| Sensitivity | Fraction of ground-truth vein pixels detected |
| Specificity | Fraction of background pixels correctly rejected |
| clDice | Overlap of skeletons, measuring topological agreement |
| HD95 | 95th percentile of symmetric boundary distances, in pixels |
| ASSD | Average symmetric surface distance, in pixels |

HD95 and ASSD are reported as NaN when either mask is empty, and these cases are excluded from the summary.





## License

`[Add license, e.g., MIT.]`
