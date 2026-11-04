"""
Training-free NIR vein segmentation: pipeline, morphology, and metrics.

Usage:
    python vein_pipeline.py --images PATH/TO/images --masks PATH/TO/masks --out metrics.csv

Image files:  raw_<N>.png
Mask files:   raw_<N>_mask.png
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np
from scipy import ndimage, stats
from skimage import io
from skimage.morphology import skeletonize


# ───────────────────────── Segmentation pipeline ─────────────────────────

def segment_veins(bgr):
    """Return a binary vein mask (255 = vein) from a BGR NIR image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16))
    c1 = clahe.apply(hist)
    clamped = np.uint8(np.interp(c1, [15, 255], [75, 255]))
    c2 = clahe.apply(clamped)
    c2 = cv2.addWeighted(c2, 0.8, np.zeros_like(c2), 1, 1)

    filtered = cv2.bilateralFilter(c2, 7, 250, 250)

    b1 = cv2.threshold(filtered, 50, 205, cv2.THRESH_BINARY)[1]
    median = cv2.medianBlur(b1, 5)
    gaussian = cv2.GaussianBlur(b1, (15, 15), 3, 3)

    b2 = cv2.threshold(gaussian, 10, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(median)
    cv2.drawContours(contour_mask, contours, -1, 255, 5)

    blended = np.where(contour_mask == 255, gaussian, median).astype(np.uint8)
    final = cv2.medianBlur(blended, 5)
    return cv2.inRange(final, 0, 5)   # veins appear dark in NIR


# ───────────────────────── Morphology ─────────────────────────

def morphological_skeleton(binary):
    """Skeleton by repeated opening and erosion with a cross element."""
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = cv2.erode(img, element)
    return skel


def extract_morphology(binary):
    """Return centerline and boundary masks (255 = feature)."""
    centerlines = morphological_skeleton(binary)
    centerlines = np.where(centerlines >= 200, 255, 0).astype(np.uint8)
    boundaries = cv2.Canny(binary, 10, 100)
    boundaries = np.where(boundaries >= 200, 255, 0).astype(np.uint8)
    return centerlines, boundaries


def make_overlay(gray, centerlines, boundaries):
    """RGB visualization held in memory: centerlines in green, boundaries in red."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[boundaries == 255] = (255, 0, 0)
    rgb[centerlines == 255] = (0, 255, 0)
    return rgb


# ───────────────────────── Metrics ─────────────────────────

def _surface(mask):
    """Boundary pixels of a boolean mask."""
    return mask & ~ndimage.binary_erosion(mask)


def _surface_distances(a, b):
    """Distances (pixels) from surface pixels of a to the surface of b.
    Returns None if either surface is empty."""
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return None
    dist_to_b = ndimage.distance_transform_edt(~sb)
    return dist_to_b[sa]


def compute_metrics(pred_mask, gt_mask):
    eps = 1e-7
    pred = pred_mask > 127
    gt = gt_mask > 127

    tp = np.sum(pred & gt)
    tn = np.sum(~pred & ~gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    dice = 2 * tp / (2 * tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    # clDice: topology-aware overlap of skeletons
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    tprec = np.sum(pred_skel & gt) / (np.sum(pred_skel) + eps)
    tsens = np.sum(gt_skel & pred) / (np.sum(gt_skel) + eps)
    cldice = 2 * tprec * tsens / (tprec + tsens + eps)

    # HD95 and ASSD in pixels, computed on boundary pixels
    d_pg = _surface_distances(pred, gt)
    d_gp = _surface_distances(gt, pred)
    if d_pg is None or d_gp is None:
        hd95 = assd = np.nan
    else:
        all_d = np.concatenate([d_pg, d_gp])
        hd95 = np.percentile(all_d, 95)
        assd = all_d.mean()

    return {
        'Dice': dice,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'clDice': cldice,
        'HD95': hd95,
        'ASSD': assd,
    }


# ───────────────────────── Data handling ─────────────────────────

def discover_pairs(image_dir, mask_dir):
    pattern = re.compile(r'^raw_(\d+)\.png$')
    pairs = []
    for fname in sorted(os.listdir(image_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        mask_path = os.path.join(mask_dir, f'raw_{idx}_mask.png')
        if os.path.exists(mask_path):
            pairs.append((idx, os.path.join(image_dir, fname), mask_path))
        else:
            print(f'Skipping index {idx}: mask not found')
    return sorted(pairs)


def load_mask(path):
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]


# ───────────────────────── Main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='folder with raw_<N>.png')
    parser.add_argument('--masks', required=True, help='folder with raw_<N>_mask.png')
    parser.add_argument('--out', default='metrics.csv', help='per-image CSV output')
    args = parser.parse_args()

    pairs = discover_pairs(args.images, args.masks)
    if not pairs:
        raise SystemExit('No valid image/mask pairs found.')
    print(f'Found {len(pairs)} image/mask pairs.')

    rows = []
    keys = ['Dice', 'Sensitivity', 'Specificity', 'clDice', 'HD95', 'ASSD']

    for idx, img_path, mask_path in pairs:
        image = cv2.imread(img_path)
        gt = load_mask(mask_path)
        if image is None:
            print(f'Cannot load image {idx}, skipping')
            continue
        if image.shape[:2] != gt.shape:
            raise ValueError(f'Shape mismatch for index {idx}: '
                             f'image {image.shape[:2]} vs mask {gt.shape}')

        binary = segment_veins(image)
        extract_morphology(binary)  # morphology is computed; not used for metrics

        metrics = compute_metrics(binary, gt)
        rows.append({'Image': os.path.basename(img_path), **metrics})
        print(f'{idx}: ' + '  '.join(f'{k}={metrics[k]:.4f}' for k in keys))

    # Per-image CSV
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Image'] + keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f'{row[k]:.4f}' if k != 'Image' else row[k])
                             for k in ['Image'] + keys})
    print(f'\nPer-image metrics saved to {args.out}')

    # Summary with mean ± SD and 95% CI
    print('\nMetric summary (mean ± SD, 95% CI)')
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        mean, sd = vals.mean(), vals.std(ddof=1)
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sd / np.sqrt(n))
        print(f'{k:<12}: {mean:.4f} ± {sd:.4f}   95% CI ({lo:.4f}, {hi:.4f})   n={n}')


if __name__ == '__main__':
    main()