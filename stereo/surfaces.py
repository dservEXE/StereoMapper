"""Explicitly estimated planar surfaces, separate from measured stereo depth."""
import cv2
import numpy as np


def plane_estimates(disparity, valid, image, allowed, segmenter=None):
    result = np.zeros(disparity.shape, np.float32)
    estimated = np.zeros(valid.shape, bool)
    if valid.sum() < 80 or segmenter is None:
        return result, estimated
    # This is a visual estimate, so segmentation need not run at measurement resolution.
    height,width = image.shape[:2]
    if width>480:
        small = cv2.resize(image,(480,round(height*480/width)),interpolation=cv2.INTER_AREA)
        labels = cv2.resize(segmenter.processImage(small),(width,height),interpolation=cv2.INTER_NEAREST)
    else:
        labels = segmenter.processImage(image)
    counts = np.bincount(labels.ravel())
    support = np.bincount(labels[valid], minlength=len(counts))
    selected = np.where((counts >= 400) & (support >= 60) & (support >= counts * .03))[0]
    selected = sorted(selected, key=lambda label: counts[label], reverse=True)[:16]
    for label in selected:
        region = labels == label
        yy, xx = np.where(region & valid)
        if len(xx) > 1000:
            indices = np.linspace(0, len(xx) - 1, 1000).astype(int)
            xx, yy = xx[indices], yy[indices]
        ry, rx = np.where(region)
        if np.ptp(xx) < .65 * np.ptp(rx) or np.ptp(yy) < .65 * np.ptp(ry):
            continue
        # Normalized coordinates keep the plane fit well conditioned.
        cx, cy = float(xx.mean()), float(yy.mean())
        scale = max(1., float(max(np.ptp(xx), np.ptp(yy))))
        design = np.column_stack(((xx - cx) / scale, (yy - cy) / scale, np.ones(len(xx))))
        values = disparity[yy, xx]
        if np.linalg.matrix_rank(design) < 3:
            continue
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = np.abs(design @ coefficients - values)
        inliers = residual <= .75
        if inliers.mean() < .95:
            continue
        coefficients = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)[0]
        hull = cv2.convexHull(np.column_stack((xx[inliers], yy[inliers])).astype(np.int32))
        bounded = np.zeros(valid.shape, np.uint8)
        cv2.fillConvexPoly(bounded, hull, 1)
        fill = region & bounded.astype(bool) & allowed & ~valid
        fy, fx = np.where(fill)
        if not len(fx):
            continue
        predicted = ((fx - cx) / scale * coefficients[0] +
                     (fy - cy) / scale * coefficients[1] + coefficients[2])
        good = np.isfinite(predicted) & (predicted > .5)
        result[fy[good], fx[good]] = predicted[good]
        estimated[fy[good], fx[good]] = True
    return result, estimated
