from __future__ import annotations

import cv2
import numpy as np


def _valid_mask(pano_bgr: np.ndarray, thresh: int = 1) -> np.ndarray:
    """
    Binary mask of non-black pixels. Works even if pano is float/odd types.
    """
    if pano_bgr is None or pano_bgr.size == 0:
        raise ValueError("Empty image passed to postprocess")

    if pano_bgr.dtype != np.uint8:
        img = np.clip(pano_bgr, 0, 255).astype(np.uint8)
    else:
        img = pano_bgr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return mask


def crop_black(pano_bgr: np.ndarray, thresh: int = 1) -> np.ndarray:
    """
    Crops to the bounding box of non-black pixels.
    Safe against empty masks.
    """
    mask = _valid_mask(pano_bgr, thresh=thresh)
    if cv2.countNonZero(mask) == 0:
        return pano_bgr

    x, y, w, h = cv2.boundingRect(mask)
    return pano_bgr[y : y + h, x : x + w]


def auto_rotate(
    pano_bgr: np.ndarray,
    canny1: int = 50,
    canny2: int = 150,
    hough_thresh: int = 150,
    max_abs_angle_deg: float = 15.0,
) -> np.ndarray:
    """
    Attempts to deskew panorama using dominant Hough line angles.
    - Using angle filtering
    - Clamps rotation to max_abs_angle_deg
    """
    gray = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
    if lines is None:
        return pano_bgr

    angles = []
    for rho, theta in lines[:, 0, :]:
        # Convert to degrees around horizontal baseline
        ang = (theta - np.pi / 2) * 180.0 / np.pi
        # Keep near-horizontal lines only
        if abs(ang) <= max_abs_angle_deg:
            angles.append(ang)

    if not angles:
        return pano_bgr

    angle = float(np.median(angles))
    if abs(angle) < 0.1:
        return pano_bgr

    h, w = pano_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        pano_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def crop_largest_inner_rect(
    pano_bgr: np.ndarray,
    shrink_iters: int = 3,
    thresh: int = 1,
    close_iters: int = 2,
    kernel_scale: float = 0.01,
    use_rotated_rect: bool = False,
) -> np.ndarray:

    mask = _valid_mask(pano_bgr, thresh=thresh)
    if cv2.countNonZero(mask) == 0:
        return pano_bgr

    h, w = mask.shape[:2]

    # Adaptive kernel size based on image size
    k = max(3, int(min(h, w) * kernel_scale))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Close small holes / connect regions
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))

    # Erode to remove spikes/protrusions
    if shrink_iters > 0:
        mask2 = cv2.erode(mask, kernel, iterations=int(shrink_iters))
    else:
        mask2 = mask

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Too aggressive erosion. Fall back to non-eroded bbox crop.
        return crop_black(pano_bgr, thresh=thresh)

    c = max(contours, key=cv2.contourArea)

    if use_rotated_rect:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Get axis-aligned bounding box of the rotated rectangle
        x, y, ww, hh = cv2.boundingRect(box)
        x = max(0, x)
        y = max(0, y)
        ww = min(w - x, ww)
        hh = min(h - y, hh)
        return pano_bgr[y : y + hh, x : x + ww]

    # Default: axis-aligned bounding rect
    x, y, ww, hh = cv2.boundingRect(c)
    return pano_bgr[y : y + hh, x : x + ww]

