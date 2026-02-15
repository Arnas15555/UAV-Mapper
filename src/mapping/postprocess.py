import cv2
import numpy as np

def crop_black(pano_bgr):
    gray = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(th)
    return pano_bgr[y:y+h, x:x+w]

def auto_rotate(pano_bgr):
    gray = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return pano_bgr

    angles = []
    for rho, theta in lines[:, 0, :]:
        ang = (theta - np.pi / 2) * 180 / np.pi
        angles.append(ang)

    angle = float(np.median(angles))
    h, w = pano_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(pano_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

def crop_largest_inner_rect(pano_bgr, shrink_iters: int = 3):
    """
    Finds a big rectangle guaranteed to be inside the valid (non-black) region.
    We do it by shrinking the mask a bit so spikes disappear, then crop bbox.
    """
    gray = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Smooth mask and remove spikes by eroding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Erode to kill protrusions (tune shrink_iters)
    mask2 = cv2.erode(mask, kernel, iterations=shrink_iters)

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pano_bgr  # erosion may remove everything if too aggressive

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop original pano using rect from eroded mask
    return pano_bgr[y:y+h, x:x+w]
