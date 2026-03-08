from __future__ import annotations

"""
Shared defaults for all pipeline parameters.
Both the UI and the worker classes draw from this single source of truth.

Settings auto-scale based on whether a CUDA-capable GPU is present:
  - GPU path: tuned for 4K (3840×2160) 30 fps drone footage (~20 s)
  - CPU path: reduced resolution and feature counts to keep processing time reasonable
"""

import cv2

_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

DEFAULTS: dict = {
    # Extraction
    "seconds_step":          0.33 if _CUDA else 0.5,
    "max_frames":            120  if _CUDA else 60,
    "extract_megapix":       4.0  if _CUDA else 2.0,
    "similar_threshold":     10.0,
    "blur_threshold":        0.0,

    # Stitching
    "stitch_mode":           "panorama",
    "work_megapix":          3.0  if _CUDA else 1.5,
    "min_keypoints":         150  if _CUDA else 100,
    "orb_nfeatures":         8000 if _CUDA else 3000,
    "min_motion_px":         5.0,
    "target_motion_px":      25.0,
    "max_frames_for_stitch": 80   if _CUDA else 40,
}