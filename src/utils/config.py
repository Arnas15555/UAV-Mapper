from __future__ import annotations

"""
Shared defaults for all pipeline parameters.
Both the UI and the worker classes draw from this dict,
so there is a single source of truth.

Defaults are tuned for low-altitude UAV footage:
- panorama mode handles perspective/parallax better than scans
- lower min_motion_px / target_motion_px to retain frames from slow UAV movement
- higher orb_nfeatures for more robust feature matching
- max_frames_for_stitch increased so more frames contribute to the mosaic
"""

DEFAULTS: dict = {
    # Extraction
    "seconds_step":          0.50,
    "max_frames":            200,
    "extract_megapix":       2.0,
    "similar_threshold":     8.0,

    # Stitching
    "stitch_mode":           "panorama",   # FIX: was "scans"
    "work_megapix":          2.0,          # FIX: was 1.5
    "min_keypoints":         100,          # FIX: was 120
    "orb_nfeatures":         4000,         # FIX: was 2000
    "min_motion_px":         3.0,          # FIX: was hard-coded 8.0
    "target_motion_px":      20.0,         # FIX: was hard-coded 40.0
    "max_frames_for_stitch": 120,          # FIX: was hard-coded 60
}