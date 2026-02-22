from __future__ import annotations

"""
Shared defaults for all pipeline parameters.
Both the UI and the worker classes draw from this dict,
so there is a single source of truth.
"""

DEFAULTS: dict = {
    "seconds_step":       0.50,
    "max_frames":         30,
    "stitch_mode":        "scans",
    "work_megapix":       1.50,
    "min_keypoints":      120,
    "orb_nfeatures":      2000,
    "extract_megapix":    2.0,
    "similar_threshold":  6.0,
}