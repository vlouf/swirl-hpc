"""
Radar reflectivity calibration.

Reads S3CAR calibration JSON files to obtain reflectivity offsets.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def get_calibration_offset(rid: int, calib_dir: str) -> float:
    """
    Get the radar reflectivity calibration offset from S3CAR data.

    Parameters
    ----------
    rid : int
        Radar ID.
    calib_dir : str
        Path to directory containing s3car.{rid}.json files.
        If empty string, returns 0.0 (no calibration).

    Returns
    -------
    float
        Calibration offset in dB. Returns 0.0 if no calibration file exists.
    """
    if not calib_dir:
        return 0.0

    filepath = os.path.join(calib_dir, f"s3car.{rid}.json")
    if not os.path.isfile(filepath):
        logger.debug("No calibration file for radar %s at %s", rid, filepath)
        return 0.0

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            offset = data["best_calibration_estimate"]
            logger.info("Radar %s calibration offset: %.2f dB", rid, offset)
            return float(offset)
    except (KeyError, TypeError):
        logger.warning("Calibration file for radar %s exists but has no valid estimate", rid)
        return 0.0
    except Exception as err:
        logger.error("Failed to read calibration file %s: %r", filepath, err)
        return 0.0
