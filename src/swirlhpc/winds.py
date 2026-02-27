"""
3D wind retrieval processing.

Wraps the 3dwinds and dvad_2radars_daily binaries. Handles both single-radar
and multi-Doppler (region) wind retrievals.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import netCDF4
import numpy as np
import pandas as pd
import pyproj

import aura

from swirlhpc.config import SwirlHPCConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_radar_location(rid: int) -> Tuple[float, float]:
    """Get (lat, lon) for a radar from the AURA metadata."""
    radar = aura.get_radar(rid)
    return radar.lat, radar.lon


def _get_domain_centre(flow_files: List[str]) -> Tuple[float, float]:
    """Compute the barycentre (lon, lat) of a set of flow NetCDF files."""
    longitudes = []
    latitudes = []
    for f in flow_files:
        with netCDF4.Dataset(f, keepweakref=True) as ncid:
            lat = ncid["latitude"][:]
            lon = ncid["longitude"][:]
            longitudes.extend([lon.min(), lon.max()])
            latitudes.extend([lat.min(), lat.max()])

    bar_lon = (np.min(longitudes) + np.max(longitudes)) / 2
    bar_lat = (np.min(latitudes) + np.max(latitudes)) / 2
    return float(bar_lon), float(bar_lat)


def _get_nxny(
    bar_lon: float,
    bar_lat: float,
    rids: List[int],
    dxy: float,
    maxrange: float = 150e3,
) -> Tuple[int, int]:
    """
    Compute the (nx, ny) grid dimensions for a multi-Doppler domain.
    """
    proj = pyproj.Proj(
        f"+proj=aea +lon_0={bar_lon} +lat_0={bar_lat} "
        f"+lat_1=-18 +lat_2=-36 +units=m +ellps=GRS80"
    )

    coords = []
    for rid in rids:
        lat, lon = _get_radar_location(rid)
        coords.append(proj(lon, lat))

    x_vals, y_vals = zip(*coords)
    nx = int(np.ceil(((max(x_vals) - min(x_vals)) + 2 * maxrange) / dxy))
    ny = int(np.ceil(((max(y_vals) - min(y_vals)) + 2 * maxrange) / dxy))
    return nx, ny


def _euclidean_distance_sort(rids: List[int]) -> List[int]:
    """Sort radar IDs by distance from the barycentre (closest first)."""
    if len(rids) < 2:
        return list(rids)

    lats, lons = [], []
    for rid in rids:
        lat, lon = _get_radar_location(rid)
        lats.append(lat)
        lons.append(lon)

    lats = np.array(lats)
    lons = np.array(lons)
    bar_lat = lats.mean()
    bar_lon = lons.mean()
    distances = np.hypot(lats - bar_lat, lons - bar_lon)

    sorted_indices = np.argsort(distances)
    return [rids[i] for i in sorted_indices]


def _get_vvad_path_from_flow(flow_path: str) -> str:
    """
    Derive the VVAD .dat path from a flow .nc path by substituting
    the directory and suffix.
    """
    return flow_path.replace("flow.nc", "vvad.dat").replace("/flow/", "/vvad/")


# ---------------------------------------------------------------------------
# DVAD
# ---------------------------------------------------------------------------

def run_dvad(
    config: SwirlHPCConfig,
    vvad_files: List[Tuple[int, str]],
    region_name: str,
    date: pd.Timestamp,
) -> str:
    """
    Run DVAD for all pairs of radars in a region.

    Parameters
    ----------
    config : SwirlHPCConfig
    vvad_files : list of (rid, vvad_path) tuples
        One entry per radar at the current timestep (not the lag).
    region_name : str
        Name identifier for this multi-Doppler region.
    date : pd.Timestamp
        Timestamp being processed.

    Returns
    -------
    str
        Path to the DVAD output directory (with trailing slash for Fortran).
    """
    datestr = date.strftime("%Y%m%d")
    dvad_dir = str(config.paths.dvad_dir(region_name, datestr))
    if not dvad_dir.endswith("/"):
        dvad_dir += "/"

    env = config.binaries.get_env()
    for (r0, f0), (r1, f1) in combinations(vvad_files, 2):
        cmd = [
            config.binaries.dvad_2radars_daily,
            str(r0), str(r1), str(f0), str(f1), dvad_dir,
        ]
        logger.info("Running dvad_2radars_daily: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.stdout.strip():
            logger.info(result.stdout.strip())
        if result.returncode != 0 and result.stderr.strip():
            logger.warning("dvad stderr: %s", result.stderr.strip())

    return dvad_dir


# ---------------------------------------------------------------------------
# 3D Winds
# ---------------------------------------------------------------------------

def run_3dwinds(
    config: SwirlHPCConfig,
    n_radars: int,
    r3dbrc_file: str,
    log_file: Optional[str] = None,
) -> None:
    """
    Run the 3dwinds binary.

    Parameters
    ----------
    config : SwirlHPCConfig
    n_radars : int
        Number of radars being processed.
    r3dbrc_file : str
        Path to the r3dbrc configuration file.
    log_file : str, optional
        If set, dump stdout/stderr to this file.
    """
    cmd = [config.binaries.three_d_winds, str(n_radars), r3dbrc_file]
    logger.info("Running 3dwinds: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, env=config.binaries.get_env())

    if log_file:
        with open(log_file, "a") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)

    # Log output, filtering noise
    for line in result.stdout.splitlines():
        line = line.rstrip()
        if not line or "iteration" in line:
            continue
        if "Number of valid retrieval grids is too low" in line:
            continue
        if "[ERROR]" in line:
            logger.error(line)
        elif "[WARN]" in line:
            logger.warning(line)
        else:
            logger.info(line)

    if result.stderr:
        for line in result.stderr.splitlines():
            if "IEEE_INVALID_FLAG" in line:
                logger.debug(line)
            elif line.strip():
                logger.error(line)

    logger.info("3dwinds finished with return code %s", result.returncode)


# ---------------------------------------------------------------------------
# Single radar winds
# ---------------------------------------------------------------------------

@dataclass
class WindsResult:
    """Result of a 3D winds retrieval."""
    output_3d: str        # 3D wind field NetCDF
    output_2d: str        # 2D (lowest-sweep) wind field NetCDF
    region_name: str
    n_radars: int


def process_single_radar_winds(
    config: SwirlHPCConfig,
    rid: int,
    radar_dtime: pd.Timestamp,
    flow_files: List[str],
) -> WindsResult:
    """
    Run single-radar 3D wind retrieval.

    Parameters
    ----------
    config : SwirlHPCConfig
    rid : int
        Radar ID.
    radar_dtime : pd.Timestamp
        Timestamp of the flow files.
    flow_files : list of str
        The flow .nc files for this radar: [lag_flow, current_flow].
        For single-radar without lag, pass [current_flow, current_flow].

    Returns
    -------
    WindsResult
    """
    region_name = str(rid)
    datestr = radar_dtime.strftime("%Y%m%d")
    dtimestr = radar_dtime.strftime("%Y%m%d_%H%M")

    # Output files
    winds_dir = config.paths.winds_dir(region_name, datestr)
    outfile_3d = str(winds_dir / f"{region_name}_{dtimestr}.nc")
    outfile_2d = str(winds_dir / f"{region_name}_{dtimestr}_lwstwind.nc")

    if not config.processing.overwrite:
        if os.path.isfile(outfile_3d) and os.path.isfile(outfile_2d):
            logger.info("Winds output already exists, skipping: %s", outfile_3d)
            return WindsResult(outfile_3d, outfile_2d, region_name, 1)

    # Radar location
    lat, lon = _get_radar_location(rid)

    # VVAD directory (for the VAD constraint)
    vvad_dir = str(config.paths.vvad_dir(rid, datestr))

    # Config files
    config_dir = config.paths.config_dir / datestr
    config_dir.mkdir(parents=True, exist_ok=True)
    r3d_main_file = str(config_dir / f"r3d_{region_name}_{dtimestr}.init")
    r3dbrc_file = str(config_dir / f"r3dbrc_{region_name}_{dtimestr}")

    # Generate configs
    w3d = config.three_d_winds
    w3d.generate_r3d_main_init(r3d_main_file, nx=w3d.nx_single, ny=w3d.ny_single, ivad=1)

    # flow_base_dir: where 3dwinds looks for flow files (the parent of rid dirs)
    flow_base_dir = str(config.paths.output_dir / "flow") + "/"

    w3d.generate_r3dbrc(
        r3dbrc_file,
        config_file=r3d_main_file,
        dvad_path=vvad_dir,
        flow_base_dir=flow_base_dir,
        infile_list=flow_files,
        output_files=[outfile_3d, outfile_2d],
        centre_lon=lon,
        centre_lat=lat,
    )

    # Run
    log_file = outfile_3d.replace(".nc", ".log")
    run_3dwinds(config, n_radars=1, r3dbrc_file=r3dbrc_file, log_file=log_file)

    if os.path.isfile(outfile_2d):
        logger.info("Single-radar winds generated: %s", outfile_3d)
    else:
        logger.error("Single-radar winds FAILED for radar %s at %s", rid, dtimestr)

    return WindsResult(outfile_3d, outfile_2d, region_name, 1)


# ---------------------------------------------------------------------------
# Multi-Doppler winds
# ---------------------------------------------------------------------------

def process_multidoppler_winds(
    config: SwirlHPCConfig,
    rids: List[int],
    radar_dtime: pd.Timestamp,
    flow_files_by_rid: Dict[int, List[str]],
    vvad_files_by_rid: Dict[int, str],
    region_name: Optional[str] = None,
) -> WindsResult:
    """
    Run multi-Doppler 3D wind retrieval for a region.

    Parameters
    ----------
    config : SwirlHPCConfig
    rids : list of int
        Radar IDs in this multi-Doppler region.
    radar_dtime : pd.Timestamp
        Timestamp being processed.
    flow_files_by_rid : dict
        {rid: [lag_flow_file, current_flow_file]} for each radar.
    vvad_files_by_rid : dict
        {rid: vvad_dat_file} for each radar at the current time.
    region_name : str, optional
        Name for this multi-Doppler region (e.g. "503"). If None,
        defaults to sorted radar IDs joined by underscore (e.g. "2_49_68").

    Returns
    -------
    WindsResult
    """
    sorted_rids = _euclidean_distance_sort(rids)
    if region_name is None:
        region_name = "_".join(str(r) for r in sorted(rids))
    datestr = radar_dtime.strftime("%Y%m%d")
    dtimestr = radar_dtime.strftime("%Y%m%d_%H%M")
    n_radars = len(sorted_rids)

    # Output files
    winds_dir = config.paths.winds_dir(region_name, datestr)
    outfile_3d = str(winds_dir / f"{region_name}_{dtimestr}.nc")
    outfile_2d = str(winds_dir / f"{region_name}_{dtimestr}_lwstwind.nc")

    if not config.processing.overwrite:
        if os.path.isfile(outfile_3d) and os.path.isfile(outfile_2d):
            logger.info("Multi-Doppler output already exists, skipping: %s", outfile_3d)
            return WindsResult(outfile_3d, outfile_2d, region_name, n_radars)

    # Build the interleaved file list: [lag0_r0, cur_r0, lag0_r1, cur_r1, ...]
    all_flow_files = []
    for rid in sorted_rids:
        all_flow_files.extend(flow_files_by_rid[rid])

    # Get domain centre from the current (non-lag) flow files
    current_flow_files = [flow_files_by_rid[rid][1] for rid in sorted_rids]
    bar_lon, bar_lat = _get_domain_centre(current_flow_files)

    # Run DVAD for all radar pairs
    vvad_tuples = [(rid, vvad_files_by_rid[rid]) for rid in sorted_rids]
    dvad_dir = run_dvad(config, vvad_tuples, region_name, radar_dtime)

    # Compute domain size
    w3d = config.three_d_winds
    nx, ny = _get_nxny(bar_lon, bar_lat, sorted_rids, dxy=w3d.dxy)

    # Config files
    config_dir = config.paths.config_dir / datestr
    config_dir.mkdir(parents=True, exist_ok=True)
    r3d_main_file = str(config_dir / f"r3d_{region_name}_{dtimestr}.init")
    r3dbrc_file = str(config_dir / f"r3dbrc_{region_name}_{dtimestr}")

    w3d.generate_r3d_main_init(r3d_main_file, nx=nx, ny=ny, ivad=0)

    flow_base_dir = str(config.paths.output_dir / "flow") + "/"
    w3d.generate_r3dbrc(
        r3dbrc_file,
        config_file=r3d_main_file,
        dvad_path=dvad_dir,
        flow_base_dir=flow_base_dir,
        infile_list=all_flow_files,
        output_files=[outfile_3d, outfile_2d],
        centre_lon=bar_lon,
        centre_lat=bar_lat,
    )

    # Run 3D winds
    log_file = outfile_3d.replace(".nc", ".log")
    run_3dwinds(config, n_radars=n_radars, r3dbrc_file=r3dbrc_file, log_file=log_file)

    if os.path.isfile(outfile_2d):
        logger.info("Multi-Doppler winds generated: %s (radars: %s)", outfile_3d, sorted_rids)
    else:
        logger.error("Multi-Doppler winds FAILED for region %s at %s", region_name, dtimestr)

    return WindsResult(outfile_3d, outfile_2d, region_name, n_radars)