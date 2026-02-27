"""
Optical flow processing.

Wraps the layered-flow and vvad_daily binaries. Generates config files,
runs the binaries, and manages outputs.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from swirlhpc.calibration import get_calibration_offset
from swirlhpc.config import SwirlHPCConfig
from swirlhpc.odim import DopplerNotFoundError, ODIMMetadata

logger = logging.getLogger(__name__)


@dataclass
class FlowResult:
    """Result of processing a single timestep through layered-flow."""

    flow_file: str  # NetCDF optical flow output
    vvad_file: str  # VAD .dat file
    vvad_json: str  # VAD JSON file
    rid: int
    timestamp: str


def _write_flow_config(
    config: SwirlHPCConfig,
    metadata: ODIMMetadata,
    calib_offset: float,
) -> str:
    """
    Generate and write the layered-flow config file.

    Returns the path to the generated config file.
    """
    # Use configured velocity field override, or auto-detect from the file
    if config.layered_flow.velocity:
        velocity_field = config.layered_flow.velocity
        logger.info("Using configured velocity field: %s", velocity_field)
    else:
        try:
            velocity_field = metadata.get_doppler_field()
        except DopplerNotFoundError:
            raise

    config_dir = config.paths.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / f"flow.{metadata.rid}.conf"

    content = config.layered_flow.generate_config_text(
        radar_lon=metadata.lon,
        radar_lat=metadata.lat,
        velocity_field=velocity_field,
        topography_path=str(config.paths.topography),
        calib_offset=calib_offset,
    )

    with open(config_file, "w") as f:
        f.write(content)

    return str(config_file)


def read_dat_file(filename: str) -> pd.DataFrame:
    """Read a vvad_daily .dat output file into a DataFrame."""
    with open(filename, "r") as f:
        lines = f.readlines()

    data = [[float(val) for val in line.split()] for line in lines[5:] if line.split()]
    columns = [
        "z",
        "npts",
        "error",
        "u0",
        "v0",
        "w0",
        "vt",
        "divergence",
        "stretching",
        "shearing",
    ]
    df = pd.DataFrame(data, columns=columns).drop(columns=["error"]).replace(-999.0, np.nan)
    df["npts"] = df["npts"].astype(int)
    df["z"] *= 1000  # km to metres
    return df


def run_vvad(
    config: SwirlHPCConfig,
    pvol_file: str,
    vvad_outfile: str,
) -> str:
    """
    Run vvad_daily on a pvol file.

    Parameters
    ----------
    config : SwirlHPCConfig
    pvol_file : str
        Path to the input .pvol.h5 file.
    vvad_outfile : str
        Path for the output .dat file.

    Returns
    -------
    str
        Path to the generated .dat file.
    """
    cmd = f"{config.binaries.vvad_daily} {pvol_file} {vvad_outfile}"
    logger.info("Running vvad_daily: %s", cmd)

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env=config.binaries.get_env(),
    )
    if result.returncode != 0:
        logger.warning("vvad_daily returned %s: %s", result.returncode, result.stderr.strip())

    if not os.path.isfile(vvad_outfile):
        raise FileNotFoundError(f"vvad_daily did not produce output: {vvad_outfile}")

    logger.info("Generated VAD file: %s", vvad_outfile)
    return vvad_outfile


def vvad_to_json(vvad_file: str, json_file: str) -> str:
    """Convert a VAD .dat file to JSON."""
    df = read_dat_file(vvad_file)
    with open(json_file, "w") as f:
        df.to_json(f)

    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"Failed to write VAD JSON: {json_file}")

    logger.info("Generated VAD JSON: %s", json_file)
    return json_file


def run_layered_flow(
    config: SwirlHPCConfig,
    config_file: str,
    lag_file: str,
    current_file: str,
    flow_outfile: str,
    vad_file: str,
) -> str:
    """
    Run the layered-flow binary.

    Parameters
    ----------
    config : SwirlHPCConfig
    config_file : str
        Path to the layered-flow .conf file.
    lag_file : str
        Path to the lagging (t-5min) pvol file.
    current_file : str
        Path to the current (t) pvol file.
    flow_outfile : str
        Path for the output flow .nc file.
    vad_file : str
        Path to the VAD .dat file.

    Returns
    -------
    str
        Path to the generated flow .nc file.
    """
    cmd = f"{config.binaries.layered_flow} {config_file} " f"{lag_file} {current_file} {flow_outfile} --vad {vad_file}"
    logger.info("Running layered-flow: %s", cmd)

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env=config.binaries.get_env(),
    )
    if result.returncode != 0:
        logger.warning("layered-flow returned %s: %s", result.returncode, result.stderr.strip())

    if not os.path.isfile(flow_outfile):
        raise FileNotFoundError(f"layered-flow did not produce output: {flow_outfile}")

    logger.info("Generated flow file: %s", flow_outfile)
    return flow_outfile


def process_flow_timestep(
    config: SwirlHPCConfig,
    current_pvol: str,
    lag_pvol: str,
    vad_file: Optional[str] = None,
) -> FlowResult:
    """
    Run the full optical-flow processing for a single timestep.

    This is the main entry point for flow processing. It:
    1. Reads metadata from the current pvol file
    2. Gets the calibration offset
    3. Generates the layered-flow config
    4. Runs vvad_daily (if no pre-existing VAD file)
    5. Runs layered-flow
    6. Converts VAD output to JSON

    Parameters
    ----------
    config : SwirlHPCConfig
    current_pvol : str
        Path to the current (time t) pvol.h5 file on disk.
    lag_pvol : str
        Path to the lagging (time t-5min) pvol.h5 file on disk.
    vad_file : str, optional
        Path to a pre-existing VAD .dat file. If None, vvad_daily is run.

    Returns
    -------
    FlowResult
    """
    metadata = ODIMMetadata.from_file(current_pvol)
    rid = metadata.rid
    datestr = metadata.datetime.strftime("%Y%m%d")
    dtimestr = metadata.datetime.strftime("%Y%m%d_%H%M00")

    # Output paths
    flow_dir = config.paths.flow_dir(rid, datestr)
    vvad_dir = config.paths.vvad_dir(rid, datestr)

    flow_outfile = str(flow_dir / f"{rid}_{dtimestr}_flow.nc")
    vvad_outfile = str(vvad_dir / f"{rid}_{dtimestr}_vvad.dat")
    json_outfile = str(vvad_dir / f"IDR{rid}VAD1_{dtimestr}.json")

    # Check for existing flow output — skip the whole timestep
    if not config.processing.overwrite and os.path.isfile(flow_outfile):
        logger.info("Flow output already exists, skipping: %s", flow_outfile)
        # Use existing VAD if available, otherwise point to expected path
        existing_vvad = vvad_outfile if os.path.isfile(vvad_outfile) else vvad_outfile
        existing_json = json_outfile if os.path.isfile(json_outfile) else json_outfile
        return FlowResult(
            flow_file=flow_outfile,
            vvad_file=existing_vvad,
            vvad_json=existing_json,
            rid=rid,
            timestamp=dtimestr,
        )

    # Calibration
    calib_offset = get_calibration_offset(rid, config.paths.calib_dir)

    # Generate layered-flow config
    flow_config = _write_flow_config(config, metadata, calib_offset)

    # Run VAD if output doesn't already exist
    if vad_file is not None and os.path.isfile(vad_file):
        logger.info("Using provided VAD file: %s", vad_file)
    elif os.path.isfile(vvad_outfile):
        logger.info("VAD output already exists, reusing: %s", vvad_outfile)
        vad_file = vvad_outfile
    else:
        logger.info("Running vvad_daily for %s", os.path.basename(current_pvol))
        vad_file = run_vvad(config, current_pvol, vvad_outfile)

    # Convert VAD to JSON (skip if it already exists)
    if not config.processing.overwrite and os.path.isfile(json_outfile):
        logger.info("VAD JSON already exists, reusing: %s", json_outfile)
        json_file = json_outfile
    else:
        json_file = vvad_to_json(vad_file, json_outfile)

    # Run layered-flow
    run_layered_flow(config, flow_config, lag_pvol, current_pvol, flow_outfile, vad_file)

    return FlowResult(
        flow_file=flow_outfile,
        vvad_file=vad_file,
        vvad_json=json_file,
        rid=rid,
        timestamp=dtimestr,
    )
