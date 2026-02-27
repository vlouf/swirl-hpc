"""
Configuration management for SWIRL HPC.

Loads defaults from the bundled swirl_defaults.toml file, then merges any
user-provided overrides from a custom TOML file.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Dataclasses mirroring the TOML structure
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    output_dir: Path = Path("/scratch/swirl")
    scratch_dir: Optional[Path] = None
    topography: Path = Path("/g/data/rq0/admin/topography/australian_topography_1000m.nc")
    calib_dir: str = ""

    def __post_init__(self):
        self.output_dir = Path(os.path.expandvars(str(self.output_dir)))
        if self.scratch_dir is not None:
            self.scratch_dir = Path(os.path.expandvars(str(self.scratch_dir)))
        self.topography = Path(os.path.expandvars(str(self.topography)))
        self.calib_dir = os.path.expandvars(self.calib_dir)

    @property
    def effective_scratch_dir(self) -> Path:
        return self.scratch_dir if self.scratch_dir else self.output_dir / "scratch"

    @property
    def config_dir(self) -> Path:
        return self.output_dir / "config"

    def flow_dir(self, rid: int, datestr: str) -> Path:
        p = self.output_dir / "flow" / str(rid) / datestr
        p.mkdir(parents=True, exist_ok=True)
        return p

    def vvad_dir(self, rid: int, datestr: str) -> Path:
        p = self.output_dir / "vvad" / str(rid) / datestr
        p.mkdir(parents=True, exist_ok=True)
        return p

    def winds_dir(self, region_name: str, datestr: str) -> Path:
        p = self.output_dir / "winds" / region_name / datestr
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dvad_dir(self, region_name: str, datestr: str) -> Path:
        p = self.output_dir / "dvad" / region_name / datestr
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class BinariesConfig:
    layered_flow: str = "layered-flow"
    vvad_daily: str = "vvad_daily"
    three_d_winds: str = "3dwinds"
    dvad_2radars_daily: str = "dvad_2radars_daily"
    env: Dict[str, str] = field(default_factory=dict)

    def get_env(self) -> Dict[str, str]:
        """Build environment dict for subprocess calls, merging with current env."""
        env = os.environ.copy()
        # Disable HDF5 file locking — required for parallel processing where
        # multiple workers may read the same pvol file simultaneously.
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        for k, v in self.env.items():
            # Allow ${VAR} references in values
            env[k] = os.path.expandvars(v)
        return env


@dataclass
class OpticalFlowParams:
    alpha: int = 80
    gamma: float = 7.0
    scales: int = 100
    zfactor: float = 0.5
    tol: float = 0.005
    initer: int = 3
    outiter: int = 12


@dataclass
class ProjectionConfig:
    proj: str = "aea"
    lat_1: float = -18.0
    lat_2: float = -36.0
    ellps: str = "GRS80"

    def proj4_string(self, lon_0: float, lat_0: float) -> str:
        return (
            f"+proj={self.proj} +lon_0={lon_0} +lat_0={lat_0} "
            f"+lat_1={self.lat_1} +lat_2={self.lat_2} +units=m +ellps={self.ellps}"
        )


@dataclass
class LayeredFlowConfig:
    size: List[int] = field(default_factory=lambda: [301, 301])
    left_top: List[int] = field(default_factory=lambda: [-150500, 150500])
    cell_delta: List[int] = field(default_factory=lambda: [1000, -1000])
    units: str = "m"
    altitude_base: float = 0.0
    altitude_step: float = 500.0
    layer_count: int = 13
    moment: str = "DBZH"
    output_cappis: bool = True
    output_polar: bool = True
    max_alt_dist: int = 20000
    idw_pwr: float = 2.0
    min_dbz: int = 20
    origin: str = "xy"
    speckle_min_neighbours: int = 3
    speckle_iterations: int = 3
    # If set, force this velocity field name instead of auto-detecting from
    # the ODIM file.  Leave as empty string "" to auto-detect (default).
    velocity: str = ""
    optical_flow: OpticalFlowParams = field(default_factory=OpticalFlowParams)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    def generate_config_text(
        self,
        radar_lon: float,
        radar_lat: float,
        velocity_field: str,
        topography_path: str,
        calib_offset: float = 0.0,
    ) -> str:
        """Generate the full layered-flow config file content."""
        proj4 = self.projection.proj4_string(radar_lon, radar_lat)
        of = self.optical_flow

        return f"""# domain projection
proj4 "{proj4}"

# calibration offset
calib_offset {calib_offset}

# Velocity field name
velocity {velocity_field}

# Land/sea mask file
topography "{topography_path}"

# grid size
size "{self.size[0]} {self.size[1]}"

# top left coordinates
left_top "{self.left_top[0]} {self.left_top[1]}"

# grid resolution
cell_delta "{self.cell_delta[0]} {self.cell_delta[1]}"

# horizontal grid units
units {self.units}

# altitude of lowest layer (m)
altitude_base {self.altitude_base}

# altitude step between layers (m)
altitude_step {self.altitude_step}

# number of layers
layer_count {self.layer_count}

# radar moment to generate CAPPIs from
moment {self.moment}

# whether to output the cappis as well as flow fields
output_cappis {"true" if self.output_cappis else "false"}

# whether to output the flow magnitude and angle fields
output_polar {"true" if self.output_polar else "false"}

# maximum distance from CAPPI altitude to use reflectivities
max_alt_dist {self.max_alt_dist}

# exponent for inverse distance weighting
idw_pwr {self.idw_pwr}

# threshold out cappis to this minimum DBZ before tracking
min_dbz {self.min_dbz}

# Matrix orientation
origin {self.origin}

# speckle filter
speckle_min_neighbours {self.speckle_min_neighbours}
speckle_iterations {self.speckle_iterations}

# parameters for optical flow algorithm
optical_flow
{{
  alpha {of.alpha}
  gamma {of.gamma}
  scales {of.scales}
  zfactor {of.zfactor}
  tol {of.tol}
  initer {of.initer}
  outiter {of.outiter}
}}
"""


@dataclass
class ThreeDWindsConfig:
    dxy: float = 1500.0
    nz: int = 13
    diz: float = 500.0
    dtemp: float = 300.0
    maxrange: float = 150.0
    nx_single: int = 201
    ny_single: int = 201
    nit: int = 100
    gk_func: List[float] = field(
        default_factory=lambda: [
            4.8,
            1.6,
            4.4,
            4.2,
            4.0,
            3.85,
            3.7,
            3.55,
            3.4,
            3.2,
            3.0,
            2.8,
            2.6,
            2.45,
            2.3,
            2.15,
            2.0,
        ]
    )
    overrides: Dict[str, Any] = field(default_factory=dict)

    def generate_r3d_main_init(self, fname: str, nx: int, ny: int, ivad: int = 0) -> None:
        """Generate the r3d_main.init config file for the 3D winds Fortran code."""
        nz = self.nz
        dxy = self.dxy

        configuration = {
            "section0": {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "nt": 1,
                "mag": 0,
                "magzb": 0,
                "magzt": 0,
                "nfa": 0,
                "nmf": 0,
                "idpol": 0,
                "ihdf": 0,
                "icfrad": 1,
                "ivad": ivad,
                "igrid": 1,
                "ioptflow": 1,
            },
            "section1": {
                "dtemp(s)": self.dtemp,
                "diz(m)": self.diz,
                "dixy(m)": dxy,
                "u0": 0.0,
                "v0": 0.0,
            },
            "section2": {"maxrange": self.maxrange, "notused": 0.0, "rz1": 0.0, "nxout": 6, "nyout": 9},
            "section3": {"notused": 0.0, "dummyr": 0.0},
            "section4": {"ntec": 1, "nit": self.nit, "initype": 0, "mreso": 0},
            "section5": {
                "nx1": nx,
                "ny1": ny,
                "nzr1": nz,
                "itr1": 100,
                "nmo1": 7,
                "por1": 0.0,
                "pov1": 5.0,
                "aph1": 1.0,
            },
            "section6": {
                "nx2": nx,
                "ny2": ny,
                "nzr2": nz,
                "itr2": 100,
                "nmo2": 11,
                "por2": 0.0,
                "pov2": 5.0,
                "aph2": 1.0,
            },
            "section7": {
                "nx3": nx,
                "ny3": ny,
                "nzr3": nz,
                "itr3": 100,
                "nmo3": 10,
                "por3": 0,
                "pov3": 5.0,
                "aph3": 1.0,
            },
            "section8": {
                "nx4": 75,
                "ny4": 75,
                "nzr4": 75,
                "itr4": 200,
                "nmo4": 10,
                "por4": 0.0,
                "pov4": 5.0,
                "aph4": 1.0,
            },
            "section9": {"aq": 3.8, "bq": 0.57, "av": -2.7, "bv": 0.107, "efic": 0.0},
            "section10": {"rinf2": 0.5, "nrr": 1},
        }

        # Apply user overrides
        for key, value in self.overrides.items():
            for section in configuration.values():
                if key in section:
                    section[key] = value

        # Write the Fortran config
        content = ""
        for section in configuration.values():
            keys = list(section.keys())
            values = list(section.values())
            content += "-" * 33 + " MAIN " + "-" * 33 + "\n"
            content += "\t".join(keys) + "\n"
            content += "\t".join(str(v) for v in values) + "\n"
        content += 76 * "-"
        content += "\nGenerating function G(k)\n"
        for g in self.gk_func:
            content += f"{g}e-4\n"
        content += "2.0e-4\n" * nz
        content += 76 * "-" + "\n"
        content += 76 * "*" + "\n"
        content += "Australian Bureau of Meteorology radar\n"

        with open(fname, "w") as fid:
            fid.write(content)

    @staticmethod
    def generate_r3dbrc(
        fname: str,
        config_file: str,
        dvad_path: str,
        flow_base_dir: str,
        infile_list: List[str],
        output_files: Union[List[str], str],
        centre_lon: float,
        centre_lat: float,
    ) -> None:
        """Generate the .r3dbrc configuration file for 3D Winds."""
        content = f"{centre_lon:0.2f}\n{centre_lat:0.2f}\n"
        content += config_file + "\n"
        content += dvad_path + "/\n"
        content += flow_base_dir + "\n"
        content += "\n".join(infile_list) + "\n"
        if isinstance(output_files, list):
            content += "\n".join(output_files) + "\n"
        else:
            content += output_files + "\n"
        with open(fname, "w") as fid:
            fid.write(content)


@dataclass
class ProcessingConfig:
    lag_minutes: int = 5
    lag_tolerance_minutes: float = 2.5
    overwrite: bool = False
    cleanup_scratch: bool = True
    ncpus: int = 1


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = ""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class SwirlHPCConfig:
    """Complete configuration for a SWIRL HPC run."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    binaries: BinariesConfig = field(default_factory=BinariesConfig)
    layered_flow: LayeredFlowConfig = field(default_factory=LayeredFlowConfig)
    three_d_winds: ThreeDWindsConfig = field(default_factory=ThreeDWindsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def setup_logging(self) -> None:
        """Configure the Python logging module from this config."""
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        handlers: list = [logging.StreamHandler()]
        if self.logging.file:
            logpath = Path(os.path.expandvars(self.logging.file))
            logpath.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(logpath)))

        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,
        )

    def ensure_dirs(self) -> None:
        """Create all required output directories."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.effective_scratch_dir.mkdir(parents=True, exist_ok=True)
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(d: dict) -> SwirlHPCConfig:
    """Convert a flat dict (from TOML) into the nested dataclass structure."""
    paths_d = d.get("paths", {})
    bins_d = d.get("binaries", {})
    lf_d = d.get("layered_flow", {})
    w3d_d = d.get("three_d_winds", {})
    proc_d = d.get("processing", {})
    log_d = d.get("logging", {})

    # Nested sub-configs for layered_flow
    of_d = lf_d.pop("optical_flow", {})
    proj_d = lf_d.pop("projection", {})

    # Nested sub-config for binaries
    env_d = bins_d.pop("env", {})

    # Build path objects
    paths_kw = {}
    for k, v in paths_d.items():
        if k in ("output_dir", "scratch_dir", "topography"):
            paths_kw[k] = Path(os.path.expandvars(str(v))) if v else None
        else:
            paths_kw[k] = v

    return SwirlHPCConfig(
        paths=PathsConfig(**paths_kw),
        binaries=BinariesConfig(**bins_d, env=env_d),
        layered_flow=LayeredFlowConfig(
            **lf_d,
            optical_flow=OpticalFlowParams(**of_d),
            projection=ProjectionConfig(**proj_d),
        ),
        three_d_winds=ThreeDWindsConfig(**w3d_d),
        processing=ProcessingConfig(**proc_d),
        logging=LoggingConfig(**log_d),
    )


def _load_toml(path: Union[str, Path]) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(user_config: Union[str, Path, None] = None) -> SwirlHPCConfig:
    """
    Load configuration from the bundled defaults and optionally merge with
    a user-provided TOML file.

    Parameters
    ----------
    user_config : str, Path, or None
        Path to a user TOML config file. If None, only defaults are used.

    Returns
    -------
    SwirlHPCConfig
        The merged configuration.
    """
    # Load bundled defaults
    defaults_path = Path(__file__).parent.parent.parent / "swirl_defaults.toml"
    if defaults_path.exists():
        defaults = _load_toml(defaults_path)
    else:
        defaults = {}

    # Merge user overrides
    if user_config is not None:
        user = _load_toml(user_config)
        merged = _deep_merge(defaults, user)
    else:
        merged = defaults

    if not merged:
        return SwirlHPCConfig()

    return _dict_to_config(merged)
