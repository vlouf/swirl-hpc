# swirlhpc — SWIRL on NCI Gadi

Run the SWIRL (Synthetic Wind Information from Radar and Lidar) radar wind retrieval pipeline on NCI Gadi HPC, using the [AURA](https://github.com/vlouf/aura) archive for data access.

## What it does

`swirlhpc` extracts the processing logic from the operational SWIRL real-time services and wraps it in a batch-friendly Python library. It replaces the socket-based microservice architecture with direct function calls, and replaces Rainfields data ingest with lazy access to the AURA archive.

The pipeline runs: **AURA archive → vvad_daily → layered-flow → dvad_2radars_daily → 3dwinds**

## Installation

```bash
cd swirlhpc
pip install -e .
```

### Requirements

- Python ≥ 3.9
- Access to NCI Gadi and the `rq0` project (for AURA data)
- The [aura](https://github.com/vlouf/aura) library
- SWIRL binaries: `layered-flow`, `vvad_daily`, `3dwinds`, `dvad_2radars_daily`

## Quick start

### Python API

```python
import swirlhpc
from datetime import date

# Single radar, single day
swirlhpc.run(2, date(2025, 10, 16), config="my_config.toml")

# Multi-Doppler region, date range
swirlhpc.run([2, 3, 4], date(2025, 10, 1), date(2025, 10, 7))

# Modify config on the fly
cfg = swirlhpc.load_config("my_config.toml")
cfg.processing.overwrite = True
cfg.paths.output_dir = Path("/scratch/ab12/user/experiment_2")
swirlhpc.run(2, date(2025, 10, 16), config=cfg)
```

### Command line

```bash
# Single radar
swirl-run --radars 2 --start 2025-10-16 --config my_config.toml

# Multi-Doppler, date range
swirl-run --radars 2 3 4 --start 2025-10-01 --end 2025-10-07 --config my_config.toml
```

## Configuration

Copy `swirl_defaults.toml` and edit it for your environment. The config file controls everything: binary paths, environment variables, all algorithm parameters, and output locations.

```toml
[paths]
output_dir = "/scratch/ab12/user/swirl"
topography = "/g/data/rq0/admin/topography/australian_topography_1000m.nc"

[binaries]
layered_flow = "/opt/swirl/bin/layered-flow"
vvad_daily = "/opt/swirl/bin/vvad_daily"
three_d_winds = "/opt/swirl/bin/3dwinds"
dvad_2radars_daily = "/opt/swirl/bin/dvad_2radars_daily"

[binaries.env]
LD_LIBRARY_PATH = "/opt/swirl/lib:/opt/eccodes/lib"

[layered_flow]
min_dbz = 15              # override a single parameter

[three_d_winds.overrides]
pov1 = 10.0               # override Fortran config values
```

Any parameter left unset uses the built-in defaults (which match the operational SWIRL configuration).

## Output structure

```
{output_dir}/
├── flow/{rid}/{YYYYMMDD}/       # layered-flow NetCDF output
│   └── {rid}_{YYYYMMDD_HHMM00}_flow.nc
├── vvad/{rid}/{YYYYMMDD}/       # VAD .dat and .json files
│   ├── {rid}_{YYYYMMDD_HHMM00}_vvad.dat
│   └── IDR{rid}VAD1_{YYYYMMDD_HHMM00}.json
├── winds/{region}/{YYYYMMDD}/   # 3D winds NetCDF output
│   ├── {region}_{YYYYMMDD_HHMM}.nc
│   └── {region}_{YYYYMMDD_HHMM}_lwstwind.nc
├── dvad/{region}/{YYYYMMDD}/    # DVAD output (multi-Doppler only)
├── config/                       # Generated config files
└── scratch/                      # Temporary extracted pvol files
```

For multi-Doppler regions, `{region}` is the sorted radar IDs joined by underscores (e.g. `2_3_4`).

## How it works

1. **Data access**: Uses `aura.get_vol()` to lazily list all radar volumes for a day from the NCI AURA archive (zip files, no extraction yet).

2. **Lag pairing**: For each volume at time *t*, finds the nearest volume at *t − 5 min* (configurable) within a tolerance window.

3. **Extraction**: Extracts only the needed pair of pvol.h5 files to a scratch directory using `LazyVolume.extract_to()`.

4. **VAD**: Runs `vvad_daily` on the current pvol file to produce a wind profile, then converts the output to JSON.

5. **Optical flow**: Generates a per-radar layered-flow config file and runs `layered-flow` on the (lag, current) pair to produce gridded flow fields.

6. **3D winds (single-radar)**: For each radar, runs `3dwinds` with ivad=1 (VAD constraint) using the flow output.

7. **3D winds (multi-Doppler)**: If multiple radar IDs are provided, runs `dvad_2radars_daily` for all radar pairs, then `3dwinds` with the combined inputs on a domain sized to cover all radars.

## Differences from operational SWIRL

| Aspect | Operational | swirlhpc |
|---|---|---|
| Data source | Rainfields download | AURA archive (NCI) |
| Architecture | Async socket services | Direct function calls |
| Lag file discovery | Directory scanning | `VolumeList.nearest()` |
| Timing guards | Real-time checks | Removed (batch mode) |
| Region definitions | `regions.json` + swirlconf | User passes radar ID list |
| Configuration | swirlconf Python module | TOML file |
| File cleanup | Deletes processed inputs | Optional scratch cleanup |
| CMSS/egress links | Hard-links for pickup | Not applicable |

## License

Apache License 2.0 — same as the operational SWIRL code.
