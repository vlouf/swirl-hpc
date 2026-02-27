"""
swirlhpc — Run the SWIRL radar wind retrieval pipeline on NCI Gadi HPC.

Quick start:
    import swirlhpc
    from datetime import date

    # Single radar, single day
    swirlhpc.run(2, date(2025, 10, 16), config="my_config.toml")

    # Multi-Doppler region, date range
    swirlhpc.run([2, 3, 4], date(2025, 10, 1), date(2025, 10, 7))
"""

from swirlhpc.pipeline import run
from swirlhpc.config import SwirlHPCConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "run",
    "load_config",
    "SwirlHPCConfig",
]
