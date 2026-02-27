"""
ODIM HDF5 file metadata extraction.

Replaces ODIMFileInformation from the operational code, adapted for use
with files extracted from the AURA archive.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple

import h5py

logger = logging.getLogger(__name__)


class DopplerNotFoundError(Exception):
    """Raised when no Doppler velocity field is found in the radar file."""
    pass


@dataclass
class ODIMMetadata:
    """Metadata extracted from an ODIM HDF5 radar volume file."""
    filename: str
    rid: int
    lat: float
    lon: float
    datetime: datetime
    moments: List[str]

    @classmethod
    def from_file(cls, filepath: str) -> "ODIMMetadata":
        """
        Extract metadata from an ODIM HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the .pvol.h5 file on disk.

        Returns
        -------
        ODIMMetadata
        """
        filepath = str(filepath)
        if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
            raise FileNotFoundError(f"File '{filepath}' does not exist or is empty.")

        # Extract RID from filename (e.g. "2_20251016_123456.pvol.h5" -> 2)
        basename = os.path.basename(filepath)
        rid = int(basename.split("_")[0])

        moments = []
        with h5py.File(filepath, "r") as odim:
            lat = float(odim["where"].attrs["lat"])
            lon = float(odim["where"].attrs["lon"])

            date_str = odim["what"].attrs["date"].decode()
            time_str = odim["what"].attrs["time"].decode()
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")

            dataset1 = odim["dataset1"]
            for dt_idx in itertools.count(1):
                data_key = f"data{dt_idx}"
                if data_key not in dataset1:
                    break
                quantity = dataset1[f"data{dt_idx}/what"].attrs["quantity"].decode()
                moments.append(quantity)

        return cls(
            filename=filepath,
            rid=rid,
            lat=lat,
            lon=lon,
            datetime=dt,
            moments=moments,
        )

    def get_doppler_field(self) -> str:
        """
        Find the Doppler velocity field name in this file.

        Returns
        -------
        str
            The field name (VRADH, VRAD, or VRADDH).

        Raises
        ------
        DopplerNotFoundError
            If no Doppler velocity field is found.
        """
        for name in ("VRADH", "VRAD", "VRADDH"):
            if name in self.moments:
                return name
        raise DopplerNotFoundError(
            f"No Doppler velocity field in {self.filename}. "
            f"Available moments: {self.moments}"
        )
