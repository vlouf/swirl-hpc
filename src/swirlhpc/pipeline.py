"""
SWIRL HPC Pipeline Orchestrator.

Provides the high-level `run()` function that processes radar data
end-to-end: extract from AURA → VAD → optical flow → 3D winds.

Both the flow and winds phases are parallelised using a process pool
whose size is controlled by ``processing.ncpus`` in the config.
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

import aura
from aura.volume import LazyVolume, VolumeList

from swirlhpc.config import SwirlHPCConfig, load_config
from swirlhpc.flow import FlowResult, process_flow_timestep
from swirlhpc.odim import DopplerNotFoundError
from swirlhpc.winds import (
    WindsResult,
    process_multidoppler_winds,
    process_single_radar_winds,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(ts: str) -> pd.Timestamp:
    """Parse a SWIRL timestamp string (YYYYMMDD_HHMMSS) to pd.Timestamp."""
    return pd.Timestamp(datetime.strptime(ts, "%Y%m%d_%H%M%S"))


# =========================================================================
# Top-level worker functions  (must be picklable for multiprocessing)
# =========================================================================

def _flow_worker(
    config: SwirlHPCConfig,
    current_path: str,
    lag_path: str,
) -> Optional[FlowResult]:
    """
    Worker function: run VAD + optical flow for one timestep.

    Returns FlowResult on success, None on expected skip (e.g. no Doppler).
    Raises on unexpected errors so the caller can count them.
    """
    try:
        return process_flow_timestep(config, current_path, lag_path)
    except DopplerNotFoundError:
        logging.getLogger(__name__).info(
            "No Doppler field in %s, skipping", os.path.basename(current_path),
        )
        return None
    except FileNotFoundError as err:
        logging.getLogger(__name__).warning("Flow failed for %s: %s", os.path.basename(current_path), err)
        return None


def _single_winds_worker(
    config: SwirlHPCConfig,
    rid: int,
    timestamp_str: str,
    flow_files: List[str],
) -> Optional[WindsResult]:
    """Worker function: run single-radar 3D winds for one timestep."""
    try:
        radar_dtime = _parse_timestamp(timestamp_str)
        return process_single_radar_winds(config, rid, radar_dtime, flow_files)
    except Exception as err:
        logging.getLogger(__name__).error(
            "Single-radar winds failed for %s at %s: %r", rid, timestamp_str, err,
        )
        return None


def _multidoppler_winds_worker(
    config: SwirlHPCConfig,
    rids: List[int],
    timestamp_str: str,
    flow_files_by_rid: Dict[int, List[str]],
    vvad_files_by_rid: Dict[int, str],
) -> Optional[WindsResult]:
    """Worker function: run multi-Doppler 3D winds for one timestep."""
    try:
        radar_dtime = _parse_timestamp(timestamp_str)
        return process_multidoppler_winds(
            config, rids, radar_dtime, flow_files_by_rid, vvad_files_by_rid,
        )
    except Exception as err:
        logging.getLogger(__name__).error(
            "Multi-Doppler winds failed at %s: %r", timestamp_str, err,
        )
        return None


# =========================================================================
# Volume extraction and lag pairing
# =========================================================================

def _extract_volume(vol: LazyVolume, scratch_dir: Path) -> str:
    """Extract a LazyVolume to the scratch directory, returning the path."""
    outpath = scratch_dir / vol.filename
    if outpath.exists():
        return str(outpath)
    vol.extract_to(scratch_dir)
    return str(outpath)


def _get_volumes_with_previous_day(
    rid: int,
    day: date,
) -> Tuple[Optional[VolumeList], List[LazyVolume]]:
    """
    Load volumes for a given day, and also fetch the tail end of the
    previous day so that the first volume of `day` can find its lag
    across the day boundary.

    Returns
    -------
    today_volumes : VolumeList or None
        The day's VolumeList (None if no data).
    all_volumes : list of LazyVolume
        Ordered list of volumes including previous-day tail + today,
        suitable for predecessor lookups.
    """
    try:
        today_volumes = aura.get_vol(rid, day)
    except FileNotFoundError:
        return None, []

    all_vols = list(today_volumes)

    # Try to get previous day's volumes for cross-day lag
    prev_day = day - timedelta(days=1)
    try:
        prev_volumes = aura.get_vol(rid, prev_day)
        # Prepend previous day's volumes so predecessor lookup works
        all_vols = list(prev_volumes) + all_vols
    except FileNotFoundError:
        logger.debug("No previous-day data for radar %s on %s", rid, prev_day.isoformat())

    return today_volumes, all_vols


# =========================================================================
# Prepare tasks: extract volumes and build (current, lag) pairs
# =========================================================================

def _prepare_flow_tasks(
    config: SwirlHPCConfig,
    rid: int,
    all_volumes: List[LazyVolume],
    target_volumes: Optional[List[LazyVolume]] = None,
) -> List[Tuple[str, str]]:
    """
    Extract all required volumes to scratch and return a list of
    (current_path, lag_path) pairs ready for flow processing.

    For each target volume, the lag is the immediately preceding volume
    in ``all_volumes`` (which may include the previous day's tail).
    A maximum gap tolerance rejects lags that are too far in the past
    (e.g. after hours-long outages).

    Extraction is sequential (I/O-bound zip reads) so that the parallel
    workers only need to do CPU-bound computation.
    """
    if target_volumes is None:
        target_volumes = list(all_volumes)

    # Build an ordered index of all volumes for fast predecessor lookup
    all_vols = list(all_volumes)
    vol_index = {id(v): i for i, v in enumerate(all_vols)}

    max_gap_seconds = config.processing.lag_minutes * 60 + config.processing.lag_tolerance_minutes * 60

    scratch_dir = config.paths.effective_scratch_dir / str(rid)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[str, str]] = []
    for current_vol in target_volumes:
        # Find this volume's position in the full list
        idx = vol_index.get(id(current_vol))
        if idx is None:
            # target_volumes entry not in all_vols — find nearest by timestamp
            for i, v in enumerate(all_vols):
                if v.timestamp == current_vol.timestamp:
                    idx = i
                    break
        if idx is None or idx == 0:
            logger.warning(
                "No preceding volume for %s, skipping",
                current_vol.timestamp.isoformat(),
            )
            continue

        lag_vol = all_vols[idx - 1]
        gap_seconds = (current_vol.timestamp - lag_vol.timestamp).total_seconds()

        if gap_seconds > max_gap_seconds:
            logger.warning(
                "Gap too large before %s: %.0fs (max %.0fs), skipping",
                current_vol.timestamp.isoformat(), gap_seconds, max_gap_seconds,
            )
            continue

        if gap_seconds <= 0:
            logger.warning(
                "Non-positive gap before %s (lag=%s), skipping",
                current_vol.timestamp.isoformat(), lag_vol.timestamp.isoformat(),
            )
            continue

        try:
            current_path = _extract_volume(current_vol, scratch_dir)
            lag_path = _extract_volume(lag_vol, scratch_dir)
            pairs.append((current_path, lag_path))
        except Exception as err:
            logger.error("Failed to extract volumes for %s: %r", current_vol.timestamp.isoformat(), err)

    return pairs


# =========================================================================
# Phase 1: parallel flow processing
# =========================================================================

def _run_flow_parallel(
    config: SwirlHPCConfig,
    tasks: List[Tuple[str, str]],
    ncpus: int,
) -> List[FlowResult]:
    """Submit all flow tasks to a process pool and collect results."""
    if not tasks:
        return []

    results: List[FlowResult] = []
    n_total = len(tasks)

    if ncpus <= 1:
        for i, (cur, lag) in enumerate(tasks, 1):
            fr = _flow_worker(config, cur, lag)
            if fr is not None:
                results.append(fr)
                logger.info("[%d/%d] Flow OK: %s", i, n_total, os.path.basename(cur))
        return results

    logger.info("Submitting %d flow tasks to %d workers", n_total, ncpus)
    with ProcessPoolExecutor(max_workers=ncpus) as pool:
        future_to_idx = {
            pool.submit(_flow_worker, config, cur, lag): i
            for i, (cur, lag) in enumerate(tasks)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            cur_path = tasks[idx][0]
            try:
                fr = future.result()
                if fr is not None:
                    results.append(fr)
                    logger.info(
                        "[%d/%d] Flow OK: %s",
                        len(results), n_total, os.path.basename(cur_path),
                    )
            except Exception as err:
                logger.error("Flow worker crashed for %s: %r", os.path.basename(cur_path), err)

    logger.info("Flow phase complete: %d/%d succeeded", len(results), n_total)
    return results


# =========================================================================
# Phase 2: parallel winds processing
# =========================================================================

def _run_winds_parallel(
    config: SwirlHPCConfig,
    rids: List[int],
    day: date,
    flow_results_by_rid: Dict[int, List[FlowResult]],
    ncpus: int,
) -> List[WindsResult]:
    """Submit all winds tasks (single-radar + multi-Doppler) to a process pool."""
    wind_results: List[WindsResult] = []

    # Build lookup
    flow_lookup: Dict[Tuple[int, str], FlowResult] = {}
    for rid, frs in flow_results_by_rid.items():
        for fr in frs:
            flow_lookup[(fr.rid, fr.timestamp)] = fr

    # --- Collect single-radar tasks ---
    single_tasks: List[Tuple[int, str, List[str]]] = []
    for rid in rids:
        for fr in flow_results_by_rid.get(rid, []):
            single_tasks.append((rid, fr.timestamp, [fr.flow_file, fr.flow_file]))

    # --- Collect multi-Doppler tasks ---
    multi_tasks: List[Tuple[str, Dict[int, List[str]], Dict[int, str]]] = []
    if len(rids) > 1:
        timestamps_per_rid = {
            rid: {fr.timestamp for fr in frs}
            for rid, frs in flow_results_by_rid.items()
        }

        # Log per-radar flow counts for debugging multi-Doppler issues
        for rid in rids:
            n = len(timestamps_per_rid.get(rid, set()))
            logger.info("  Radar %s: %d flow timesteps available for multi-Doppler", rid, n)

        # Only intersect radars that actually have data
        non_empty = [ts for rid, ts in timestamps_per_rid.items() if ts]
        if len(non_empty) == len(rids):
            common_timestamps = sorted(set.intersection(*non_empty))
            logger.info("  Common timesteps across all %d radars: %d", len(rids), len(common_timestamps))
        elif len(non_empty) >= 2:
            # Some radars missing — intersect what we have, log warning
            available_rids = [rid for rid, ts in timestamps_per_rid.items() if ts]
            common_timestamps = sorted(set.intersection(*non_empty))
            logger.warning(
                "  Only %d/%d radars have data (radars %s). "
                "Multi-Doppler will use available radars (%d common timesteps).",
                len(non_empty), len(rids), available_rids, len(common_timestamps),
            )
            # Update rids for multi-Doppler to only include radars with data
            rids_for_multi = available_rids
        else:
            common_timestamps = []
            logger.warning("  Fewer than 2 radars have flow data — skipping multi-Doppler.")

        if 'rids_for_multi' not in locals():
            rids_for_multi = list(rids)

        for ts in common_timestamps:
            flow_files_by_rid: Dict[int, List[str]] = {}
            vvad_files_by_rid: Dict[int, str] = {}
            skip = False
            for rid in rids_for_multi:
                fr = flow_lookup.get((rid, ts))
                if fr is None:
                    skip = True
                    break
                vvad_files_by_rid[rid] = fr.vvad_file
                flow_files_by_rid[rid] = [fr.flow_file, fr.flow_file]
            if not skip:
                multi_tasks.append((ts, rids_for_multi, flow_files_by_rid, vvad_files_by_rid))

    n_total = len(single_tasks) + len(multi_tasks)
    if n_total == 0:
        return []

    # --- Sequential fallback ---
    if ncpus <= 1:
        for rid, ts, ffiles in single_tasks:
            wr = _single_winds_worker(config, rid, ts, ffiles)
            if wr is not None:
                wind_results.append(wr)
        for ts, multi_rids, ff_by_rid, vv_by_rid in multi_tasks:
            wr = _multidoppler_winds_worker(config, multi_rids, ts, ff_by_rid, vv_by_rid)
            if wr is not None:
                wind_results.append(wr)
        return wind_results

    # --- Parallel ---
    logger.info(
        "Submitting %d winds tasks (%d single + %d multi-Doppler) to %d workers",
        n_total, len(single_tasks), len(multi_tasks), ncpus,
    )

    with ProcessPoolExecutor(max_workers=ncpus) as pool:
        futures: Dict = {}

        for rid, ts, ffiles in single_tasks:
            fut = pool.submit(_single_winds_worker, config, rid, ts, ffiles)
            futures[fut] = f"single-{rid}-{ts}"

        for ts, multi_rids, ff_by_rid, vv_by_rid in multi_tasks:
            fut = pool.submit(
                _multidoppler_winds_worker, config, multi_rids, ts, ff_by_rid, vv_by_rid,
            )
            futures[fut] = f"multi-{ts}"

        for future in as_completed(futures):
            label = futures[future]
            try:
                wr = future.result()
                if wr is not None:
                    wind_results.append(wr)
                    logger.info("[%d/%d] Winds OK: %s", len(wind_results), n_total, label)
            except Exception as err:
                logger.error("Winds worker crashed for %s: %r", label, err)

    logger.info("Winds phase complete: %d/%d succeeded", len(wind_results), n_total)
    return wind_results


# =========================================================================
# Day-level orchestration
# =========================================================================

def _process_radar_day(
    config: SwirlHPCConfig,
    rid: int,
    day: date,
) -> List[FlowResult]:
    """Extract, pair, and run flow for all timesteps of one radar on one day."""
    logger.info("=" * 60)
    logger.info("Processing radar %s for %s", rid, day.isoformat())
    logger.info("=" * 60)

    today_volumes, all_vols = _get_volumes_with_previous_day(rid, day)
    if today_volumes is None or len(today_volumes) == 0:
        logger.warning("No data for radar %s on %s", rid, day.isoformat())
        return []

    # Only process today's volumes, but pass full list for lag lookup
    target_volumes = [v for v in all_vols if v.timestamp.date() == day]
    logger.info(
        "Found %d volumes for radar %s on %s (%d total incl. prev-day tail)",
        len(target_volumes), rid, day.isoformat(), len(all_vols),
    )

    # Prepare: extract volumes and build (current, lag) pairs
    # Pass full volume list (incl. prev-day) so first-of-day can find its lag
    tasks = _prepare_flow_tasks(config, rid, all_vols, target_volumes=target_volumes)
    logger.info("Prepared %d flow tasks for radar %s", len(tasks), rid)

    # Run flow in parallel
    results = _run_flow_parallel(config, tasks, config.processing.ncpus)

    # Cleanup scratch
    if config.processing.cleanup_scratch:
        scratch_dir = config.paths.effective_scratch_dir / str(rid)
        logger.info("Cleaning up scratch for radar %s", rid)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return results


def _process_radar_timestamp(
    config: SwirlHPCConfig,
    rid: int,
    target_time: datetime,
) -> List[FlowResult]:
    """Extract, pair, and run flow for a single timestep."""
    logger.info("=" * 60)
    logger.info("Processing radar %s at %s", rid, target_time.isoformat())
    logger.info("=" * 60)

    day = target_time.date()
    today_volumes, all_vols = _get_volumes_with_previous_day(rid, day)
    if today_volumes is None or len(today_volumes) == 0:
        logger.warning("No data for radar %s on %s", rid, day.isoformat())
        return []

    current_vol = today_volumes.nearest(target_time)
    logger.info(
        "Nearest volume to %s is %s",
        target_time.strftime("%H:%M:%S"),
        current_vol.timestamp.strftime("%H:%M:%S"),
    )

    tasks = _prepare_flow_tasks(config, rid, all_vols, target_volumes=[current_vol])
    # Single timestep — run directly, no pool overhead
    results = _run_flow_parallel(config, tasks, ncpus=1)

    if config.processing.cleanup_scratch:
        scratch_dir = config.paths.effective_scratch_dir / str(rid)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return results


# =========================================================================
# Public API
# =========================================================================

def run(
    radar_ids: Union[int, List[int]],
    start_date: Union[date, datetime],
    end_date: Optional[Union[date, datetime]] = None,
    config: Union[str, Path, SwirlHPCConfig, None] = None,
) -> Dict:
    """
    Run the full SWIRL processing pipeline.

    This is the main entry point for the library. It processes radar data
    through the complete pipeline:

        AURA archive → extract → vvad_daily → layered-flow → 3dwinds

    Both the flow and winds phases are parallelised using a process pool
    whose size is controlled by ``processing.ncpus`` in the config.

    If multiple radar IDs are provided, multi-Doppler wind retrieval is
    performed automatically on timesteps where all radars have data,
    in addition to single-radar retrieval for each radar.

    Parameters
    ----------
    radar_ids : int or list of int
        Radar ID(s) to process. If a list with >1 element, multi-Doppler
        retrieval is performed on that group.
    start_date : date or datetime
        - If a ``date``: process all timesteps on that day (or date range
          if ``end_date`` is also given).
        - If a ``datetime``: process only the single timestep nearest to
          that time. ``end_date`` is ignored in this case.
    end_date : date or datetime, optional
        End date (inclusive) for date-range mode. Ignored when
        ``start_date`` is a ``datetime``.
    config : str, Path, SwirlHPCConfig, or None
        Configuration. Can be:
        - A path to a TOML config file
        - A pre-loaded SwirlHPCConfig object
        - None to use defaults

    Returns
    -------
    dict
        Summary of processing results with keys:
        - "flow_results": dict of {rid: [FlowResult, ...]}
        - "wind_results": [WindsResult, ...]
        - "days_processed": int
        - "timesteps_processed": int
        - "errors": int

    Examples
    --------
    >>> import swirlhpc
    >>> from datetime import date, datetime
    >>>
    >>> # Single radar, single timestep
    >>> swirlhpc.run(2, datetime(2026, 1, 1, 12, 30))
    >>>
    >>> # Single radar, full day, parallelised (set ncpus in config)
    >>> swirlhpc.run(2, date(2025, 10, 16), config="my_config.toml")
    >>>
    >>> # Multi-Doppler region, date range
    >>> swirlhpc.run([2, 3, 4], date(2025, 10, 1), date(2025, 10, 7))
    """
    # Normalise inputs
    if isinstance(radar_ids, int):
        radar_ids = [radar_ids]

    # Detect mode: single-timestep (datetime) vs date-range (date)
    single_timestamp: Optional[datetime] = None
    if isinstance(start_date, datetime) and type(start_date) is not date:
        single_timestamp = start_date

    # Load config
    if isinstance(config, SwirlHPCConfig):
        cfg = config
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = load_config()

    cfg.setup_logging()
    cfg.ensure_dirs()

    ncpus = cfg.processing.ncpus

    logger.info("=" * 70)
    logger.info("SWIRL HPC Pipeline")
    logger.info("  Radars:     %s", radar_ids)
    if single_timestamp is not None:
        logger.info("  Timestamp:  %s", single_timestamp.isoformat())
        logger.info("  Mode:       single timestep")
    else:
        _start = start_date if isinstance(start_date, date) else start_date.date()
        _end = end_date.date() if isinstance(end_date, datetime) else (end_date or _start)
        logger.info("  Date range: %s to %s", _start, _end)
    if len(radar_ids) > 1:
        logger.info("  Retrieval:  Multi-Doppler + single-radar")
    else:
        logger.info("  Retrieval:  Single-radar")
    logger.info("  Workers:    %d", ncpus)
    logger.info("  Output:     %s", cfg.paths.output_dir)
    logger.info("=" * 70)

    all_flow_results: Dict[int, List[FlowResult]] = {rid: [] for rid in radar_ids}
    all_wind_results: List[WindsResult] = []
    total_timesteps = 0
    total_errors = 0
    days_processed = 0

    # ----- Single timestamp mode -----
    if single_timestamp is not None:
        day = single_timestamp.date()
        day_flow_results: Dict[int, List[FlowResult]] = {}

        for rid in radar_ids:
            results = _process_radar_timestamp(cfg, rid, single_timestamp)
            day_flow_results[rid] = results
            all_flow_results[rid].extend(results)
            total_timesteps += len(results)

        try:
            wind_results = _run_winds_parallel(cfg, radar_ids, day, day_flow_results, ncpus=1)
            all_wind_results.extend(wind_results)
        except Exception as err:
            logger.error("Winds processing failed for %s: %r", single_timestamp, err)
            total_errors += 1

        days_processed = 1

    # ----- Date range mode -----
    else:
        _start = start_date if isinstance(start_date, date) else start_date.date()
        _end = end_date.date() if isinstance(end_date, datetime) else (end_date or _start)

        current_day = _start
        while current_day <= _end:
            day_flow_results = {}

            # Phase 1: flow (parallel)
            for rid in radar_ids:
                results = _process_radar_day(cfg, rid, current_day)
                day_flow_results[rid] = results
                all_flow_results[rid].extend(results)
                total_timesteps += len(results)

            # Phase 2: winds (parallel)
            try:
                wind_results = _run_winds_parallel(
                    cfg, radar_ids, current_day, day_flow_results, ncpus,
                )
                all_wind_results.extend(wind_results)
            except Exception as err:
                logger.error("Winds processing failed for %s: %r", current_day, err)
                total_errors += 1

            days_processed += 1
            current_day += timedelta(days=1)

    # Summary
    n_wind_ok = sum(1 for wr in all_wind_results if os.path.isfile(wr.output_2d))
    logger.info("=" * 70)
    logger.info("Pipeline complete")
    logger.info("  Days processed:      %d", days_processed)
    logger.info("  Flow timesteps:      %d", total_timesteps)
    logger.info("  Wind retrievals:     %d (%d successful)", len(all_wind_results), n_wind_ok)
    logger.info("=" * 70)

    return {
        "flow_results": all_flow_results,
        "wind_results": all_wind_results,
        "days_processed": days_processed,
        "timesteps_processed": total_timesteps,
        "errors": total_errors,
    }
