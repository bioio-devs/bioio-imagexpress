#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from bioio_base import constants, exceptions, io
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER_LIST,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES,
    DimensionNames,
)
from bioio_base.reader import Reader as BaseReader
from bioio_base.standard_metadata import StandardMetadata
from bioio_base.types import DimSpec, PhysicalPixelSizes, TimeInterval
from dask import delayed
from fsspec.spec import AbstractFileSystem

from . import acquisition
from .acquisition import SceneKey

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Reader(BaseReader):
    """
    Read Molecular Devices ImageXpress / MetaXpress acquisitions

    An acquisition is a directory rather than a single file: every plane is its
    own TIFF, and the structure tying them together lives in a ``.jdce``
    descriptor, an ``image_metadata_*.csv`` manifest, and the plane filenames.
    One scene is assembled per well, with the well's acquisition positions as
    mosaic tiles on ``M``.

    Parameters
    ----------
    image: Path or str
        Path to a ``.jdce`` descriptor or to the acquisition directory holding it.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    mosaic: bool
        Treat each well as one scene, with its acquisition positions on the ``M``
        dimension. Turn it off for one scene per (well, position) and no ``M``.
        Falls back to off, with a warning, when the tiles could not be stitched
        anyway: placing them needs the manifest's stage positions and the
        descriptor's pixel size.
        Default: True

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the path is not an ImageXpress acquisition.
    """

    NAME = "bioio-imagexpress"

    # Required Methods

    def __init__(
        self,
        image: Any,
        fs_kwargs: Dict[str, Any] = {},
        mosaic: bool = True,
        **kwargs: Any,
    ):
        self._fs, self._path = io.pathlike_to_fs(
            image, enforce_exists=True, fs_kwargs=fs_kwargs
        )

        discovered = acquisition.discover_unit(self._fs, self._path)
        if discovered is None:
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path, msg_extra=self._why_unsupported()
            )

        # Manifest parsing only; no TIFF is opened until pixels are requested.
        self._unit = acquisition.build_unit(self._fs, *discovered)
        if not self._unit.planes:
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                msg_extra="No readable planes were found in this acquisition.",
            )

        if mosaic and not self._tiles_placeable():
            log.warning(
                "Stage positions or pixel size are missing, so tiles cannot be "
                "stitched; reading one scene per acquisition position instead."
            )
            mosaic = False

        # Scene index -> (well, the acquisition positions it covers).
        self._mosaic = mosaic
        if mosaic:
            self._scene_table = [
                (well, tuple(self._unit.sites(well))) for well in self._unit.wells
            ]
        else:
            self._scene_table = [
                (well, (site,)) for well, site in self._unit.scene_keys
            ]

        # Per-scene pyramid shapes, dtype and MetaSeries tags, probed from one
        # plane on first use. Keyed by scene index because wells within one unit
        # can differ in shape.
        self._scene_levels: Dict[int, Tuple[List[Tuple[int, ...]], np.dtype]] = {}
        self._scene_metaseries: Dict[int, Optional[Dict[str, Any]]] = {}

        self._scenes: Optional[Tuple[str, ...]] = None

    def _tiles_placeable(self) -> bool:
        """Whether every multi-position well can be laid out as a mosaic."""
        unit = self._unit
        multi = [well for well in unit.wells if len(unit.sites(well)) > 1]
        if not multi:
            return True

        if not (unit.jdce.pixel_size_x and unit.jdce.pixel_size_y):
            return False

        return all(
            None not in unit.positions.get((well, site), (None, None))
            for well in multi
            for site in unit.sites(well)
        )

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        return acquisition.discover_unit(fs, path) is not None

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            One scene per well (``"B07"``), or one per well and acquisition
            position with ``mosaic=False`` (``"B07-s0"``).
        """
        if self._scenes is None:
            self._scenes = tuple(
                well if self._mosaic else f"{well}-s{sites[0]}"
                for well, sites in self._scene_table
            )

        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._build(delayed_read=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._build(delayed_read=False)

    def _read_indexed(self, given_dims: str, dim_specs: List[DimSpec]) -> np.ndarray:
        """
        Slice the delayed graph so that only the planes asked for are opened.

        Every plane is a separate file, so the base implementation -- materialize
        the scene, then slice -- opens one file per TxCxZ even for a single plane.

        Parameters
        ----------
        given_dims: str
            The native dimension ordering of the image (``self.dims.order``).
        dim_specs: List[DimSpec]
            One getitem operation per dimension in ``given_dims``.

        Returns
        -------
        data: np.ndarray
            The indexed image data in native (reduced) dimension order.
        """
        # Dask does not support fancy indexing on more than one axis at a time.
        if sum(isinstance(spec, (list, tuple)) for spec in dim_specs) > 1:
            return super()._read_indexed(given_dims, dim_specs)

        return self.dask_data[tuple(dim_specs)].compute()

    def _build(self, delayed_read: bool) -> xr.DataArray:
        well, sites = self._current()
        timepoints, channels, zs = self._unit.extents(self._current_keys())
        shape, dtype = self._plane_spec()
        level = self._level()

        planes = [
            self._plane_array(
                self._unit.planes[(well, site)].get((t, channel, z)),
                shape,
                dtype,
                level,
                delayed_read,
            )
            for site, t, channel, z in product(sites, timepoints, channels, zs)
        ]

        stacked = da.stack(planes) if delayed_read else np.stack(planes)
        data = stacked.reshape(
            (len(sites), len(timepoints), len(channels), len(zs)) + tuple(shape)
        )

        return xr.DataArray(
            data if self._mosaic else data[0],
            dims=(
                DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES
                if self._mosaic
                else DEFAULT_DIMENSION_ORDER_LIST
            ),
            coords=self._coords(zs),
            attrs={constants.METADATA_UNPROCESSED: self._unprocessed_metadata()},
        )

    # Resolution levels

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        resolution_levels: Tuple[int, ...]
            The levels of the pyramid MetaXpress writes into each plane.
        """
        return tuple(range(len(self._levels()[0])))

    # Metadata

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            The floats representing physical pixel sizes in micrometers for
            dimensions Z, Y and X, from the ``.jdce`` descriptor. Y and X are
            scaled to the current resolution level, which is read off a plane, so
            this opens one TIFF the first time a scene is asked about.
        """
        scale = self._level_scale()
        pixel_y = self._unit.jdce.pixel_size_y
        pixel_x = self._unit.jdce.pixel_size_x

        return PhysicalPixelSizes(
            self._unit.jdce.z_step,
            pixel_y * scale if pixel_y is not None else None,
            pixel_x * scale if pixel_x is not None else None,
        )

    @property
    def binning(self) -> Optional[str]:
        """Camera binning as ``"<x>x<y>"``, e.g. ``"1x1"``."""
        return self._unit.jdce.binning

    @property
    def objective(self) -> Optional[str]:
        """The objective the acquisition was taken with."""
        return self._unit.jdce.objective

    @property
    def row(self) -> Optional[str]:
        """Plate row letters of the current scene, e.g. ``"B"``."""
        return acquisition.split_well(self._current()[0])[0]

    @property
    def column(self) -> Optional[str]:
        """Plate column of the current scene, unpadded, e.g. ``"7"``."""
        return str(acquisition.split_well(self._current()[0])[1])

    @property
    def position_index(self) -> Optional[int]:
        """
        The current scene's acquisition position within its well.

        None with ``mosaic=True``, where a scene is a whole well rather than one
        of its positions.
        """
        return None if self._mosaic else self._current()[1][0]

    @property
    def imaged_by(self) -> Optional[str]:
        """The acquisition protocol's user, falling back to the station login."""
        return self._unit.jdce.operator

    @property
    def imaging_datetime(self) -> Optional[datetime]:
        """
        When the acquisition began.

        The descriptor's ``Creation`` stamp, which is naive instrument local
        time. Falls back to the manifest's first timestamp, which is a Unix epoch
        and so returns a UTC-aware datetime instead.
        """
        if self._unit.jdce.acquired_at is not None:
            return self._unit.jdce.acquired_at

        first = self._unit.first_timestamp
        if first is None:
            return None

        return datetime.fromtimestamp(first, tz=timezone.utc)

    @property
    def stage_position(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Stage X and Y of the current scene in microns, from the manifest.

        With ``mosaic=True`` this is the first tile's position. Every tile's own
        position is on ``metadata["tile_stage_positions_um"]``; the stitched
        image is placed from the top-left-most of them, which need not be this
        one.
        """
        well, sites = self._current()

        return self._unit.positions.get((well, sites[0]), (None, None))

    @property
    def time_interval(self) -> TimeInterval:
        """
        Average interval between the current scene's time points.

        Measured from the manifest's timestamps rather than the descriptor's
        schedule, so a run that drifted or was cut short reports what happened.
        """
        duration = self.total_time_duration
        if duration is None:
            return None

        timepoints, _, _ = self._unit.extents(self._current_keys())

        return duration / (len(timepoints) - 1)

    @property
    def total_time_duration(self) -> Optional[timedelta]:
        """
        Elapsed time from the first to the last time point of the current scene.

        None when the manifest is absent or does not timestamp every time point.
        """
        elapsed = self._unit.time_coords(self._current_keys())
        if elapsed is None or len(elapsed) < 2:
            return None

        return timedelta(seconds=elapsed[-1] - elapsed[0])

    @property
    def standard_metadata(self) -> StandardMetadata:
        """
        Returns
        -------
        metadata: StandardMetadata
            The standard field set, filled from the descriptor and the manifest.
            Sizes, dimension order and pixel sizes come from the base.
        """
        metadata = super().standard_metadata

        metadata.binning = self.binning
        metadata.column = self.column
        metadata.imaged_by = self.imaged_by
        metadata.imaging_datetime = self.imaging_datetime
        metadata.objective = self.objective
        metadata.position_index = self.position_index
        metadata.row = self.row
        metadata.stage_position_x, metadata.stage_position_y = self.stage_position
        metadata.total_time_duration = self.total_time_duration

        return metadata

    # Mosaic

    def get_mosaic_tile_position(
        self, mosaic_tile_index: int, **kwargs: int
    ) -> Tuple[int, int]:
        """
        Parameters
        ----------
        mosaic_tile_index: int
            The tile to get the position of.

        Returns
        -------
        position: Tuple[int, int]
            The tile's top-left pixel within the stitched well, as (top, left).
        """
        return self.get_mosaic_tile_positions(**kwargs)[mosaic_tile_index]

    def get_mosaic_tile_positions(self, **kwargs: int) -> List[Tuple[int, int]]:
        """
        Returns
        -------
        positions: List[Tuple[int, int]]
            The top-left pixel of every tile in the current scene, in ``M`` order.

        Raises
        ------
        exceptions.UnexpectedShapeError
            The scene has no mosaic tile dimension.
        ValueError
            The manifest carries no stage positions to place the tiles from.
        """
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError(
                "Cannot compute tile positions for an image without tiles."
            )

        return self._tile_positions()

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        return self._stitch(self.xarray_dask_data)

    def _get_stitched_mosaic(self) -> xr.DataArray:
        return self._stitch(self.xarray_data)

    def _tile_positions(self) -> List[Tuple[int, int]]:
        """
        Tile origins in pixels at the current resolution level.

        Placed from the manifest's stage coordinates, scaled by the descriptor's
        pixel size and referenced to the top-left-most tile. Both axes run the
        same way as the image: a larger stage Y is further down.
        """
        well, sites = self._current()
        if len(sites) == 1:
            return [(0, 0)]

        stage = [self._unit.positions.get((well, site), (None, None)) for site in sites]
        pixel_y = self._unit.jdce.pixel_size_y
        pixel_x = self._unit.jdce.pixel_size_x

        if not all(x is not None and y is not None for x, y in stage) or not (
            pixel_x and pixel_y
        ):
            raise ValueError(
                f"Scene '{self.current_scene}' has {len(sites)} tiles but no stage "
                "positions or pixel size to place them from."
            )

        positions = [(float(x), float(y)) for x, y in stage]  # type: ignore[arg-type]
        scale = self._level_scale()
        step_y, step_x = pixel_y * scale, pixel_x * scale
        origin_x = min(x for x, _ in positions)
        origin_y = min(y for _, y in positions)

        return [
            (int(round((y - origin_y) / step_y)), int(round((x - origin_x) / step_x)))
            for x, y in positions
        ]

    def _stitch(self, tiles: xr.DataArray) -> xr.DataArray:
        """Lay the tiles into one TCZYX array, resolving overlap last tile wins."""
        positions = self._tile_positions()
        tile_y, tile_x = tiles.shape[-2:]
        height = max(top for top, _ in positions) + tile_y
        width = max(left for _, left in positions) + tile_x

        data = tiles.data
        shape = data.shape[1:-2] + (height, width)
        stitched = (
            da.zeros(shape, dtype=data.dtype, chunks=data.chunksize[1:])
            if isinstance(data, da.Array)
            else np.zeros(shape, dtype=data.dtype)
        )

        for tile, (top, left) in enumerate(positions):
            stitched[..., top : top + tile_y, left : left + tile_x] = data[tile]

        return xr.DataArray(
            stitched,
            dims=DEFAULT_DIMENSION_ORDER_LIST,
            coords={
                name: value
                for name, value in tiles.coords.items()
                if name != DimensionNames.MosaicTile
            },
            attrs=tiles.attrs,
        )

    # Internals

    def _current(self) -> Tuple[str, Tuple[int, ...]]:
        return self._scene_table[self._current_scene_index]

    def _current_keys(self) -> List[SceneKey]:
        well, sites = self._current()

        return [(well, site) for site in sites]

    def _levels(self) -> Tuple[List[Tuple[int, ...]], np.dtype]:
        """
        The current scene's pyramid shapes and dtype, probed from one plane.

        A plane that cannot be opened must not take the whole scene's metadata
        with it, so the probe moves on to the next plane; its own pixels still
        raise when they are asked for.
        """
        scene_index = self._current_scene_index
        if scene_index not in self._scene_levels:
            error: Optional[Exception] = None
            for path in self._plane_paths():
                try:
                    with self._fs.open(path, "rb") as handle:
                        with tifffile.TiffFile(handle) as tiff:
                            series = tiff.series[0]
                            self._scene_levels[scene_index] = (
                                [tuple(level.shape) for level in series.levels],
                                np.dtype(series.dtype),
                            )
                            try:
                                self._scene_metaseries[
                                    scene_index
                                ] = tiff.metaseries_metadata
                            except Exception as exc:
                                log.debug("Could not read MetaSeries tags: %s", exc)
                                self._scene_metaseries[scene_index] = None
                    break
                except Exception as exc:
                    error = exc
                    log.warning("Could not probe %s: %s", path, exc)
            else:
                raise IOError(
                    f"No plane of scene '{self.current_scene}' could be opened."
                ) from error

        return self._scene_levels[scene_index]

    def _plane_paths(self) -> List[str]:
        """Every plane path of the current scene, in index order."""
        return [
            self._unit.planes[key][plane]
            for key in self._current_keys()
            for plane in sorted(self._unit.planes[key])
        ]

    def _level(self) -> int:
        level = self._current_resolution_level
        available = len(self._levels()[0])
        if level >= available:
            # set_resolution_level validated against another scene's pyramid.
            raise IndexError(
                f"Scene '{self.current_scene}' has {available} resolution levels; "
                f"level {level} was set. Call set_resolution_level to pick one of "
                "this scene's levels."
            )

        return level

    def _level_scale(self) -> float:
        """Linear downsample factor of the current level relative to level 0."""
        shapes, _ = self._levels()

        return shapes[0][-1] / shapes[self._level()][-1]

    def _plane_spec(self) -> Tuple[Tuple[int, ...], np.dtype]:
        shapes, dtype = self._levels()

        return shapes[self._level()], dtype

    def _plane_array(
        self,
        path: Optional[str],
        shape: Tuple[int, ...],
        dtype: np.dtype,
        level: int,
        delayed_read: bool,
    ) -> Any:
        if not delayed_read:
            return _read_plane(self._fs, path, level)

        return da.from_delayed(
            delayed(_read_plane)(self._fs, path, level), shape=shape, dtype=dtype
        )

    def _coords(self, zs: List[int]) -> Dict[str, Any]:
        keys = self._current_keys()
        coords: Dict[str, Any] = {
            DimensionNames.Channel: self._unit.channel_names(keys)
        }

        times = self._unit.time_coords(keys)
        if times is not None:
            coords[DimensionNames.Time] = times

        if self._unit.jdce.z_step:
            coords[DimensionNames.SpatialZ] = [z * self._unit.jdce.z_step for z in zs]

        return coords

    def _unprocessed_metadata(self) -> Dict[str, Any]:
        """The descriptor plus the plate position a writer needs to place a scene."""
        well, sites = self._current()
        positions = [self._unit.positions.get((well, s), (None, None)) for s in sites]
        origin_x, origin_y = positions[0]

        row, column = acquisition.split_well(well)
        metadata: Dict[str, Any] = {
            "jdce": self._unit.jdce.raw,
            "well": well,
            "row": row,
            "column": column,
            "sites": list(sites),
            "stage_position_um": {"x": origin_x, "y": origin_y},
            "tile_stage_positions_um": [
                {"site": site, "x": x, "y": y} for site, (x, y) in zip(sites, positions)
            ],
        }

        self._levels()  # The probe that fills the MetaSeries cache.
        metaseries = self._scene_metaseries.get(self._current_scene_index)
        if metaseries is not None:
            metadata["metaseries"] = metaseries

        return metadata

    def _why_unsupported(self) -> str:
        sub_units = acquisition.find_sub_units(self._fs, self._path)
        if sub_units:
            return (
                "This is a run root holding several acquisitions; open one of: "
                f"{', '.join(sub_units)}."
            )

        return (
            "Expected an ImageXpress acquisition directory containing a '.jdce' "
            "descriptor and 'timepoint<N>' folders, or the '.jdce' file itself. "
            "On a filesystem that serves files but cannot list directories, name "
            "the '.jdce' descriptor directly."
        )


###############################################################################


def _read_plane(fs: AbstractFileSystem, path: Optional[str], level: int) -> np.ndarray:
    if path is None:
        raise FileNotFoundError("The acquisition indexes no plane at this coordinate.")

    with fs.open(path, "rb") as handle:
        with tifffile.TiffFile(handle) as tiff:
            series = tiff.series[0]
            if level >= len(series.levels):
                # Quietly returning another level would hand the dask graph a
                # chunk of the wrong shape, and it does not check.
                raise IndexError(
                    f"{path} has {len(series.levels)} resolution levels; "
                    f"level {level} was requested."
                )

            return series.levels[level].asarray()
