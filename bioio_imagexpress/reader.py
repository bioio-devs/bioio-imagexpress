#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime, timedelta, timezone
from itertools import product
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple, cast

import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from bioio_base import constants, exceptions, io
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER_LIST,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES,
    DimensionNames,
    Dimensions,
)
from bioio_base.reader import Reader as BaseReader
from bioio_base.standard_metadata import StandardMetadata
from bioio_base.types import DimSpec, PhysicalPixelSizes, TimeInterval
from dask import delayed
from fsspec.spec import AbstractFileSystem

from . import index, parsers
from .index import AcquisitionUnit, PlaneKey, SceneKey

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Reader(BaseReader):
    """
    Reader for Molecular Devices ImageXpress / MetaXpress acquisitions.

    An ImageXpress acquisition is a directory rather than a single file: each plane
    is its own TIFF, and the structure tying them together lives in a ``.jdce``
    descriptor, an ``image_metadata_*.csv`` manifest, and the plane filenames. This
    reader assembles one ``MTCZYX`` image per well.

    Parameters
    ----------
    image: Any
        Path to an acquisition directory, to a run root containing several, or to a
        ``.jdce`` descriptor.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    verify_planes: bool
        Confirm every manifest row against the files actually present. Costs a full
        directory walk and needs a listable filesystem; see ``index.build_unit``.
        Default: False
    mosaic: bool
        Treat a well's acquisition positions as the mosaic tiles they are: one scene
        per well, tiles on the ``M`` dimension, stitched through
        ``mosaic_xarray_data``. Turn it off to get one scene per (well, site) and no
        ``M`` dimension, which is what you want when the tiles are the unit of
        analysis rather than the well.
        Default: True

    Notes
    -----
    Name the ``.jdce`` descriptor. Indexing is manifest-driven -- the descriptor
    names its ``image_metadata_*.csv`` manifests, and those name every plane's
    subfolder and filename -- so a unit resolves from two file reads without
    listing a single directory::

        Reader("/path/to/experiment/Acquisition.jdce")
        BioImage("/path/to/experiment/Acquisition.jdce")

    That is the form ``bioio`` can route on its own, since it matches plugins by
    path suffix, and the only form that works on a filesystem which serves files
    but refuses to list, such as a read-only HTTP endpoint::

        Reader("https://host/experiment/Acquisition.jdce")

    An acquisition directory or a run root still reads, on a filesystem that can
    list, but has to be routed by hand -- a directory name carries no suffix::

        Reader("/path/to/experiment")
        BioImage("/path/to/experiment", reader=Reader)

    Expected layout::

        <PlateBarcode>/<Protocol>_<timestamp>/     # run root
          experiment/                              # acquisition unit
              <name>.jdce
              image_metadata_1.csv
              timepoint0/
                  <Project>_t0_<Well>_s<Site>_w<Channel>_z<Z>.tif
              timepoint1/ ...
          experiment_montage/ experiment_z_stack/  # further units

    Scenes are one per well. A well is imaged as a grid of overlapping fields --
    MetaXpress's own ``experiment_montage`` unit is its stitch of exactly those --
    so those sites are mosaic tiles and land on ``M``. Time points, channels and Z
    steps stack into the array. A run root exposes every unit's scenes, prefixed
    with the unit name; pointing at a single unit leaves the ids unprefixed.

    Units within one run root routinely differ in shape, channel count and pixel
    size, so shape and metadata are always resolved against the current scene.

    Stitching places tiles from their manifest stage positions, last tile winning
    in the overlap. MetaXpress refines its own montage by image registration, so
    ``experiment_montage`` is not reproduced pixel for pixel -- prefer that unit
    where it exists, and this where it does not.
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    _fs: "AbstractFileSystem"
    _path: str

    NAME = "bioio-imagexpress"

    # Required Methods

    def __init__(
        self,
        image: Any,
        fs_kwargs: Dict[str, Any] = {},
        verify_planes: bool = False,
        mosaic: bool = True,
        **kwargs: Any,
    ):
        self._fs, self._path = io.pathlike_to_fs(
            image, enforce_exists=True, fs_kwargs=fs_kwargs
        )

        discovered = parsers.discover_units(self._fs, self._path)
        if not discovered:
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                msg_extra=(
                    "Expected an ImageXpress acquisition directory containing a "
                    "'.jdce' descriptor and 'timepoint<N>' folders, a run root "
                    "containing such directories, or a '.jdce' file. On a "
                    "filesystem that serves files but cannot list directories, "
                    "name the '.jdce' descriptor directly."
                ),
            )

        # Index every unit up front. This is manifest parsing only -- no TIFF is
        # opened until pixels are actually requested.
        self._units: List[AcquisitionUnit] = []
        for unit_spec in discovered:
            unit = index.build_unit(self._fs, unit_spec, verify_planes=verify_planes)
            if unit.planes:
                self._units.append(unit)

        if not self._units:
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__,
                self._path,
                msg_extra="No readable planes were found in this acquisition.",
            )

        # Flat scene table: scene index -> (unit, well, the tiles it covers).
        #
        # A well is imaged as a grid of overlapping fields, so with mosaic on a
        # scene is a whole well and its sites become the M dimension. With it off
        # each tile is its own scene and M does not appear at all.
        self._mosaic = mosaic
        self._scene_table: List[Tuple[AcquisitionUnit, str, Tuple[int, ...]]] = []
        for unit in self._units:
            if mosaic:
                self._scene_table.extend(
                    (unit, well, tuple(unit.sites(well))) for well in unit.wells
                )
            else:
                self._scene_table.extend(
                    (unit, well, (site,)) for well, site in unit.scene_keys
                )

        # Per-scene shape/dtype, filled in on first access. Keyed by scene index so
        # a scene switch can never serve another scene's shape.
        self._plane_specs: Dict[int, Tuple[Tuple[int, int], np.dtype]] = {}
        self._level_shapes: Dict[int, List[Tuple[int, ...]]] = {}

    @staticmethod
    def _is_supported_image(fs: "AbstractFileSystem", path: str, **kwargs: Any) -> bool:
        """
        Accept a directory that resolves to at least one acquisition unit, or a
        ``.jdce`` descriptor. Bare TIFFs are deliberately not claimed -- they
        belong to bioio-tifffile.
        """
        return bool(parsers.discover_units(fs, path))

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        One scene per well (``"B07"``), or per well and tile with mosaic off
        (``"B07-s0"``).

        When several acquisition units were found under a run root, ids are
        prefixed with the unit name (``"experiment_z_stack/B07"``) so that scenes
        from different units stay distinguishable.
        """
        if self._scenes is None:
            self._scenes = tuple(
                _scene_id(unit, well, sites, self._mosaic)
                for unit, well, sites in self._scene_table
            )

        return self._scenes

    def _read_delayed(self) -> "xr.DataArray":
        return self._build(delayed_read=True)

    def _read_immediate(self) -> "xr.DataArray":
        return self._build(delayed_read=False)

    def _read_indexed(self, given_dims: str, dim_specs: List[DimSpec]) -> "np.ndarray":
        """
        Return the native-order array with ``dim_specs`` applied.

        This is where ``get_image_data`` earns its keep on this format. Every
        plane is a separate file, so the base implementation -- materialize the
        whole scene, then slice -- opens one file per T x C x Z even when a single
        plane was asked for. Here the T, C and Z specs are resolved to the exact
        set of files they name, and only those are opened. Asking for one plane of
        a 9 x 2 x 11 scene reads 1 file rather than 198.

        The Y and X specs are applied after each plane is read: a MetaXpress plane
        is one strip-per-row TIFF with no tiling to exploit, so a spatial crop
        saves memory rather than I/O.

        Parameters
        ----------
        given_dims: str
            The native dimension ordering of the image (``self.dims.order``).
        dim_specs: List[DimSpec]
            One getitem operation per dimension in ``given_dims``, as produced by
            ``transforms.compute_dim_specs``.

        Returns
        -------
        data: np.ndarray
            The indexed image data in native (reduced) dimension order.
        """
        unit, well, sites = self._current()
        timepoints, channels, zs = unit.extents(self._current_keys())

        plane_shape, dtype = self._plane_spec()
        level = min(
            self._current_resolution_level, len(self._current_level_shapes()) - 1
        )

        # Plane-selecting dims index into the coordinates that actually exist on
        # disk, which an aborted acquisition leaves sparse and non-contiguous.
        coordinates = {
            DimensionNames.MosaicTile: list(sites),
            DimensionNames.Time: timepoints,
            DimensionNames.Channel: channels,
            DimensionNames.SpatialZ: zs,
        }
        spatial = (DimensionNames.SpatialY, DimensionNames.SpatialX)
        plane_specs = tuple(
            spec for dim, spec in zip(given_dims, dim_specs) if dim in spatial
        )

        # Per plane-selecting dim, the (output position, coordinate) pairs it
        # selects. An integer spec contributes no output axis, matching the base's
        # ``self.data[tuple(dim_specs)]``.
        plane_dims = [dim for dim in given_dims if dim not in spatial]
        selections: List[List[Tuple[Optional[int], int]]] = []
        for dim, spec in zip(given_dims, dim_specs):
            if dim in spatial:
                continue

            values = coordinates[dim]
            if isinstance(spec, Integral):
                selections.append([(None, values[int(spec)])])
            elif isinstance(spec, slice):
                selections.append(list(enumerate(values[spec])))
            else:
                indices = cast(List[int], spec)
                selections.append(list(enumerate(values[i] for i in indices)))

        subset = np.empty(
            self._indexed_shape(given_dims, dim_specs, plane_shape), dtype=dtype
        )
        if 0 in subset.shape:
            return subset

        requested = 0
        missing: List[Tuple[int, ...]] = []
        for combination in product(*selections):
            chosen = dict(zip(plane_dims, combination))
            # Keyed by name rather than by position so the plane lookup does not
            # silently depend on MTCZYX being the native order.
            key: PlaneKey = (
                chosen[DimensionNames.Time][1],
                chosen[DimensionNames.Channel][1],
                chosen[DimensionNames.SpatialZ][1],
            )
            site = (
                chosen[DimensionNames.MosaicTile][1]
                if DimensionNames.MosaicTile in chosen
                else sites[0]
            )

            requested += 1
            path = unit.planes.get((well, site), {}).get(key)
            if path is None:
                missing.append((site,) + key)

            subset_index = tuple(
                slice(None) if dim in spatial else chosen[dim][0]
                for dim, spec in zip(given_dims, dim_specs)
                if not isinstance(spec, Integral)
            )
            plane = _read_plane(self._fs, path, plane_shape, dtype, level)
            subset[subset_index] = plane[plane_specs]

        if missing:
            log.warning(
                "Scene '%s' is missing %d of the %d requested plane(s); "
                "filled with zeros.",
                self.current_scene,
                len(missing),
                requested,
            )

        return subset

    def _indexed_shape(
        self,
        given_dims: str,
        dim_specs: List[DimSpec],
        plane_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """
        The shape ``dim_specs`` produces, computed without allocating a plane.
        """
        sizes = dict(zip(given_dims, self.dims.shape))
        sizes[DimensionNames.SpatialY] = plane_shape[0]
        sizes[DimensionNames.SpatialX] = plane_shape[1]

        shape = []
        for dim, spec in zip(given_dims, dim_specs):
            if isinstance(spec, Integral):
                continue
            if isinstance(spec, slice):
                shape.append(len(range(*spec.indices(sizes[dim]))))
            else:
                shape.append(len(cast(List[int], spec)))

        return tuple(shape)

    # Resolution levels

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        """
        MetaXpress writes each plane as a small pyramid, which lines up with the
        multiscale levels an OME-Zarr conversion wants.
        """
        return tuple(range(len(self._current_level_shapes())))

    # Metadata

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Physical pixel sizes in micrometers, from the ``.jdce`` descriptor.

        Z spacing is the configured step, which is only recorded in the
        descriptor. Y and X are scaled to the current resolution level.
        """
        unit, _, _ = self._current()
        scale = self._level_scale()

        pixel_y = unit.jdce.pixel_size_y
        pixel_x = unit.jdce.pixel_size_x

        return PhysicalPixelSizes(
            unit.jdce.z_step,
            pixel_y * scale if pixel_y is not None else None,
            pixel_x * scale if pixel_x is not None else None,
        )

    @property
    def binning(self) -> Optional[str]:
        """
        Camera binning as ``"<x>x<y>"``, e.g. ``"1x1"``, from the descriptor.
        """
        return self._current()[0].jdce.binning

    @property
    def row(self) -> Optional[str]:
        """Plate row letter of the current scene, e.g. ``"B"``."""
        return self._current()[1][0]

    @property
    def column(self) -> Optional[str]:
        """
        Plate column of the current scene as an unpadded string, e.g. ``"7"``.

        ``StandardMetadata`` types row and column as strings, so the zero padding
        the filenames carry (``B07``) is dropped here; the padded well label stays
        available under ``attrs["unprocessed"]["well"]``.
        """
        return str(int(self._current()[1][1:]))

    @property
    def position_index(self) -> Optional[int]:
        """
        Tile index of the current scene within its well.

        None with mosaic on, where the scene is the whole well rather than one
        of its acquisition positions.
        """
        _, _, sites = self._current()

        return None if self._mosaic else sites[0]

    @property
    def imaged_by(self) -> Optional[str]:
        """The acquisition protocol's user, falling back to the station login."""
        return self._current()[0].jdce.operator

    @property
    def imaging_datetime(self) -> Optional[datetime]:
        """
        When the acquisition began.

        Normally the descriptor's ``Creation`` stamp, which is naive instrument
        local time (see ``parsers._parse_creation``). Only when the descriptor
        does not carry one does this fall back to the manifest's first
        ``TimeStampSec``, which is a Unix epoch and therefore returns a UTC-aware
        datetime instead -- worth knowing before comparing values across
        acquisitions.
        """
        unit, _, _ = self._current()
        if unit.jdce.acquired_at is not None:
            return unit.jdce.acquired_at

        if not unit.timestamps:
            return None

        return datetime.fromtimestamp(min(unit.timestamps.values()), tz=timezone.utc)

    @property
    def stage_position(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Stage X and Y of the current scene in microns, from the manifest.

        With mosaic on this is the first tile's position, which is the origin the
        stitched image is placed from -- the individual tile positions are on
        ``metadata["tile_stage_positions_um"]``.
        """
        unit, well, sites = self._current()

        return unit.positions.get((well, sites[0]), (None, None))

    @property
    def time_interval(self) -> TimeInterval:
        """
        Average interval between the current scene's time points.

        Measured from the manifest's timestamps rather than the descriptor's
        ``TimeSchedule``, so a run that drifted or was cut short reports what
        actually happened.
        """
        duration = self.total_time_duration
        if duration is None:
            return None

        unit, _, _ = self._current()
        timepoints, _, _ = unit.extents(self._current_keys())

        return duration / (len(timepoints) - 1)

    @property
    def total_time_duration(self) -> Optional[timedelta]:
        """
        Elapsed time from the first to the last time point of the current scene.

        None when the manifest is absent or does not timestamp every time point.
        """
        unit, _, _ = self._current()

        elapsed = unit.time_coords(self._current_keys())
        if elapsed is None or len(elapsed) < 2:
            return None

        return timedelta(seconds=elapsed[-1] - elapsed[0])

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        The format's own metadata for the current scene, as JSON-able Python.

        Both files an ImageXpress acquisition carries its metadata in are here, in
        full and in their native structure::

            reader.metadata["jdce"]            # the descriptor, verbatim
            reader.metadata["image_metadata"]  # this scene's manifest rows
            reader.metadata["metaseries"]      # the plane TIFF's MetaSeries tags

        ``jdce`` is JSON already -- the extension is unusual, the content is not.
        ``image_metadata`` is the CSV manifest turned into one dict per row with
        every column kept, so the per-plane record (exposure, intensity
        statistics, temperature, CO2, O2, field offsets, FOV uuid) survives the
        conversion. The plate-position entries the base put in
        ``xarray_dask_data.attrs["unprocessed"]`` are kept verbatim alongside.

        This stays format-native throughout; the normalized field set lives on
        :attr:`standard_metadata`.

        The manifest rows are read on demand rather than held: a 24,000 plane
        manifest is 5 MB of CSV, and only the current scene's slice of it is ever
        wanted. The result is cached until the scene changes.
        """
        if self._metadata is None:
            unit, well, sites = self._current()
            unprocessed = self.xarray_dask_data.attrs[constants.METADATA_UNPROCESSED]
            self._metadata = {
                **unprocessed,
                "image_metadata": self._manifest_records(unit, well, sites),
            }

        return self._metadata

    def _manifest_records(
        self, unit: AcquisitionUnit, well: str, sites: Tuple[int, ...]
    ) -> List[Dict[str, Optional[str]]]:
        """
        This scene's manifest rows, every column kept.

        Covers every tile of the well with mosaic on, one tile with it off.

        Empty when the acquisition had no readable manifest -- that unit was
        indexed from filenames alone, which carry nothing beyond the coordinates
        already in ``dims``.
        """
        wanted = set(sites)
        records: List[Dict[str, Optional[str]]] = []

        for path in unit.manifests:
            try:
                with self._fs.open(path, "r", encoding="utf-8-sig") as handle:
                    rows = parsers.read_image_metadata_records(handle.read())
            except Exception as exc:
                log.warning("Could not read manifest %s: %s", path, exc)
                continue

            records.extend(
                row
                for row in rows
                if parsers.normalize_well(row.get("Well") or "") == well
                and _as_site(row.get("Field")) in wanted
            )

        return records

    @property
    def standard_metadata(self) -> StandardMetadata:
        """
        The standard field set, filled from the descriptor and manifest.

        Sizes, dimension order and pixel sizes come from the base implementation.
        Everything else is resolved against the current scene, since a run root
        can hold units that disagree about all of it.
        """
        metadata = super().standard_metadata

        metadata.binning = self.binning
        metadata.column = self.column
        metadata.imaged_by = self.imaged_by
        metadata.imaging_datetime = self.imaging_datetime
        metadata.objective = self._current()[0].jdce.objective
        metadata.position_index = self.position_index
        metadata.row = self.row
        metadata.stage_position_x, metadata.stage_position_y = self.stage_position
        metadata.timelapse_interval = self.time_interval
        metadata.total_time_duration = self.total_time_duration

        return metadata

    # Internals

    def _current(self) -> Tuple[AcquisitionUnit, str, Tuple[int, ...]]:
        """The current scene as ``(unit, well, the tiles it covers)``."""
        return self._scene_table[self._current_scene_index]

    def _current_keys(self) -> List[SceneKey]:
        """The ``(well, site)`` index keys the current scene draws planes from."""
        _, well, sites = self._current()

        return [(well, site) for site in sites]

    def _representative_plane(self) -> str:
        """The first plane of the current scene, whether or not it is readable."""
        unit, well, sites = self._current()
        planes = unit.planes[(well, sites[0])]

        return planes[min(planes)]

    def _candidate_planes(self) -> List[str]:
        """
        Planes to try when probing the scene for shape and dtype, best first.

        Indexing is manifest-driven, so a plane path is composed rather than
        confirmed and the first one need not exist -- a part-copied acquisition is
        the common case on an object store. Every other read treats an unreadable
        plane as recoverable, so this probe must too, or one absent file would
        take down the whole scene.
        """
        unit, well, sites = self._current()

        return [
            unit.planes[(well, site)][key]
            for site in sites
            for key in sorted(unit.planes.get((well, site), {}))
        ]

    def _current_level_shapes(self) -> List[Tuple[int, ...]]:
        """
        Pyramid level shapes for the current scene, read from one plane.

        Cached per scene index because a run root can mix units whose planes have
        different numbers of levels.
        """
        scene_index = self._current_scene_index
        if scene_index not in self._level_shapes:
            self._level_shapes[scene_index] = self._probe_scene(scene_index)

        return self._level_shapes[scene_index]

    def _probe_scene(self, scene_index: int) -> List[Tuple[int, ...]]:
        """
        Open planes until one yields the scene's level shapes and dtype.

        Raises the first failure if none of them can be read, since a scene with
        no readable plane at all has nothing to report a shape from.
        """
        first_error: Optional[Exception] = None

        for path in self._candidate_planes():
            try:
                with self._fs.open(path, "rb") as handle:
                    with tifffile.TiffFile(handle) as tiff:
                        series = tiff.series[0]
                        shapes = [tuple(level.shape) for level in series.levels]
                        self._plane_specs[scene_index] = (
                            tuple(series.levels[0].shape),  # type: ignore[assignment]
                            np.dtype(series.dtype),
                        )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                log.debug("Could not probe plane %s: %s", path, exc)
                continue

            if first_error is not None:
                log.warning(
                    "Scene '%s' was probed from %s; earlier plane(s) unreadable.",
                    self.current_scene,
                    path,
                )

            return shapes

        if first_error is not None:
            raise first_error

        raise exceptions.UnsupportedFileFormatError(
            self.__class__.__name__,
            self._path,
            msg_extra=f"Scene '{self.current_scene}' indexes no planes.",
        )

    def _level_scale(self) -> float:
        """Linear downsample factor of the current level relative to level 0."""
        shapes = self._current_level_shapes()
        level = min(self._current_resolution_level, len(shapes) - 1)
        if not shapes or shapes[level][-1] == 0:
            return 1.0

        return shapes[0][-1] / shapes[level][-1]

    def _plane_spec(self) -> Tuple[Tuple[int, ...], np.dtype]:
        """Shape and dtype of a single plane at the current resolution level."""
        shapes = self._current_level_shapes()
        level = min(self._current_resolution_level, len(shapes) - 1)
        _, dtype = self._plane_specs[self._current_scene_index]

        return shapes[level], dtype

    def _build(self, delayed_read: bool) -> "xr.DataArray":
        unit, well, sites = self._current()
        timepoints, channels, zs = unit.extents(self._current_keys())

        plane_shape, dtype = self._plane_spec()
        level = min(
            self._current_resolution_level, len(self._current_level_shapes()) - 1
        )

        # ``missing`` entries are (t, c, z) with mosaic off and (m, t, c, z) with
        # it on, matching the dimensions the array actually has.
        missing: List[Tuple[int, ...]] = []
        tiles = []
        for site in sites:
            planes = unit.planes.get((well, site), {})
            stacks = []
            for t in timepoints:
                channel_stack = []
                for channel in channels:
                    z_stack = []
                    for z in zs:
                        path = planes.get((t, channel, z))
                        if path is None:
                            missing.append(
                                (site, t, channel, z)
                                if self._mosaic
                                else (t, channel, z)
                            )

                        z_stack.append(
                            self._plane_array(
                                path, plane_shape, dtype, level, delayed_read
                            )
                        )
                    channel_stack.append(_stack(z_stack, delayed_read))
                stacks.append(_stack(channel_stack, delayed_read))
            tiles.append(_stack(stacks, delayed_read))

        data = _stack(tiles, delayed_read) if self._mosaic else tiles[0]

        if missing:
            log.warning(
                "Scene '%s' is missing %d plane(s); filled with zeros.",
                self.current_scene,
                len(missing),
            )

        return xr.DataArray(
            data,
            dims=self._dimension_order(),
            coords=self._coords(unit, zs),
            attrs={
                constants.METADATA_UNPROCESSED: self._unprocessed_metadata(unit, well),
                "missing_planes": missing,
            },
        )

    def _dimension_order(self) -> List[str]:
        """``MTCZYX`` when tiles are stacked into the array, ``TCZYX`` when not."""
        if self._mosaic:
            return DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

        return DEFAULT_DIMENSION_ORDER_LIST

    # Mosaic

    def get_mosaic_tile_position(
        self, mosaic_tile_index: int, **kwargs: int
    ) -> Tuple[int, int]:
        """
        Top-left pixel of one tile within the stitched well.

        Positions come from the manifest's stage coordinates, scaled by pixel size
        and referenced to the top-left-most tile. Both axes run the same way as
        the image: a larger stage Y is further down, verified against MetaXpress's
        own ``experiment_montage`` stitch of the same wells.
        """
        return self.get_mosaic_tile_positions(**kwargs)[mosaic_tile_index]

    def get_mosaic_tile_positions(self, **kwargs: int) -> List[Tuple[int, int]]:
        """
        Top-left pixel of every tile in the current scene, in ``M`` order.

        Raises
        ------
        UnexpectedShapeError
            The scene has no mosaic tile dimension.
        MetadataNotFoundError
            The manifest carries no stage position to place the tiles from.
        """
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError(
                "Cannot compute tile positions for an image without tiles."
            )

        return self._tile_positions()

    def _tile_positions(self) -> List[Tuple[int, int]]:
        """
        Tile origins in pixels at the current resolution level.

        A single-tile well needs no metadata -- it is its own origin -- which
        keeps `experiment_montage` (already stitched, one tile per well) and
        filename-indexed acquisitions usable.
        """
        unit, well, sites = self._current()
        if len(sites) == 1:
            return [(0, 0)]

        positions = [unit.positions.get((well, site), (None, None)) for site in sites]
        pixel_y = unit.jdce.pixel_size_y
        pixel_x = unit.jdce.pixel_size_x

        if any(x is None or y is None for x, y in positions) or not (
            pixel_x and pixel_y
        ):
            raise ValueError(
                f"Scene '{self.current_scene}' has {len(sites)} tiles but no stage "
                "positions or pixel size to place them from -- the manifest is "
                "where those live, and this acquisition was indexed from filenames. "
                "Read it with Reader(..., mosaic=False) to get the tiles as "
                "separate scenes."
            )

        scale = self._level_scale()
        step_y = pixel_y * scale
        step_x = pixel_x * scale
        origin_x = min(x for x, _ in positions)  # type: ignore[type-var]
        origin_y = min(y for _, y in positions)  # type: ignore[type-var]

        return [
            (
                int(round((y - origin_y) / step_y)),  # type: ignore[operator]
                int(round((x - origin_x) / step_x)),  # type: ignore[operator]
            )
            for x, y in positions
        ]

    def _get_stitched_dask_mosaic(self) -> "xr.DataArray":
        return self._stitch(self.xarray_dask_data)

    def _get_stitched_mosaic(self) -> "xr.DataArray":
        return self._stitch(self.xarray_data)

    def _stitch(self, tiles: "xr.DataArray") -> "xr.DataArray":
        """
        Lay the tiles into one ``TCZYX`` array at their stage positions.

        Overlap is resolved last-tile-wins rather than blended. MetaXpress's own
        ``experiment_montage`` refines placement by image registration, so its
        output is not reproduced pixel for pixel here -- observed tile spacing
        differs from the stage-derived spacing by a few percent. Use that unit
        where it exists and this where it does not.
        """
        positions = self._tile_positions()
        tile_y, tile_x = tiles.shape[-2:]

        height = max(top for top, _ in positions) + tile_y
        width = max(left for _, left in positions) + tile_x

        data = tiles.data
        stitched = _zeros(data, data.shape[1:-2] + (height, width))

        for tile, (top, left) in enumerate(positions):
            stitched[..., top : top + tile_y, left : left + tile_x] = data[tile]

        return xr.DataArray(
            stitched,
            dims=DEFAULT_DIMENSION_ORDER_LIST,
            coords={
                name: value
                for name, value in tiles.coords.items()
                if name not in (DimensionNames.MosaicTile,)
            },
            attrs=tiles.attrs,
        )

    def _plane_array(
        self,
        path: Optional[str],
        shape: Tuple[int, ...],
        dtype: np.dtype,
        level: int,
        delayed_read: bool,
    ) -> Any:
        if not delayed_read:
            return _read_plane(self._fs, path, shape, dtype, level)

        return da.from_delayed(
            delayed(_read_plane)(self._fs, path, shape, dtype, level),
            shape=shape,
            dtype=dtype,
        )

    def _coords(
        self,
        unit: AcquisitionUnit,
        zs: List[int],
    ) -> Dict[str, Any]:
        keys = self._current_keys()
        coords: Dict[str, Any] = {
            DimensionNames.Channel: unit.channel_names(keys),
        }

        times = unit.time_coords(keys)
        if times is not None:
            coords[DimensionNames.Time] = times

        if unit.jdce.z_step:
            coords[DimensionNames.SpatialZ] = [z * unit.jdce.z_step for z in zs]

        return coords

    def _unprocessed_metadata(self, unit: AcquisitionUnit, well: str) -> Dict[str, Any]:
        """
        Everything a downstream writer might want, without re-reading the files.

        Well and stage position are included because an OME-Zarr HCS plate needs
        them to place each scene.
        """
        _, _, sites = self._current()
        positions = [unit.positions.get((well, site), (None, None)) for site in sites]
        origin = positions[0]

        metadata: Dict[str, Any] = {
            "jdce": unit.jdce.raw,
            "unit": unit.name or None,
            "well": well,
            "row": well[0],
            "column": int(well[1:]),
            "site": sites[0] if not self._mosaic else None,
            "sites": list(sites),
            "stage_position_um": {"x": origin[0], "y": origin[1]},
            "tile_stage_positions_um": [
                {"site": site, "x": x, "y": y} for site, (x, y) in zip(sites, positions)
            ],
            "objective": unit.jdce.objective,
            "plate_id": unit.jdce.plate_id,
            "barcode": unit.jdce.barcode,
            "pyramid_level_shapes": self._current_level_shapes(),
            "indexed_from": "image_metadata_csv" if unit.used_csv else "filenames",
        }

        try:
            with self._fs.open(self._representative_plane(), "rb") as handle:
                with tifffile.TiffFile(handle) as tiff:
                    metadata["metaseries"] = tiff.metaseries_metadata
        except Exception as exc:
            log.debug("Could not read MetaSeries tags: %s", exc)

        return metadata


###############################################################################


def _as_site(value: Optional[str]) -> Optional[int]:
    """The manifest's ``Field`` column as an int, or None if it is not one."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _zeros(reference: Any, shape: Tuple[int, ...]) -> Any:
    """A zero array of ``shape`` in the same backend as ``reference``."""
    if isinstance(reference, da.Array):
        return da.zeros(shape, dtype=reference.dtype, chunks=reference.chunksize[1:])

    return np.zeros(shape, dtype=reference.dtype)


def _scene_id(
    unit: AcquisitionUnit, well: str, sites: Tuple[int, ...], mosaic: bool
) -> str:
    base = well if mosaic else f"{well}-s{sites[0]}"

    return f"{unit.name}/{base}" if unit.name else base


def _stack(arrays: List[Any], delayed_read: bool) -> Any:
    return da.stack(arrays) if delayed_read else np.stack(arrays)


def _read_plane(
    fs: "AbstractFileSystem",
    path: Optional[str],
    shape: Tuple[int, ...],
    dtype: np.dtype,
    level: int,
) -> np.ndarray:
    """
    Read one plane, substituting zeros when it cannot be read.

    A single unreadable plane in a 24,000 file acquisition should cost that plane,
    not the whole image; callers learn about it through ``attrs["missing_planes"]``
    and the logged warning.
    """
    if path is None:
        return np.zeros(shape, dtype=dtype)

    try:
        with fs.open(path, "rb") as handle:
            with tifffile.TiffFile(handle) as tiff:
                series = tiff.series[0]
                data = series.levels[min(level, len(series.levels) - 1)].asarray()
    except Exception as exc:
        log.warning("Could not read plane %s: %s", path, exc)
        return np.zeros(shape, dtype=dtype)

    if data.shape != tuple(shape):
        # Montage planes vary by a pixel or two between wells; pad or crop so the
        # scene still stacks rather than failing outright.
        fitted = np.zeros(shape, dtype=dtype)
        rows = min(shape[0], data.shape[0])
        cols = min(shape[1], data.shape[1])
        fitted[:rows, :cols] = data[:rows, :cols]
        return fitted

    return data.astype(dtype, copy=False)
