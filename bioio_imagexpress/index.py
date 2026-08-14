#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Indexing of an ImageXpress acquisition: which file holds which plane.

Building the index never opens a TIFF. Shape and dtype are discovered later, from
one representative plane per scene, so that constructing a Reader over a 24k-file
acquisition stays cheap.
"""

import logging
import posixpath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from fsspec.spec import AbstractFileSystem

from . import parsers
from .parsers import CsvPlaneRow, JdceMetadata

###############################################################################

log = logging.getLogger(__name__)

# (well, site) -- one scene.
SceneKey = Tuple[str, int]
# (t, channel, z) -- one plane within a scene.
PlaneKey = Tuple[int, int, int]

###############################################################################


@dataclass
class AcquisitionUnit:
    """
    One acquisition -- a descriptor, a plane index, and the metadata to interpret
    it.

    A run root may hold several of these (``experiment``, ``experiment_montage``,
    ``experiment_z_stack``), and they routinely disagree about pixel size, channel
    count and image shape. Everything that varies between them is therefore held
    here rather than on the Reader.
    """

    name: str
    path: str
    jdce: JdceMetadata
    # The manifests that read cleanly, kept so their full records can be re-read
    # on demand rather than held -- a 24,000 plane manifest is 5 MB of CSV, of
    # which any one scene wants a couple of hundred rows.
    manifests: List[str] = field(default_factory=list)
    planes: Dict[SceneKey, Dict[PlaneKey, str]] = field(default_factory=dict)
    timestamps: Dict[int, float] = field(default_factory=dict)
    csv_channel_names: Dict[int, str] = field(default_factory=dict)
    positions: Dict[SceneKey, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )
    used_csv: bool = False

    @property
    def scene_keys(self) -> List[SceneKey]:
        """Every (well, site) present, ordered by plate row, column, then site."""
        return sorted(self.planes, key=lambda key: (key[0][0], int(key[0][1:]), key[1]))

    @property
    def wells(self) -> List[str]:
        """Wells present, in plate order."""
        ordered = []
        for well, _ in self.scene_keys:
            if well not in ordered:
                ordered.append(well)

        return ordered

    def sites(self, well: str) -> List[int]:
        """
        The acquisition positions imaged in a well, in order.

        These are mosaic tiles: MetaXpress images a well as a grid of overlapping
        fields, and ``experiment_montage`` is its own stitch of exactly these.
        """
        return sorted(site for name, site in self.planes if name == well)

    def extents(
        self, keys: Sequence[SceneKey]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        The T, C and Z indices actually present across ``keys``.

        Takes several scene keys because a mosaic scene spans every tile of a
        well, and a tile that dropped a time point must not shrink the others.
        Read off the files rather than the descriptor: an aborted acquisition
        leaves a descriptor promising planes that were never written.
        """
        planes: List[PlaneKey] = []
        for key in keys:
            planes.extend(self.planes[key])

        return (
            parsers.unique_sorted([k[0] for k in planes]),
            parsers.unique_sorted([k[1] for k in planes]),
            parsers.unique_sorted([k[2] for k in planes]),
        )

    def channel_names(self, keys: Sequence[SceneKey]) -> List[str]:
        """
        Channel names for a scene, preferring the descriptor and falling back to
        the CSV's filter column.
        """
        _, channels, _ = self.extents(keys)

        names = []
        for channel in channels:
            if channel < len(self.jdce.channel_names):
                names.append(self.jdce.channel_names[channel])
            elif channel in self.csv_channel_names:
                names.append(self.csv_channel_names[channel])
            else:
                names.append(f"Channel:{channel}")

        return names

    def time_coords(self, keys: Sequence[SceneKey]) -> Optional[List[float]]:
        """
        Elapsed seconds from the start of the acquisition, or None if unknown.

        The descriptor's ``TimeSchedule.Times[].Ms`` values are time point indices
        rather than milliseconds, so they are deliberately not used here.
        """
        timepoints, _, _ = self.extents(keys)
        if not all(t in self.timestamps for t in timepoints):
            return None

        origin = min(self.timestamps.values())
        return [self.timestamps[t] - origin for t in timepoints]


###############################################################################


def build_unit(
    fs: AbstractFileSystem,
    discovered: parsers.DiscoveredUnit,
    *,
    verify_planes: bool = False,
) -> AcquisitionUnit:
    """
    Index one acquisition unit.

    The ``image_metadata_*.csv`` manifest is the index. It names every plane's
    subfolder and filename, so the whole unit resolves from two file reads -- the
    descriptor and the manifests it points at -- with no directory listing at all.
    That is both the fast path (0.2 s against 8 s of walking on the 23,760 plane
    reference acquisition) and the only path that works on a filesystem which
    serves files but refuses to list, such as a read-only HTTP endpoint.

    Falls back to walking ``timepoint<N>`` and parsing the filename grammar when
    no manifest can be read, so an acquisition whose CSVs were never copied still
    opens.

    Parameters
    ----------
    verify_planes: bool
        Confirm every manifest row against the files actually present, dropping
        rows whose plane was never written. Costs a full directory walk and needs
        a listable filesystem. Off by default: a manifest row with no file behind
        it already degrades to a zero-filled plane that is both logged and
        recorded in ``attrs["missing_planes"]``, which beats silently shortening
        the array. Worth turning on for a part-transferred copy, where the
        distinction between "not written" and "not copied yet" matters.
    """
    descriptor = discovered.descriptor or _find_jdce(fs, discovered.path)
    jdce = _load_jdce(fs, descriptor)

    # The descriptor is taken at its word about which manifests exist, so what it
    # names is only a candidate list -- a part-copied acquisition can be missing
    # one. Keep the ones that actually read.
    rows, manifests = _load_csv_rows(
        fs, parsers.resolve_metadata_csvs(fs, discovered.path, jdce)
    )

    unit = AcquisitionUnit(
        name=discovered.name,
        path=discovered.path,
        jdce=jdce,
        manifests=manifests,
    )

    if not rows:
        _index_from_walk(fs, unit)
    elif verify_planes:
        on_disk = _index_from_walk(fs, unit)
        _apply_csv(unit, rows, on_disk)
    else:
        _index_from_rows(unit, rows)

    return unit


def _find_jdce(fs: AbstractFileSystem, path: str) -> Optional[str]:
    """The unit's descriptor, when its name was not handed to us."""
    descriptors = parsers.find_jdce_files(fs, path)

    return descriptors[0] if descriptors else None


def _load_jdce(fs: AbstractFileSystem, descriptor: Optional[str]) -> JdceMetadata:
    if descriptor is None:
        return JdceMetadata()

    try:
        with fs.open(descriptor, "r", encoding="utf-8-sig") as handle:
            return parsers.parse_jdce(handle.read())
    except Exception as exc:
        # A malformed descriptor costs us pixel sizes and channel names, not the
        # image itself.
        log.warning("Could not parse descriptor %s: %s", descriptor, exc)
        return JdceMetadata()


def _load_csv_rows(
    fs: AbstractFileSystem, csv_paths: List[str]
) -> Tuple[List[CsvPlaneRow], List[str]]:
    """Parse the candidate manifests, returning the rows and the ones that read."""
    rows: List[CsvPlaneRow] = []
    readable: List[str] = []

    for csv_path in csv_paths:
        try:
            with fs.open(csv_path, "r", encoding="utf-8-sig") as handle:
                rows.extend(parsers.read_image_metadata_csv(handle.read()))
        except Exception as exc:
            log.warning("Could not parse manifest %s: %s", csv_path, exc)
            continue

        readable.append(csv_path)

    return rows, readable


def _index_from_rows(unit: AcquisitionUnit, rows: List[CsvPlaneRow]) -> None:
    """
    Build the index from the manifest alone, touching no directory.

    Every plane path is composed rather than discovered, so nothing here confirms
    the file is there. That check is deferred to the read, where a missing plane
    becomes zeros plus a warning -- see ``build_unit``'s ``verify_planes``.
    """
    for row in rows:
        scene: SceneKey = (row.well, row.site)
        unit.planes.setdefault(scene, {}).setdefault(
            (row.t, row.channel, row.z),
            posixpath.join(unit.path, row.subfolder, row.filename),
        )
        _apply_row(unit, scene, row)

    unit.used_csv = True


def _index_from_walk(fs: AbstractFileSystem, unit: AcquisitionUnit) -> Dict[str, str]:
    """
    Index the planes actually present on disk.

    Returns the full set of candidate files as ``{relative path: full path}`` so
    the CSV pass can confirm its rows without stat-ing each one.
    """
    on_disk: Dict[str, str] = {}

    for filename, plane_path in parsers.list_plane_files(fs, unit.path):
        relative = plane_path[len(unit.path) :].lstrip("/")
        on_disk[relative] = plane_path

        parsed = parsers.parse_plane_name(filename)
        if parsed is None:
            continue

        scene: SceneKey = (parsed.well, parsed.site)
        unit.planes.setdefault(scene, {}).setdefault(
            (parsed.t, parsed.channel, parsed.z), plane_path
        )

    return on_disk


def _apply_csv(
    unit: AcquisitionUnit,
    rows: List[CsvPlaneRow],
    on_disk: Dict[str, str],
) -> None:
    """
    Layer the manifest's metadata onto the walked index.

    Rows naming a file that is not on disk are skipped -- an aborted acquisition
    leaves a manifest describing planes that were never written. Rows naming a
    file the filename grammar could not parse are still indexed, so an unusual
    naming scheme degrades to "CSV only" rather than to nothing.
    """
    matched = False

    for row in rows:
        relative = posixpath.join(row.subfolder, row.filename)
        plane_path = on_disk.get(relative)
        if plane_path is None:
            continue

        matched = True
        scene: SceneKey = (row.well, row.site)
        unit.planes.setdefault(scene, {}).setdefault(
            (row.t, row.channel, row.z), plane_path
        )
        _apply_row(unit, scene, row)

    unit.used_csv = matched


def _apply_row(unit: AcquisitionUnit, scene: SceneKey, row: CsvPlaneRow) -> None:
    """Layer one manifest row's metadata onto the unit."""
    if row.timestamp_s is not None:
        # First plane of a time point stands in for the whole time point.
        existing = unit.timestamps.get(row.t)
        if existing is None or row.timestamp_s < existing:
            unit.timestamps[row.t] = row.timestamp_s

    if row.channel_name:
        unit.csv_channel_names.setdefault(row.channel, row.channel_name)

    if scene not in unit.positions and row.position_x_um is not None:
        unit.positions[scene] = (row.position_x_um, row.position_y_um)
