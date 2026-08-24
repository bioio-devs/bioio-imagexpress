#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
import json
import logging
import posixpath
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fsspec.spec import AbstractFileSystem

###############################################################################

log = logging.getLogger(__name__)

JDCE_EXTENSION = ".jdce"
TIMEPOINT_DIR_PREFIX = "timepoint"

# <Project>_t<T>_<Well>_s<Site>_w<Channel>_z<Z>.tif, all indices zero-based.
# Rows run A-Z then AA-AF, so a 1536-well plate has two-letter rows.
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)"
    r"_t(?P<t>\d+)"
    r"_(?P<well>[A-Z]{1,2}\d{2})"
    r"_s(?P<site>\d+)"
    r"_w(?P<channel>\d+)"
    r"_z(?P<z>\d+)"
    r"\.tiff?$",
    re.IGNORECASE,
)

# A normalized well label: row letters then an unpadded-or-padded column.
WELL_RE = re.compile(r"^([A-Z]{1,2})(\d+)$")

# (well, site) -- one acquisition position.
SceneKey = Tuple[str, int]
# (t, channel, z) -- one plane within an acquisition position.
PlaneKey = Tuple[int, int, int]

###############################################################################


@dataclass(frozen=True)
class PlaneName:
    """A decomposed plane filename."""

    t: int
    well: str
    site: int
    channel: int
    z: int


def parse_plane_name(name: str) -> Optional[PlaneName]:
    """
    Decompose a plane filename, or return None if it is not one.

    Parameters
    ----------
    name: str
        A file name, without any directory part.

    Returns
    -------
    plane: Optional[PlaneName]
        The plane's coordinates, or None for the sidecars and OS files that live
        beside the planes.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        return None

    return PlaneName(
        t=int(match.group("t")),
        well=match.group("well").upper(),
        site=int(match.group("site")),
        channel=int(match.group("channel")),
        z=int(match.group("z")),
    )


def normalize_well(value: str) -> Optional[str]:
    """
    Normalize a well label to the filenames' form, e.g. ``"B - 7"`` to ``"B07"``.

    Parameters
    ----------
    value: str
        A well label from either the manifest or a filename.

    Returns
    -------
    well: Optional[str]
        The normalized label, or None if the value is not a well.
    """
    match = re.match(
        r"^([A-Za-z]{1,2})-?(\d{1,2})$", (value or "").strip().replace(" ", "")
    )
    if match is None:
        return None

    return f"{match.group(1).upper()}{int(match.group(2)):02d}"


def split_well(well: str) -> Tuple[str, int]:
    """
    Split a normalized well label into its plate coordinates.

    Parameters
    ----------
    well: str
        A label as ``normalize_well`` or ``parse_plane_name`` produce it.

    Returns
    -------
    coordinates: Tuple[str, int]
        The row letters and the column number, e.g. ``("AA", 1)`` for ``"AA01"``.
    """
    match = WELL_RE.match(well)
    if match is None:
        raise ValueError(f"Not a well label: {well!r}")

    return match.group(1), int(match.group(2))


###############################################################################


@dataclass
class JdceMetadata:
    """The subset of the ``.jdce`` descriptor this reader consumes."""

    raw: Dict[str, Any] = field(default_factory=dict)
    channel_names: Dict[int, str] = field(default_factory=dict)
    pixel_size_x: Optional[float] = None
    pixel_size_y: Optional[float] = None
    z_step: Optional[float] = None
    objective: Optional[str] = None
    binning: Optional[str] = None
    operator: Optional[str] = None
    acquired_at: Optional[datetime] = None
    metadata_files: List[str] = field(default_factory=list)


def parse_jdce(contents: str) -> JdceMetadata:
    """
    Pull the acquisition parameters out of a ``.jdce`` descriptor.

    Parameters
    ----------
    contents: str
        The descriptor's text. It is JSON, despite the extension.

    Returns
    -------
    metadata: JdceMetadata
        Every field is optional; a sparse descriptor yields Nones rather than
        raising, so the read can fall back to the manifest and the filenames.
    """
    raw = json.loads(contents)
    stack = raw.get("ImageStack", {})
    protocol = stack.get("AutoLeadAcquisitionProtocol", {})
    calibration = protocol.get("ObjectiveCalibration", {})

    # Names are keyed by the descriptor's own Index, which is what the plane
    # filenames' _w<N> and the manifest's Wavelength column refer to. List
    # position stands in only when a wavelength carries no Index.
    wavelengths = sorted(
        protocol.get("Wavelengths", []) or [],
        key=lambda w: _as_int(w.get("Index")) or 0,
    )
    channel_names: Dict[int, str] = {}
    for position, wavelength in enumerate(wavelengths):
        name = (wavelength.get("EmissionFilter") or {}).get("Name")
        index = _as_int(wavelength.get("Index"))
        if name:
            channel_names.setdefault(index if index is not None else position, name)

    # Z spacing lives only here: the manifest's PositionZUm is absolute stage Z
    # and the TIFF resolution tags are unset. 0.0 and unparseable values are
    # both "no spacing"; keep scanning past them.
    z_step = next(
        (v for v in (_as_float(w.get("ZStep")) for w in wavelengths) if v), None
    )
    if not z_step:
        z_params = protocol.get("PlateMap", {}).get("ZDimensionParameters", {})
        z_step = _as_float(z_params.get("Step"))

    binning = protocol.get("Camera", {}).get("Binning")
    project_information = protocol.get("ProjectInformation", {})

    return JdceMetadata(
        raw=raw,
        channel_names=channel_names,
        pixel_size_x=_as_float(calibration.get("PixelWidth")),
        pixel_size_y=_as_float(calibration.get("PixelHeight")),
        z_step=z_step or None,
        objective=calibration.get("ObjectiveName"),
        # "1 X 1" becomes "1x1", the form other bioio readers report.
        binning=re.sub(r"\s*[xX]\s*", "x", str(binning).strip()) if binning else None,
        operator=(
            project_information.get("User", {}).get("Name")
            or stack.get("Operator", {}).get("Login")
            or None
        ),
        acquired_at=_parse_creation(stack.get("Creation", {})),
        # The descriptor names its own manifests, which is what lets a unit be
        # indexed without listing a single directory.
        metadata_files=[
            _as_relpath(str(n)) for n in (stack.get("ImageMetadataFiles") or []) if n
        ],
    )


def _parse_creation(creation: Dict[str, Any]) -> Optional[datetime]:
    # Naive on purpose: this is the instrument's local wall clock, and the
    # accompanying TimeZoneOffset does not describe it.
    date = (creation.get("Date") or "").strip()
    if not date:
        return None

    try:
        return datetime.fromisoformat(
            f"{date}T{(creation.get('Time') or '').strip() or '00:00:00'}"
        )
    except ValueError:
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_relpath(value: str) -> str:
    # The instrument runs Windows, so its path fields may arrive backslashed.
    return value.strip().replace("\\", "/").strip("/")


###############################################################################


@dataclass(frozen=True)
class ManifestRow:
    """One row of ``image_metadata_*.csv``, reduced to what the index needs."""

    well: str
    site: int
    channel: int
    t: int
    z: int
    subfolder: str
    filename: str
    timestamp_s: Optional[float]
    position_x_um: Optional[float]
    position_y_um: Optional[float]


def read_manifest(contents: str) -> List[ManifestRow]:
    """
    Parse an ``image_metadata_*.csv`` manifest.

    Parameters
    ----------
    contents: str
        The manifest's text.

    Returns
    -------
    rows: List[ManifestRow]
        One row per plane. Rows that do not resolve to a full (well, site,
        channel, t, z) coordinate are dropped. A manifest that yields no rows at
        all sends ``build_unit`` to the filename walk instead.
    """
    rows = []

    for record in csv.DictReader(io.StringIO(contents)):
        well = normalize_well(record.get("Well") or "")
        site = _as_int(record.get("Field"))
        channel = _as_int(record.get("Wavelength"))
        t = _as_int(record.get("Timepoint"))
        z = _as_int(record.get("ZIndex"))
        filename = (record.get("ImageFileName") or "").strip()

        if not filename or None in (well, site, channel, t, z):
            continue

        rows.append(
            ManifestRow(
                well=str(well),
                site=int(site),  # type: ignore[arg-type]
                channel=int(channel),  # type: ignore[arg-type]
                t=int(t),  # type: ignore[arg-type]
                z=int(z),  # type: ignore[arg-type]
                subfolder=_as_relpath(record.get("ImageSubFolderPath") or "")
                or f"{TIMEPOINT_DIR_PREFIX}{t}",
                filename=_as_relpath(filename),
                timestamp_s=_as_float(record.get("TimeStampSec")),
                position_x_um=_as_float(record.get("PositionXUm")),
                position_y_um=_as_float(record.get("PositionYUm")),
            )
        )

    return rows


###############################################################################


@dataclass
class AcquisitionUnit:
    """One acquisition: its descriptor, its plane index, and its stage metadata."""

    path: str
    jdce: JdceMetadata
    planes: Dict[SceneKey, Dict[PlaneKey, str]] = field(default_factory=dict)
    timestamps: Dict[SceneKey, Dict[int, float]] = field(default_factory=dict)
    positions: Dict[SceneKey, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )

    @property
    def scene_keys(self) -> List[SceneKey]:
        """Every (well, site) present, ordered by plate row, column, then site."""

        def order(key: SceneKey) -> Tuple[int, str, int, int]:
            row, column = split_well(key[0])
            # Single-letter rows precede the double-letter rows below them.
            return len(row), row, column, key[1]

        return sorted(self.planes, key=order)

    @property
    def wells(self) -> List[str]:
        """The wells imaged, in plate order."""
        return list(dict.fromkeys(well for well, _ in self.scene_keys))

    def sites(self, well: str) -> List[int]:
        """The acquisition positions imaged in ``well``, in order."""
        return sorted(site for name, site in self.planes if name == well)

    def extents(
        self, keys: Sequence[SceneKey]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        The T, C and Z indices present across ``keys``.

        Takes several keys because a mosaic scene spans every tile of a well, and
        a tile that dropped a time point must not shrink the others. Read off the
        index rather than the descriptor, which promises planes an aborted
        acquisition never wrote.
        """
        planes: List[PlaneKey] = []
        for key in keys:
            planes.extend(self.planes[key])

        return (
            sorted({plane[0] for plane in planes}),
            sorted({plane[1] for plane in planes}),
            sorted({plane[2] for plane in planes}),
        )

    def channel_names(self, keys: Sequence[SceneKey]) -> List[str]:
        """The descriptor's emission filter names for the channels ``keys`` cover."""
        _, channels, _ = self.extents(keys)
        names = self.jdce.channel_names

        return [names.get(channel, f"Channel:{channel}") for channel in channels]

    def time_coords(self, keys: Sequence[SceneKey]) -> Optional[List[float]]:
        """
        Elapsed seconds from the first time point of the scene ``keys`` span.

        Measured from the manifest's timestamps for these keys alone, because
        wells are imaged in sequence and another well's clock says nothing about
        this one. The descriptor's ``TimeSchedule.Times[].Ms`` values are time
        point indices rather than milliseconds, so they are deliberately not used.
        """
        stamps: Dict[int, float] = {}
        for key in keys:
            for t, seconds in self.timestamps.get(key, {}).items():
                if t not in stamps or seconds < stamps[t]:
                    stamps[t] = seconds

        timepoints, _, _ = self.extents(keys)
        if not all(t in stamps for t in timepoints):
            return None

        origin = min(stamps.values())

        return [stamps[t] - origin for t in timepoints]

    @property
    def first_timestamp(self) -> Optional[float]:
        """The acquisition's earliest manifest timestamp, or None without one."""
        stamps = [
            s for per_scene in self.timestamps.values() for s in per_scene.values()
        ]

        return min(stamps) if stamps else None


###############################################################################


def discover_unit(
    fs: AbstractFileSystem, path: str
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Resolve a user-supplied path to a single acquisition unit.

    Parameters
    ----------
    fs: AbstractFileSystem
        The filesystem ``path`` lives on.
    path: str
        A ``.jdce`` descriptor or an acquisition directory.

    Returns
    -------
    unit: Optional[Tuple[str, Optional[str]]]
        The unit's directory and its descriptor, or None if the path is not an
        acquisition. Naming the descriptor is accepted on its existence alone,
        without the ``timepoint<N>`` check the directory form makes, because that
        check costs a directory listing.
    """
    path = path.rstrip("/")

    if path.lower().endswith(JDCE_EXTENSION) and not _isdir(fs, path):
        return (posixpath.dirname(path), path) if _exists(fs, path) else None

    descriptors = find_descriptors(fs, path)
    if descriptors and find_timepoint_dirs(fs, path):
        if len(descriptors) > 1:
            log.warning(
                "%s holds %d descriptors; reading %s. Name a descriptor directly "
                "to read another.",
                path,
                len(descriptors),
                descriptors[0],
            )
        return path, descriptors[0]

    return None


def find_sub_units(fs: AbstractFileSystem, path: str) -> List[str]:
    """The names of any acquisition units one level below ``path``."""
    return [
        name
        for name in sorted(_names_in(fs, path))
        if discover_unit(fs, posixpath.join(path, name)) is not None
    ]


def build_unit(
    fs: AbstractFileSystem, path: str, descriptor: Optional[str]
) -> AcquisitionUnit:
    """
    Index one acquisition unit, opening no TIFF.

    Parameters
    ----------
    fs: AbstractFileSystem
        The filesystem the unit lives on.
    path: str
        The unit's directory.
    descriptor: Optional[str]
        The unit's ``.jdce`` file.

    Returns
    -------
    unit: AcquisitionUnit
        Indexed from the ``image_metadata_*.csv`` manifests, which name every
        plane's subfolder and filename, so the whole unit resolves without
        listing a directory. Falls back to walking ``timepoint<N>`` and parsing
        the filename grammar when no manifest can be read.
    """
    unit = AcquisitionUnit(path=path, jdce=_read_jdce(fs, descriptor))

    rows: List[ManifestRow] = []
    for manifest in _manifest_paths(fs, path, unit.jdce):
        try:
            with fs.open(manifest, "r", encoding="utf-8-sig") as handle:
                rows.extend(read_manifest(handle.read()))
        except Exception as exc:
            log.warning("Could not read manifest %s: %s", manifest, exc)

    if rows:
        _index_from_manifest(unit, rows)
    else:
        _index_from_walk(fs, unit)

    return unit


def find_descriptors(fs: AbstractFileSystem, path: str) -> List[str]:
    """The ``.jdce`` files directly inside ``path``, sorted for determinism."""
    return sorted(
        posixpath.join(path, name)
        for name in _names_in(fs, path)
        if name.lower().endswith(JDCE_EXTENSION)
    )


def find_timepoint_dirs(fs: AbstractFileSystem, path: str) -> List[str]:
    """The ``timepoint<N>`` directories inside ``path``, ordered by N."""
    found = []
    for name in _names_in(fs, path):
        if not name.lower().startswith(TIMEPOINT_DIR_PREFIX):
            continue
        index = _as_int(name[len(TIMEPOINT_DIR_PREFIX) :])
        if index is not None:
            found.append((index, posixpath.join(path, name)))

    return [path for _, path in sorted(found)]


def _manifest_paths(fs: AbstractFileSystem, path: str, jdce: JdceMetadata) -> List[str]:
    # The descriptor names its own manifests; only fall back to listing when it
    # does not, or could not be read.
    if jdce.metadata_files:
        return [posixpath.join(path, name) for name in jdce.metadata_files]

    return sorted(
        posixpath.join(path, name)
        for name in _names_in(fs, path)
        if name.lower().startswith("image_metadata_") and name.lower().endswith(".csv")
    )


def _read_jdce(fs: AbstractFileSystem, descriptor: Optional[str]) -> JdceMetadata:
    if descriptor is None:
        return JdceMetadata()

    try:
        with fs.open(descriptor, "r", encoding="utf-8-sig") as handle:
            return parse_jdce(handle.read())
    except Exception as exc:
        # A malformed descriptor costs pixel sizes and channel names, not the read.
        log.warning("Could not parse descriptor %s: %s", descriptor, exc)
        return JdceMetadata()


def _index_from_manifest(unit: AcquisitionUnit, rows: List[ManifestRow]) -> None:
    # Plane paths are composed rather than confirmed; a row naming a file that
    # was never written surfaces as a read error when that plane is asked for.
    for row in rows:
        scene: SceneKey = (row.well, row.site)
        unit.planes.setdefault(scene, {}).setdefault(
            (row.t, row.channel, row.z),
            posixpath.join(unit.path, row.subfolder, row.filename),
        )

        if row.timestamp_s is not None:
            # The scene's first plane of a time point stands in for the whole
            # time point; other scenes were imaged at other times.
            stamps = unit.timestamps.setdefault(scene, {})
            recorded = stamps.get(row.t)
            if recorded is None or row.timestamp_s < recorded:
                stamps[row.t] = row.timestamp_s

        if scene not in unit.positions and row.position_x_um is not None:
            unit.positions[scene] = (row.position_x_um, row.position_y_um)


def _index_from_walk(fs: AbstractFileSystem, unit: AcquisitionUnit) -> None:
    for timepoint_dir in find_timepoint_dirs(fs, unit.path):
        for name in sorted(_names_in(fs, timepoint_dir)):
            plane = parse_plane_name(name)
            if plane is None:
                continue

            unit.planes.setdefault((plane.well, plane.site), {}).setdefault(
                (plane.t, plane.channel, plane.z),
                posixpath.join(timepoint_dir, name),
            )


def _names_in(fs: AbstractFileSystem, path: str) -> List[str]:
    # "Cannot list" is a supported state here rather than a failure, and backends
    # signal it with everything from OSError to aiohttp's own exceptions.
    try:
        return [posixpath.basename(p.rstrip("/")) for p in fs.ls(path, detail=False)]
    except Exception as exc:
        log.debug("Could not list %s: %s", path, exc)
        return []


def _isdir(fs: AbstractFileSystem, path: str) -> bool:
    try:
        return bool(fs.isdir(path))
    except Exception:
        return False


def _exists(fs: AbstractFileSystem, path: str) -> bool:
    try:
        return bool(fs.exists(path))
    except Exception:
        return False
