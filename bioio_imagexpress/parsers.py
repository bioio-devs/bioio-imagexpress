#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pure parsing helpers for Molecular Devices ImageXpress / MetaXpress acquisitions.

Nothing in here touches the Reader class, which keeps the fiddly bits (filename
grammar, ``.jdce`` shape, CSV column names) independently testable.
"""

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

###############################################################################

# Descriptor written once per acquisition unit by MetaXpress 2026+. JSON despite
# the unfamiliar extension.
JDCE_EXTENSION = ".jdce"

# Per-plane manifest sitting next to the descriptor. Numbered because very large
# acquisitions are split across several files.
IMAGE_METADATA_GLOB = "image_metadata_*.csv"

# Plane directories, one per acquisition time point. Zero-based, unlike the
# ``TimePoint_1`` folders used by the older .HTD-based exports.
TIMEPOINT_DIR_PREFIX = "timepoint"

# Written alongside the planes by MetaXpress / Windows. Never image data.
SIDECAR_SUFFIX = ".statistics.json"
IGNORED_NAMES = frozenset({"thumbs.db", ".ds_store"})

# <Project>_t<T>_<Well>_s<Site>_w<Channel>_z<Z>.tif
#
# Verified against every plane of both reference acquisitions (26k+ files, zero
# misses). All indices are zero-based, and the ``t`` token always agrees with the
# containing ``timepoint<N>`` directory.
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)"
    r"_t(?P<t>\d+)"
    r"_(?P<well>[A-Z]\d{2})"
    r"_s(?P<site>\d+)"
    r"_w(?P<channel>\d+)"
    r"_z(?P<z>\d+)"
    r"\.tif{1,2}$",
    re.IGNORECASE,
)

###############################################################################


@dataclass(frozen=True)
class PlaneName:
    """A decomposed plane filename."""

    prefix: str
    t: int
    well: str
    site: int
    channel: int
    z: int


def parse_plane_name(name: str) -> Optional[PlaneName]:
    """
    Decompose a plane filename, or return None if it is not one.

    Returning None rather than raising is deliberate: acquisition directories are
    full of sidecars and OS detritus, and "not a plane" is the common case, not an
    error.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        return None

    return PlaneName(
        prefix=match.group("prefix"),
        t=int(match.group("t")),
        well=match.group("well").upper(),
        site=int(match.group("site")),
        channel=int(match.group("channel")),
        z=int(match.group("z")),
    )


def is_ignorable(name: str) -> bool:
    """True for sidecars and OS junk that live beside the planes."""
    return name.lower() in IGNORED_NAMES or name.endswith(SIDECAR_SUFFIX)


###############################################################################


@dataclass
class JdceMetadata:
    """The subset of the ``.jdce`` descriptor this reader consumes."""

    raw: Dict[str, Any] = field(default_factory=dict)
    channel_names: List[str] = field(default_factory=list)
    pixel_size_x: Optional[float] = None
    pixel_size_y: Optional[float] = None
    z_step: Optional[float] = None
    objective: Optional[str] = None
    plate_id: Optional[str] = None
    barcode: Optional[str] = None
    project: Optional[str] = None
    binning: Optional[str] = None
    operator: Optional[str] = None
    acquired_at: Optional[datetime] = None
    metadata_files: List[str] = field(default_factory=list)


def parse_jdce(contents: str) -> JdceMetadata:
    """
    Pull the acquisition parameters out of a ``.jdce`` descriptor.

    Every field is optional. A descriptor that is present but sparse should
    degrade to "unknown" rather than break the read, so callers can fall back to
    the CSV manifest or the TIFF tags.
    """
    raw = json.loads(contents)
    stack = raw.get("ImageStack", {})
    protocol = stack.get("AutoLeadAcquisitionProtocol", {})

    calibration = protocol.get("ObjectiveCalibration", {})
    wavelengths = protocol.get("Wavelengths", []) or []

    # Channel names come from the emission filter -- "TL", "FITC", etc. Keep the
    # descriptor's own ordering by Index rather than list position.
    ordered = sorted(wavelengths, key=lambda w: w.get("Index", 0))
    channel_names = []
    for i, wavelength in enumerate(ordered):
        emission = wavelength.get("EmissionFilter", {}) or {}
        channel_names.append(emission.get("Name") or f"Channel:{i}")

    # Z spacing lives only here -- the CSV's PositionZUm is absolute stage Z, and
    # the TIFF resolution tags are unset. Prefer the per-wavelength ZStep and fall
    # back to the plate-level Z parameters.
    z_step = None
    for wavelength in ordered:
        step = wavelength.get("ZStep")
        if step:
            z_step = float(step)
            break
    if not z_step:
        z_params = protocol.get("PlateMap", {}).get("ZDimensionParameters", {})
        if z_params.get("Step"):
            z_step = float(z_params["Step"])

    project_info = protocol.get("ProjectInformation", {})
    project = project_info.get("Project", {}).get("Name")
    holder = stack.get("SpecimenHolder", {})

    # The protocol's own user is the person who set the acquisition running; the
    # station login is the same account on every instrument we have seen, so it
    # is only a fallback.
    operator = (
        project_info.get("User", {}).get("Name")
        or stack.get("Operator", {}).get("Login")
        or None
    )

    return JdceMetadata(
        raw=raw,
        channel_names=channel_names,
        pixel_size_x=_as_float(calibration.get("PixelWidth")),
        pixel_size_y=_as_float(calibration.get("PixelHeight")),
        z_step=z_step,
        objective=calibration.get("ObjectiveName"),
        plate_id=stack.get("PlateId"),
        barcode=holder.get("Barcode") or None,
        project=project,
        binning=_parse_binning(protocol.get("Camera", {}).get("Binning")),
        operator=operator,
        acquired_at=_parse_creation(stack.get("Creation", {})),
        # The descriptor names its own manifests, which is what lets an
        # acquisition be indexed without a single directory listing.
        metadata_files=[
            str(name) for name in (stack.get("ImageMetadataFiles") or []) if name
        ],
    )


def _parse_binning(value: Any) -> Optional[str]:
    """
    Normalize the descriptor's ``"1 X 1"`` to the ``"1x1"`` other bioio readers
    report, so a binning comparison across plugins is a string comparison.
    """
    if not value:
        return None

    text = re.sub(r"\s*[xX]\s*", "x", str(value).strip())

    return text or None


def _parse_creation(creation: Dict[str, Any]) -> Optional[datetime]:
    """
    Combine the descriptor's ``Creation`` date and time into a datetime.

    Deliberately naive. The wall clock recorded here is the instrument's local
    time, and the accompanying ``TimeZoneOffset`` does not describe it -- both
    reference acquisitions were taken at UTC-7 and report an offset of 0. Tagging
    the result with that offset would turn an accurate local time into an
    inaccurate absolute one, so the offset is ignored.
    """
    date = (creation.get("Date") or "").strip()
    if not date:
        return None

    time = (creation.get("Time") or "").strip() or "00:00:00"

    try:
        return datetime.fromisoformat(f"{date}T{time}")
    except ValueError:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


###############################################################################


@dataclass(frozen=True)
class CsvPlaneRow:
    """One row of ``image_metadata_*.csv``, reduced to what the reader needs."""

    well: str
    site: int
    channel: int
    t: int
    z: int
    subfolder: str
    filename: str
    timestamp_s: Optional[float]
    channel_name: Optional[str]
    position_x_um: Optional[float]
    position_y_um: Optional[float]
    position_z_um: Optional[float]


def normalize_well(value: str) -> Optional[str]:
    """
    Normalize the CSV's ``"B - 7"`` well label to the filename's ``"B07"``.

    Already-normalized values pass through, so this is safe to apply to either
    source.
    """
    if value is None:
        return None

    text = value.strip().replace(" ", "")
    match = re.match(r"^([A-Za-z])-?(\d{1,2})$", text)
    if match is None:
        return None

    return f"{match.group(1).upper()}{int(match.group(2)):02d}"


def read_image_metadata_csv(contents: str) -> List[CsvPlaneRow]:
    """
    Parse an ``image_metadata_*.csv`` manifest.

    Rows that cannot be reduced to a full (well, site, channel, t, z) coordinate
    are dropped -- the filename walk is the fallback for anything this misses.
    """
    rows: List[CsvPlaneRow] = []

    for record in csv.DictReader(io.StringIO(contents)):
        filename = (record.get("ImageFileName") or "").strip()
        if not filename:
            continue

        well = normalize_well(record.get("Well") or "")
        site = _as_int(record.get("Field"))
        channel = _as_int(record.get("Wavelength"))
        t = _as_int(record.get("Timepoint"))
        z = _as_int(record.get("ZIndex"))

        # Fall back to the filename for any coordinate the CSV is missing. The two
        # agree on every row of both reference datasets, so this only matters for
        # damaged or hand-edited manifests.
        if None in (well, site, channel, t, z):
            parsed = parse_plane_name(filename)
            if parsed is None:
                continue
            well = well or parsed.well
            site = parsed.site if site is None else site
            channel = parsed.channel if channel is None else channel
            t = parsed.t if t is None else t
            z = parsed.z if z is None else z

        subfolder = (record.get("ImageSubFolderPath") or "").strip()
        if not subfolder:
            subfolder = f"{TIMEPOINT_DIR_PREFIX}{t}"

        rows.append(
            CsvPlaneRow(
                well=str(well),
                site=int(site),  # type: ignore[arg-type]
                channel=int(channel),  # type: ignore[arg-type]
                t=int(t),  # type: ignore[arg-type]
                z=int(z),  # type: ignore[arg-type]
                subfolder=subfolder,
                filename=filename,
                timestamp_s=_as_float(record.get("TimeStampSec")),
                channel_name=(record.get("ExcitationEmissionFilter") or "").strip()
                or None,
                position_x_um=_as_float(record.get("PositionXUm")),
                position_y_um=_as_float(record.get("PositionYUm")),
                position_z_um=_as_float(record.get("PositionZUm")),
            )
        )

    return rows


def read_image_metadata_records(contents: str) -> List[Dict[str, Optional[str]]]:
    """
    The manifest as JSON-able records -- one dict per row, every column kept.

    ``read_image_metadata_csv`` reduces each row to the handful of fields the index
    needs. This keeps the rest, which is where the per-plane acquisition record
    lives: exposure, intensity statistics, incubation temperature, CO2 and O2,
    field offsets, the FOV uuid.

    Values are left as written. Coercing them to numbers would be friendlier to
    read but is not safe in general -- a checksum or a zero-padded identifier does
    not survive ``int()`` -- so the only normalizing done here is trimming
    whitespace and mapping an empty cell to ``None``.
    """
    records: List[Dict[str, Optional[str]]] = []

    for record in csv.DictReader(io.StringIO(contents)):
        records.append(
            {
                str(key).strip(): _clean_cell(value)
                for key, value in record.items()
                if key is not None
            }
        )

    return records


def _clean_cell(value: Any) -> Optional[str]:
    if isinstance(value, list):  # csv restval for a row with extra columns
        value = ",".join(str(part) for part in value)
    if value is None:
        return None

    return str(value).strip() or None


###############################################################################
# Acquisition unit discovery


def _names_in(fs: AbstractFileSystem, path: str) -> List[str]:
    """
    Entry names directly inside ``path``, or nothing when it cannot be listed.

    Every backend refuses differently and not all of the refusals are
    ``OSError``: s3fs turns a denied ``ListBucket`` into ``PermissionError``,
    while fsspec's HTTP backend surfaces ``aiohttp.ClientResponseError``, which
    descends from ``Exception`` alone. Since "cannot list" is a supported state
    here rather than a failure -- the manifest-driven index needs no listing --
    the catch is deliberately broad.
    """
    try:
        return [posixpath.basename(p.rstrip("/")) for p in fs.ls(path, detail=False)]
    except Exception as exc:
        log.debug("Could not list %s: %s", path, exc)
        return []


def _is_dir(fs: AbstractFileSystem, path: str) -> bool:
    """``fs.isdir`` that answers False rather than raising when listing is denied."""
    try:
        return bool(fs.isdir(path))
    except Exception as exc:
        log.debug("Could not stat %s: %s", path, exc)
        return False


def _exists(fs: AbstractFileSystem, path: str) -> bool:
    """``fs.exists`` that answers False rather than raising."""
    try:
        return bool(fs.exists(path))
    except Exception as exc:
        log.debug("Could not check %s: %s", path, exc)
        return False


def find_jdce_files(fs: AbstractFileSystem, path: str) -> List[str]:
    """Descriptor files directly inside ``path``, sorted for determinism."""
    return sorted(
        posixpath.join(path, name)
        for name in _names_in(fs, path)
        if name.lower().endswith(JDCE_EXTENSION)
    )


def find_timepoint_dirs(fs: AbstractFileSystem, path: str) -> List[str]:
    """
    ``timepoint<N>`` directories inside ``path``, ordered by N.

    Ordered numerically rather than lexically so ``timepoint10`` sorts after
    ``timepoint9``.
    """
    found = []
    for name in _names_in(fs, path):
        if not name.lower().startswith(TIMEPOINT_DIR_PREFIX):
            continue
        index = _as_int(name[len(TIMEPOINT_DIR_PREFIX) :])
        if index is None:
            continue
        found.append((index, posixpath.join(path, name)))

    return [p for _, p in sorted(found)]


def is_acquisition_unit(fs: AbstractFileSystem, path: str) -> bool:
    """
    True when ``path`` is a single acquisition: a descriptor plus plane folders.

    Both are required. A descriptor on its own is a protocol definition, and bare
    ``timepoint`` folders could belong to any number of other tools.
    """
    if not _is_dir(fs, path):
        return False

    return bool(find_jdce_files(fs, path)) and bool(find_timepoint_dirs(fs, path))


@dataclass(frozen=True)
class DiscoveredUnit:
    """
    One acquisition unit resolved from a user-supplied path.

    ``name`` is empty when the path named a single unit directly, which is what
    keeps scene ids unprefixed in the common case; it is the directory name when
    several units were found under a run root, so that scenes stay
    distinguishable.

    ``descriptor`` is carried along when the path named a ``.jdce`` file, because
    that is the one entry point that needs no directory listing -- knowing the
    descriptor's name means the whole unit can be indexed from it and the
    manifests it points at.
    """

    name: str
    path: str
    descriptor: Optional[str] = None


def discover_units(fs: AbstractFileSystem, path: str) -> List[DiscoveredUnit]:
    """
    Resolve a user-supplied path to the acquisition units it contains.

    Accepts a unit directory, a ``.jdce`` file (resolving to its parent), or a run
    root containing unit sub-directories.

    Naming the ``.jdce`` is deliberately the least demanding form: it is accepted
    on the descriptor's existence alone, without the ``timepoint<N>`` check the
    directory forms make, because that check costs a directory listing. Read-only
    HTTP endpoints commonly serve files but refuse to list, and this is the entry
    point that works there.
    """
    path = path.rstrip("/")

    # A descriptor file identifies its own directory.
    if path.lower().endswith(JDCE_EXTENSION) and not _is_dir(fs, path):
        if not _exists(fs, path):
            return []
        return [DiscoveredUnit("", posixpath.dirname(path), path)]

    if not _is_dir(fs, path):
        return []

    if is_acquisition_unit(fs, path):
        return [DiscoveredUnit("", path)]

    # Otherwise treat it as a run root and collect the units one level down.
    units = []
    for name in sorted(_names_in(fs, path)):
        candidate = posixpath.join(path, name)
        if is_acquisition_unit(fs, candidate):
            units.append(DiscoveredUnit(name, candidate))

    return units


def resolve_metadata_csvs(
    fs: AbstractFileSystem, unit_path: str, jdce: JdceMetadata
) -> List[str]:
    """
    The unit's manifests, preferring the names the descriptor already carries.

    ``ImageMetadataFiles`` names every manifest, so taking it at its word turns
    indexing into "read two files" instead of "walk every ``timepoint<N>``
    directory" -- the difference between 0.2 s and 8 s on the 23,760 plane
    reference acquisition, and the difference between working and not working on
    a filesystem that cannot list.
    """
    if jdce.metadata_files:
        return [posixpath.join(unit_path, name) for name in jdce.metadata_files]

    return find_metadata_csvs(fs, unit_path)


def list_plane_files(fs: AbstractFileSystem, unit_path: str) -> List[Tuple[str, str]]:
    """
    Walk a unit's ``timepoint<N>`` folders and return ``(filename, full_path)``.

    Used when no CSV manifest is available, and to validate the one that is.
    """
    planes = []
    for timepoint_dir in find_timepoint_dirs(fs, unit_path):
        for name in sorted(_names_in(fs, timepoint_dir)):
            if is_ignorable(name):
                continue
            planes.append((name, posixpath.join(timepoint_dir, name)))

    return planes


def find_metadata_csvs(fs: AbstractFileSystem, unit_path: str) -> List[str]:
    """``image_metadata_*.csv`` manifests inside a unit, sorted by name."""
    return sorted(
        posixpath.join(unit_path, name)
        for name in _names_in(fs, unit_path)
        if name.lower().startswith("image_metadata_") and name.lower().endswith(".csv")
    )


def unique_sorted(values: Sequence[int]) -> List[int]:
    return sorted(set(values))
