#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synthetic ImageXpress acquisitions for testing.

The real acquisitions these were modelled on are 6 GB and 306 GB, so the tests
build miniature ones that keep every structural feature that matters -- the
descriptor, the manifest, the ``timepoint<N>`` layout, the filename grammar, the
embedded pyramid, and the sidecar junk -- at 32x32 pixels.
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest
import tifffile

###############################################################################

# Kept small enough that a full fixture acquisition is a few hundred KB.
PLANE_SIZE = 32
PYRAMID_LEVELS = 3

DEFAULT_WELLS = ["B02", "B03"]
DEFAULT_SITES = [0, 1]
DEFAULT_CHANNELS = ["TL", "FITC"]

CSV_COLUMNS = [
    "Well",
    "Row",
    "Column",
    "Field",
    "Wavelength",
    "Timepoint",
    "ZIndex",
    "ImageSubFolderPath",
    "ImageFileName",
    "TimeStampSec",
    "ExposureTimeMs",
    "ExcitationEmissionFilter",
    "PositionXUm",
    "PositionYUm",
    "PositionZUm",
]

# Seconds between time points, matching the 6-hourly cadence of the reference
# z-stack acquisition.
TIME_INTERVAL_S = 21600.0
EPOCH = 1785864670.0

# The descriptor's own acquisition stamp. Instrument local wall clock, which is
# deliberately not the UTC rendering of EPOCH -- the reader must report this one.
CREATION_DATE = "2026-08-04"
CREATION_TIME = "10:31:10"

# Sites are mosaic tiles: MetaXpress images a well as a grid of overlapping
# fields, serpentine from the top-left. Verified against both reference
# acquisitions, where `experiment_montage` is the instrument's own stitch of
# exactly these. (column, row) of each site, and the 10% overlap it uses.
SITE_GRID = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
TILE_OVERLAP = 0.1

# Stage coordinates of the top-left tile. Arbitrary, but far from the origin so a
# test cannot pass by treating absolute stage position as a pixel offset.
STAGE_ORIGIN_X = 21579.44
STAGE_ORIGIN_Y = 18639.46


def tile_spacing(plane_size: int, pixel_size: float) -> float:
    """Stage distance between neighbouring tile centres, in microns."""
    return plane_size * pixel_size * (1.0 - TILE_OVERLAP)


def tile_stage_position(
    site: int, plane_size: int, pixel_size: float
) -> Tuple[float, float]:
    """Stage X and Y of one tile, in microns."""
    column, row = SITE_GRID[site]
    spacing = tile_spacing(plane_size, pixel_size)

    return (STAGE_ORIGIN_X + column * spacing, STAGE_ORIGIN_Y + row * spacing)


def tile_pixel_offset(site: int, plane_size: int) -> Tuple[int, int]:
    """Where `tile_stage_position` puts a tile's top-left pixel, as (top, left)."""
    column, row = SITE_GRID[site]
    step = round(plane_size * (1.0 - TILE_OVERLAP))

    return (row * step, column * step)


###############################################################################


# Bit widths for packing a plane coordinate into a uint16 sample value. The sum
# must stay below 16 so that the +1 offset (which keeps every real plane distinct
# from a zero-filled missing one) cannot overflow.
_FIELD_WIDTHS = [
    ("row", 3),
    ("column", 4),
    ("site", 2),  # a well is imaged as up to a 2x2 grid of mosaic tiles
    ("t", 2),
    ("channel", 1),
    ("z", 3),
]


def plane_value(well: str, site: int, t: int, channel: int, z: int) -> int:
    """
    A value unique to each plane coordinate, packed into a uint16 sample.

    Lets a test assert that a given position in the assembled array really came
    from the file it should have -- a transposed stack would otherwise still have
    the right shape. Never zero, so a zero-filled missing plane is unambiguous.
    """
    fields = {
        "row": ord(well[0]) - 65,
        "column": int(well[1:]),
        "site": site,
        "t": t,
        "channel": channel,
        "z": z,
    }

    packed = 0
    for name, width in _FIELD_WIDTHS:
        value = fields[name]
        if not 0 <= value < (1 << width):
            raise ValueError(
                f"Fixture {name}={value} does not fit in {width} bits; widen "
                "_FIELD_WIDTHS (keeping the total under 16) to support it."
            )
        packed = (packed << width) | value

    return packed + 1


def write_plane(path: Path, value: int, size: int = PLANE_SIZE) -> None:
    """Write one plane as a MetaSeries-style pyramidal TIFF."""
    data = np.full((size, size), value, dtype=np.uint16)

    description = (
        "<MetaData>\n"
        '<prop id="ApplicationName" type="string" value="MetaXpress Acquire"/>\n'
        "<PlaneInfo>\n"
        '<prop id="spatial-calibration-x" type="float" value="0.5817"/>\n'
        '<prop id="spatial-calibration-y" type="float" value="0.5817"/>\n'
        f'<prop id="image-name" type="string" value="{path.name}"/>\n'
        "</PlaneInfo>\n"
        "</MetaData>"
    )

    with tifffile.TiffWriter(path) as writer:
        writer.write(
            data,
            subifds=PYRAMID_LEVELS - 1,
            description=description,
            software="MetaSeries",
        )
        for level in range(1, PYRAMID_LEVELS):
            step = 2**level
            writer.write(data[::step, ::step], subfiletype=1)


def build_jdce(
    channels: Sequence[str],
    z_count: int,
    t_count: int,
    z_step: float = 3.0,
    pixel_size: float = 0.5817,
    project: str = "Fixture",
) -> dict:
    return {
        "Version": "1.1",
        "ImageStack": {
            "Uuid": "00000000-0000-0000-0000-000000000000",
            "Application": {"Name": "MetaXpress Acquire"},
            "Creation": {
                "Date": CREATION_DATE,
                "Time": CREATION_TIME,
                # Zero even on an instrument running at UTC-7, as MetaXpress
                # writes it -- the reader must not apply it.
                "TimeZoneOffset": 0,
            },
            "ImageFormat": "TIFF",
            "PlateId": "3500000000",
            "Operator": {"Login": "moldev"},
            "AutoLeadAcquisitionProtocol": {
                "Camera": {
                    "Size": {"Width": PLANE_SIZE, "Height": PLANE_SIZE},
                    "Binning": "1 X 1",
                },
                "ObjectiveCalibration": {
                    "Unit": "µm",
                    "ObjectiveName": "10X Plan Apo Lambda D",
                    "PixelWidth": pixel_size,
                    "PixelHeight": pixel_size,
                },
                "Plate": {"Name": "Fixture 96 Well", "Rows": 8, "Columns": 12},
                "Wavelengths": [
                    {
                        "Index": i,
                        "ZSlice": z_count,
                        "ZStep": z_step,
                        "EmissionFilter": {"Name": name},
                    }
                    for i, name in enumerate(channels)
                ],
                "PlateMap": {
                    "ZDimensionParameters": {
                        "Enabled": z_count > 1,
                        "Step": z_step,
                        "NumberOfSlices": z_count,
                    },
                    "TimeSchedule": {
                        "Enabled": t_count > 1,
                        "NumberOfTimepoints": t_count,
                        # Deliberately index-like, as MetaXpress writes them --
                        # the reader must not read these as milliseconds.
                        "Times": [{"Ms": i} for i in range(t_count)],
                    },
                },
                "ProjectInformation": {
                    "Project": {"Name": project},
                    "User": {"Name": "fixture_operator"},
                },
            },
            "SpecimenHolder": {"Barcode": "3500000000"},
            "ImageMetadataFiles": ["image_metadata_1.csv"],
        },
    }


def make_acquisition_unit(
    root: Path,
    *,
    project: str = "Fixture",
    wells: Sequence[str] = DEFAULT_WELLS,
    sites: Sequence[int] = DEFAULT_SITES,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    t_count: int = 2,
    z_count: int = 3,
    pixel_size: float = 0.5817,
    z_step: float = 3.0,
    write_csv: bool = True,
    write_sidecars: bool = True,
    skip_planes: Sequence[Tuple[str, int, int, int, int]] = (),
    plane_size: int = PLANE_SIZE,
) -> Path:
    """
    Write one acquisition unit and return its path.

    ``skip_planes`` omits ``(well, site, t, channel, z)`` planes so ragged
    acquisitions can be exercised.
    """
    root.mkdir(parents=True, exist_ok=True)
    skip = set(skip_planes)

    (root / f"{project}.jdce").write_text(
        json.dumps(
            build_jdce(
                channels,
                z_count,
                t_count,
                z_step=z_step,
                pixel_size=pixel_size,
                project=project,
            ),
            indent=2,
        )
    )

    csv_rows: List[Dict[str, object]] = []

    for t in range(t_count):
        timepoint_dir = root / f"timepoint{t}"
        timepoint_dir.mkdir(exist_ok=True)

        for well in wells:
            for site in sites:
                position_x, position_y = tile_stage_position(
                    site, plane_size, pixel_size
                )
                for channel in range(len(channels)):
                    for z in range(z_count):
                        name = f"{project}_t{t}_{well}_s{site}_w{channel}_z{z}.tif"
                        if (well, site, t, channel, z) in skip:
                            continue

                        write_plane(
                            timepoint_dir / name,
                            plane_value(well, site, t, channel, z),
                            size=plane_size,
                        )

                        if write_sidecars:
                            (timepoint_dir / f"{name}.statistics.json").write_text(
                                json.dumps({"channels": 1, "bitsPerSample": 16})
                            )

                        csv_rows.append(
                            {
                                "Well": f"{well[0]} - {int(well[1:])}",
                                "Row": ord(well[0]) - 64,
                                "Column": int(well[1:]),
                                "Field": site,
                                "Wavelength": channel,
                                "Timepoint": t,
                                "ZIndex": z,
                                "ImageSubFolderPath": f"timepoint{t}",
                                "ImageFileName": name,
                                "TimeStampSec": EPOCH + t * TIME_INTERVAL_S,
                                "ExposureTimeMs": 10,
                                "ExcitationEmissionFilter": channels[channel],
                                "PositionXUm": position_x,
                                "PositionYUm": position_y,
                                "PositionZUm": 20000.0 + z * z_step,
                            }
                        )

        if write_sidecars:
            (timepoint_dir / "Thumbs.db").write_bytes(b"\x00")

    if write_csv:
        lines = [",".join(CSV_COLUMNS)]
        for row in csv_rows:
            lines.append(",".join(str(row[column]) for column in CSV_COLUMNS))
        (root / "image_metadata_1.csv").write_text("\n".join(lines) + "\n")

    return root


###############################################################################
# Fixtures


@pytest.fixture
def acquisition_unit(tmp_path: Path) -> Path:
    """A 2 well x 2 site x 2 t x 2 c x 3 z acquisition."""
    return make_acquisition_unit(tmp_path / "experiment")


@pytest.fixture
def mosaic_unit(tmp_path: Path) -> Path:
    """
    A well imaged as a full 2x2 grid of overlapping tiles, as the instrument does.

    The default `acquisition_unit` uses two sites, which is a valid mosaic but
    only exercises one axis; this one places tiles on both.
    """
    return make_acquisition_unit(
        tmp_path / "mosaic",
        wells=["B02"],
        sites=[0, 1, 2, 3],
        channels=["TL"],
        t_count=1,
        z_count=1,
    )


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """
    A run root holding two units that disagree about shape and channel count --
    the arrangement that makes any globally cached state a bug.
    """
    root = tmp_path / "run"
    make_acquisition_unit(root / "experiment", project="Fixture")
    make_acquisition_unit(
        root / "experiment_montage",
        project="Fixture",
        sites=[0],
        channels=["TL"],
        t_count=1,
        z_count=1,
        pixel_size=1.6595,
        plane_size=16,
    )
    # Present in real exports, holds no images, and must be ignored.
    (root / "autofocus").mkdir()
    (root / "autofocus" / "focus_report.txt").write_text("focus")

    return root
