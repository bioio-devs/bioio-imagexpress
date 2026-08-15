#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fsspec.implementations.local import LocalFileSystem

from bioio_imagexpress import acquisition

from .conftest import LOCAL_RESOURCES_DIR, descriptor

Z_STACK = LOCAL_RESOURCES_DIR / "experiment_z_stack"
EXPERIMENT = LOCAL_RESOURCES_DIR / "run_root" / "experiment"


def build_unit(directory: pathlib.Path) -> acquisition.AcquisitionUnit:
    """Index the acquisition in ``directory`` off the local filesystem."""
    fs = LocalFileSystem()
    discovered = acquisition.discover_unit(fs, str(directory))
    assert discovered is not None

    return acquisition.build_unit(fs, *discovered)


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "Wellscan_96well_TL_488_t0_B07_s0_w1_z2.tif",
            acquisition.PlaneName(t=0, well="B07", site=0, channel=1, z=2),
        ),
        (
            "4X_Lumenoid_Cellvis_96_8-7-2026_t0_B03_s1_w0_z0.tif",
            acquisition.PlaneName(t=0, well="B03", site=1, channel=0, z=0),
        ),
        ("Thumbs.db", None),
        ("Wellscan_96well_TL_488_t0_B07_s0_w0_z0.tif.statistics.json", None),
        ("focus_report.txt", None),
        # A single digit well is not the grammar's two digit token.
        ("Wellscan_96well_TL_488_t0_B7_s0_w0_z0.tif", None),
    ],
)
def test_parse_plane_name(name: str, expected: Optional[acquisition.PlaneName]) -> None:
    assert acquisition.parse_plane_name(name) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("B - 7", "B07"),
        ("B07", "B07"),
        ("b7", "B07"),
        ("B-7", "B07"),
        ("", None),
        ("12", None),
        ("Well", None),
    ],
)
def test_normalize_well(value: str, expected: Optional[str]) -> None:
    assert acquisition.normalize_well(value) == expected


@pytest.mark.parametrize(
    "directory, expected",
    [
        (
            Z_STACK,
            {
                "channel_names": ["TL", "FITC"],
                "pixel_size_x": 0.5817,
                "pixel_size_y": 0.5817,
                "z_step": 3.0,
                "objective": "10X Plan Apo Lambda D",
                "binning": "1x1",
                "operator": "moldev",
                "acquired_at": datetime(2026, 8, 4, 10, 31, 10),
                "metadata_files": ["image_metadata_1.csv"],
            },
        ),
        (
            EXPERIMENT,
            {
                "channel_names": ["TL"],
                "pixel_size_x": 1.6595,
                "pixel_size_y": 1.6595,
                # The descriptor records ZStep 0.0, which is no Z spacing at all.
                "z_step": None,
                "objective": "4X Plan Apo Lambda D",
                "binning": "1x1",
                "operator": "moldev",
                "acquired_at": datetime(2026, 8, 7, 15, 7, 15),
                "metadata_files": ["image_metadata_1.csv"],
            },
        ),
    ],
)
def test_parse_jdce(directory: pathlib.Path, expected: Dict[str, Any]) -> None:
    metadata = acquisition.parse_jdce(
        descriptor(directory).read_text(encoding="utf-8-sig")
    )

    assert {name: getattr(metadata, name) for name in expected} == expected


def test_read_manifest() -> None:
    manifest = Z_STACK / "image_metadata_1.csv"
    rows = acquisition.read_manifest(manifest.read_text(encoding="utf-8-sig"))

    assert len(rows) == 48
    assert rows[0] == acquisition.ManifestRow(
        well="B07",
        site=0,
        channel=0,
        t=0,
        z=0,
        subfolder="timepoint0",
        filename="Wellscan_96well_TL_488_t0_B07_s0_w0_z0.tif",
        timestamp_s=1785864670.188,
        position_x_um=67696.78,
        position_y_um=19757.0,
    )


def test_read_manifest_drops_incomplete_rows() -> None:
    contents = (
        "Well,Field,Wavelength,Timepoint,ZIndex,ImageSubFolderPath,ImageFileName\n"
        "B - 2,0,0,0,0,timepoint0,good.tif\n"
        "B - 2,,0,0,0,timepoint0,no_field.tif\n"
        "notawell,1,0,0,0,timepoint0,bad_well.tif\n"
        "B - 2,1,0,0,0,timepoint0,\n"
    )
    rows = acquisition.read_manifest(contents)

    assert [row.filename for row in rows] == ["good.tif"]


@pytest.mark.parametrize(
    "directory, "
    "expected_wells, "
    "expected_sites, "
    "expected_extents, "
    "expected_channel_names, "
    "expected_time_coords",
    [
        (
            Z_STACK,
            ["B07", "B08"],
            [0, 1],
            ([0, 1], [0, 1], [0, 1, 2]),
            ["TL", "FITC"],
            [0.0, 21599.7539999485],
        ),
        (EXPERIMENT, ["B02", "B03"], [0, 1], ([0], [0], [0]), ["TL"], [0.0]),
    ],
)
def test_acquisition_unit_accessors(
    directory: pathlib.Path,
    expected_wells: List[str],
    expected_sites: List[int],
    expected_extents: Tuple[List[int], List[int], List[int]],
    expected_channel_names: List[str],
    expected_time_coords: List[float],
) -> None:
    unit = build_unit(directory)
    keys = unit.scene_keys

    assert unit.wells == expected_wells
    assert keys == [(well, site) for well in expected_wells for site in expected_sites]
    assert unit.sites(expected_wells[0]) == expected_sites
    assert unit.sites("Z99") == []
    assert unit.extents(keys) == expected_extents
    assert unit.channel_names(keys) == expected_channel_names
    assert unit.time_coords(keys) == pytest.approx(expected_time_coords)


def test_acquisition_unit_time_coords_without_manifest(
    tmp_path: pathlib.Path,
) -> None:
    copied = tmp_path / "experiment"
    shutil.copytree(EXPERIMENT, copied)
    (copied / "image_metadata_1.csv").unlink()

    unit = build_unit(copied)

    assert unit.wells == ["B02", "B03"]
    assert unit.time_coords(unit.scene_keys) is None
