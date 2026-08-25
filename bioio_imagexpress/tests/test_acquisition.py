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


def index_acquisition(directory: pathlib.Path) -> acquisition.Acquisition:
    """Index the acquisition in ``directory`` off the local filesystem."""
    fs = LocalFileSystem()
    discovered = acquisition.discover_acquisition(fs, str(directory))
    assert discovered is not None

    return acquisition.index_acquisition(fs, *discovered)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("B - 7", "B07"),
        ("B07", "B07"),
        ("b7", "B07"),
        ("B-7", "B07"),
        ("AA - 1", "AA01"),
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
                "channel_names": {0: "TL", 1: "FITC"},
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
                "channel_names": {0: "TL"},
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
        "B - 2,1,0,0,0,timepoint0\\zstep0,windows.tif\n"
    )
    rows = acquisition.read_manifest(contents)

    assert [row.filename for row in rows] == ["good.tif", "windows.tif"]
    # The instrument writes Windows separators; paths are composed as posix.
    assert rows[1].subfolder == "timepoint0/zstep0"


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
    acq = index_acquisition(directory)
    keys = acq.scene_keys

    assert acq.wells == expected_wells
    assert keys == [(well, site) for well in expected_wells for site in expected_sites]
    assert acq.sites(expected_wells[0]) == expected_sites
    assert acq.sites("Z99") == []
    assert acq.extents(keys) == expected_extents
    assert acq.channel_names(keys) == expected_channel_names
    assert acq.time_coords(keys) == pytest.approx(expected_time_coords)


def test_index_acquisition_without_manifest_indexes_nothing(
    tmp_path: pathlib.Path,
) -> None:
    copied = tmp_path / "experiment"
    shutil.copytree(EXPERIMENT, copied)
    (copied / "image_metadata_1.csv").unlink()

    acq = index_acquisition(copied)

    assert acq.planes == {}
