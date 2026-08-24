import datetime
from typing import Any, Dict, Optional

import pytest

from bioio_imagexpress.reader import Reader

from .conftest import LOCAL_RESOURCES_DIR

TEST_CASES = [
    pytest.param(
        "experiment_z_stack",
        {
            "Binning": "1x1",
            "Column": "7",
            "Dimensions Present": "MTCZYX",
            "Image Size C": 2,
            "Image Size T": 2,
            "Image Size X": 64,
            "Image Size Y": 64,
            "Image Size Z": 3,
            "Imaged By": "moldev",
            "Imaging Datetime": datetime.datetime(2026, 8, 4, 10, 31, 10),
            "Objective": "10X Plan Apo Lambda D",
            "Pixel Size X": 0.5817,
            "Pixel Size Y": 0.5817,
            "Pixel Size Z": 3.0,
            "Position Index": None,
            "Row": "B",
            "Stage Position X": 67696.78,
            "Stage Position Y": 19757.0,
            "Timelapse": True,
            "Timelapse Interval": datetime.timedelta(
                seconds=21599, microseconds=754000
            ),
            "Total Time Duration": datetime.timedelta(
                seconds=21599, microseconds=754000
            ),
        },
        id="experiment_z_stack",
    ),
    pytest.param(
        "run_root/experiment",
        {
            "Binning": "1x1",
            "Column": "2",
            "Dimensions Present": "MTCZYX",
            "Image Size C": 1,
            "Image Size T": 1,
            "Image Size X": 64,
            "Image Size Y": 64,
            "Image Size Z": 1,
            "Imaged By": "moldev",
            "Imaging Datetime": datetime.datetime(2026, 8, 7, 15, 7, 15),
            "Objective": "4X Plan Apo Lambda D",
            "Pixel Size X": 1.6595,
            "Pixel Size Y": 1.6595,
            "Pixel Size Z": None,
            "Position Index": None,
            "Row": "B",
            "Stage Position X": 21579.42,
            "Stage Position Y": 18639.46,
            "Timelapse": False,
            "Timelapse Interval": None,
            "Total Time Duration": None,
        },
        id="run_root_experiment",
    ),
]


@pytest.mark.parametrize("filename, expected_dict", TEST_CASES)
def test_imagexpress_standard_metadata(
    filename: str, expected_dict: Dict[str, Any]
) -> None:
    # Arrange
    reader = Reader(LOCAL_RESOURCES_DIR / filename)

    # Act
    sm_dict = reader.standard_metadata.to_dict()

    # Assert
    for key, expected in expected_dict.items():
        assert key in sm_dict, f"Key '{key}' missing from standard_metadata.to_dict()"
        result = sm_dict[key]
        if isinstance(expected, datetime.timedelta):
            assert isinstance(result, datetime.timedelta)
            diff = abs(result.total_seconds() - expected.total_seconds())
            assert (
                diff < 1e-3
            ), f"{key} expected {expected}, got {result} (delta={diff}s)"
        elif isinstance(expected, float):
            assert result == pytest.approx(expected, rel=1e-9, abs=1e-12)
        else:
            assert result == expected, f"{key} expected {expected}, got {result}"


@pytest.mark.parametrize(
    "filename, scene, expected_row, expected_column",
    [
        ("experiment_z_stack", "B08", "B", "8"),
        ("run_root/experiment", "B03", "B", "3"),
    ],
)
def test_standard_metadata_follows_set_scene(
    filename: str, scene: str, expected_row: str, expected_column: str
) -> None:
    """The plate position tracks the current scene."""
    reader = Reader(LOCAL_RESOURCES_DIR / filename)
    reader.set_scene(scene)
    sm_dict = reader.standard_metadata.to_dict()

    assert sm_dict["Row"] == expected_row
    assert sm_dict["Column"] == expected_column


@pytest.mark.parametrize(
    "mosaic, scene, expected_position_index",
    [
        (True, "B02", None),
        (False, "B02-s0", 0),
        (False, "B03-s1", 1),
    ],
)
def test_standard_metadata_position_index(
    mosaic: bool, scene: str, expected_position_index: Optional[int]
) -> None:
    """A whole-well scene has no position; a per-site scene reports its site."""
    reader = Reader(LOCAL_RESOURCES_DIR / "run_root/experiment", mosaic=mosaic)
    reader.set_scene(scene)

    assert reader.standard_metadata.to_dict()["Position Index"] == (
        expected_position_index
    )
