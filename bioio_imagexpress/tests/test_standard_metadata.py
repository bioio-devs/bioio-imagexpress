import dataclasses
import datetime
import pathlib
from typing import Any, Dict

import pytest
from bioio import BioImage

from .conftest import LOCAL_RESOURCES_DIR

###############################################################################


@pytest.mark.parametrize(
    "image, expected",
    [
        (
            LOCAL_RESOURCES_DIR / "experiment_minimal",
            {
                "binning": "1x1",
                "column": "6",
                "dimensions_present": "TCZYX",
                "image_size_c": 1,
                "image_size_t": 1,
                "image_size_x": 2304,
                "image_size_y": 2304,
                "image_size_z": 1,
                "imaged_by": "svc_aicspipeline",
                "imaging_datetime": datetime.datetime(2026, 9, 22, 10, 12, 10),
                "objective": "10X Plan Apo Lambda D",
                "pixel_size_x": 0.5817,
                "pixel_size_y": 0.5817,
                "pixel_size_z": None,
                "position_index": 0,
                "row": "D",
                "stage_position_x": 59300.0,
                "stage_position_y": 38360.0,
                "timelapse": False,
                "timelapse_interval": None,
                "total_time_duration": None,
            },
        ),
        (
            LOCAL_RESOURCES_DIR / "experiment_z_stack",
            {
                "binning": "1x1",
                "column": "7",
                "dimensions_present": "TCZYX",
                "image_size_c": 2,
                "image_size_t": 2,
                "image_size_x": 64,
                "image_size_y": 64,
                "image_size_z": 3,
                "imaged_by": "moldev",
                "imaging_datetime": datetime.datetime(2026, 8, 4, 10, 31, 10),
                "objective": "10X Plan Apo Lambda D",
                "pixel_size_x": 0.5817,
                "pixel_size_y": 0.5817,
                "pixel_size_z": 3.0,
                "position_index": 0,
                "row": "B",
                "stage_position_x": 67696.78,
                "stage_position_y": 19757.0,
                "timelapse": True,
                "timelapse_interval": datetime.timedelta(seconds=21599.754),
                "total_time_duration": datetime.timedelta(seconds=21599.754),
            },
        ),
        (
            LOCAL_RESOURCES_DIR / "run_root" / "experiment",
            {
                "binning": "1x1",
                "column": "2",
                "dimensions_present": "TCZYX",
                "image_size_c": 1,
                "image_size_t": 1,
                "image_size_x": 64,
                "image_size_y": 64,
                "image_size_z": 1,
                "imaged_by": "moldev",
                "imaging_datetime": datetime.datetime(2026, 8, 7, 15, 7, 15),
                "objective": "4X Plan Apo Lambda D",
                "pixel_size_x": 1.6595,
                "pixel_size_y": 1.6595,
                "pixel_size_z": None,
                "position_index": 0,
                "row": "B",
                "stage_position_x": 21579.42,
                "stage_position_y": 18639.46,
                "timelapse": False,
                "timelapse_interval": None,
                "total_time_duration": None,
            },
        ),
        (
            LOCAL_RESOURCES_DIR / "run_root" / "experiment_montage",
            {
                "binning": "1x1",
                "column": "2",
                "dimensions_present": "TCZYX",
                "image_size_c": 1,
                "image_size_t": 1,
                "image_size_x": 64,
                "image_size_y": 64,
                "image_size_z": 1,
                "imaged_by": "moldev",
                "imaging_datetime": datetime.datetime(2026, 8, 7, 15, 7, 15),
                "objective": "4X Plan Apo Lambda D",
                "pixel_size_x": 1.6595,
                "pixel_size_y": 1.6595,
                "pixel_size_z": None,
                "position_index": 0,
                "row": "B",
                "stage_position_x": 23300.02,
                "stage_position_y": 20360.01,
                "timelapse": False,
                "timelapse_interval": None,
                "total_time_duration": None,
            },
        ),
    ],
)
def test_standard_metadata(image: pathlib.Path, expected: Dict[str, Any]) -> None:
    actual = dataclasses.asdict(BioImage(image).standard_metadata)
    assert {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in actual.items()
    } == expected
