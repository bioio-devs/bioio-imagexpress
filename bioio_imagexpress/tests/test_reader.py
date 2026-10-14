import pathlib
import shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fsspec
import numpy as np
import pytest
from bioio import BioImage
from bioio_base import dimensions, exceptions, test_utilities

from bioio_imagexpress import Reader
from bioio_imagexpress.acquisition import parse_jdce, read_manifest

from .conftest import LOCAL_RESOURCES_DIR

###############################################################################


@pytest.fixture(scope="module")
def mem_fs() -> Iterator[fsspec.AbstractFileSystem]:
    mem = fsspec.filesystem("memory")
    mem.put(
        str(LOCAL_RESOURCES_DIR / "experiment_z_stack"),
        "/experiment_z_stack",
        recursive=True,
    )
    yield mem
    mem.rm("/experiment_z_stack", recursive=True)


@pytest.fixture
def acquisition_copy(tmp_path: pathlib.Path) -> pathlib.Path:
    copy = tmp_path / "experiment"
    shutil.copytree(LOCAL_RESOURCES_DIR / "run_root" / "experiment", copy)
    return copy


###############################################################################


@pytest.mark.parametrize(
    "image, set_scene, expected_scenes, expected_shape, expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            LOCAL_RESOURCES_DIR / "experiment_minimal",
            "D06-s0",
            ("D06-s0",),
            (1, 1, 1, 2304, 2304),
            ["TL"],
            (None, 0.5817, 0.5817),
        ),
        (
            LOCAL_RESOURCES_DIR / "experiment_z_stack",
            "B08-s1",
            ("B07-s0", "B07-s1", "B08-s0", "B08-s1"),
            (2, 2, 3, 64, 64),
            ["TL", "FITC"],
            (3.0, 0.5817, 0.5817),
        ),
        (
            LOCAL_RESOURCES_DIR / "run_root" / "experiment",
            "B03-s1",
            ("B02-s0", "B02-s1", "B03-s0", "B03-s1"),
            (1, 1, 1, 64, 64),
            ["TL"],
            (None, 1.6595, 1.6595),
        ),
        (
            LOCAL_RESOURCES_DIR / "run_root" / "experiment_montage",
            "B03-s0",
            ("B02-s0", "B03-s0"),
            (1, 1, 1, 64, 64),
            ["TL"],
            (None, 1.6595, 1.6595),
        ),
        pytest.param(
            LOCAL_RESOURCES_DIR,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_imagexpress_reader(
    image: pathlib.Path,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
) -> None:
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=image,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.uint16),
        expected_dims_order=dimensions.DEFAULT_DIMENSION_ORDER,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=dict,
        expected_resolution_levels=(0, 1, 2),
    )


def test_bioimage_reads_remote_acquisition(
    mem_fs: fsspec.AbstractFileSystem,
) -> None:
    img = BioImage("memory://experiment_z_stack")
    assert isinstance(img.reader, Reader)
    assert img.shape == (2, 2, 3, 64, 64)

###############################################################################


def test_parse_jdce_tolerates_bad_values() -> None:
    metadata = parse_jdce(
        '{"ImageStack": {"Creation": {"Date": "nope", "Time": "x"}, '
        '"AutoLeadAcquisitionProtocol": {"ObjectiveCalibration": '
        '{"PixelWidth": "wide"}}}}'
    )
    assert metadata.acquired_at is None
    assert metadata.pixel_size_x is None

@pytest.mark.parametrize(
    "filename, contents",
    [
        ("Lumenoid_4x_LabelfreeColony_20260807.jdce", "{not json"),
        pytest.param(
            "image_metadata_1.csv",
            "",
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_damaged_acquisition_files(
    acquisition_copy: pathlib.Path, filename: str, contents: str
) -> None:
    (acquisition_copy / filename).write_text(contents)
    assert len(Reader(acquisition_copy).scenes) == 4


def test_missing_plane_file_raises(acquisition_copy: pathlib.Path) -> None:
    for tif in (acquisition_copy / "timepoint0").glob("*B02_s0*.tif"):
        tif.unlink()
    with pytest.raises(IOError):
        BioImage(acquisition_copy).get_image_data("YX")
