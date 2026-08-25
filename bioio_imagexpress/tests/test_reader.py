import pathlib
import shutil
from typing import List, Optional, Tuple

import bioio
import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities

from bioio_imagexpress import Reader

from .conftest import LOCAL_RESOURCES_DIR, descriptor

RUN_ROOT = LOCAL_RESOURCES_DIR / "run_root"
Z_STACK = LOCAL_RESOURCES_DIR / "experiment_z_stack"
EXPERIMENT = RUN_ROOT / "experiment"
MONTAGE = RUN_ROOT / "experiment_montage"


@pytest.mark.parametrize("entry", ["directory", "descriptor"])
@pytest.mark.parametrize(
    "acquisition, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            Z_STACK,
            "B07",
            ("B07", "B08"),
            (2, 2, 2, 3, 64, 64),
            np.uint16,
            ["TL", "FITC"],
            (3.0, 0.5817, 0.5817),
        ),
        (
            EXPERIMENT,
            "B02",
            ("B02", "B03"),
            (2, 1, 1, 1, 64, 64),
            np.uint16,
            ["TL"],
            (None, 1.6595, 1.6595),
        ),
        (
            MONTAGE,
            "B03",
            ("B02", "B03"),
            (1, 1, 1, 1, 64, 64),
            np.uint16,
            ["TL"],
            (None, 1.6595, 1.6595),
        ),
    ],
)
def test_imagexpress_reader(
    entry: str,
    acquisition: pathlib.Path,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
) -> None:
    uri = acquisition if entry == "directory" else descriptor(acquisition)

    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=dict,
        expected_resolution_levels=(0, 1, 2),
    )


def test_imagexpress_reader_without_mosaic() -> None:
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=EXPERIMENT,
        set_scene="B03-s1",
        expected_scenes=("B02-s0", "B02-s1", "B03-s0", "B03-s1"),
        expected_current_scene="B03-s1",
        expected_shape=(1, 1, 1, 64, 64),
        expected_dtype=np.uint16,
        expected_dims_order=dimensions.DEFAULT_DIMENSION_ORDER,
        expected_channel_names=["TL"],
        expected_physical_pixel_sizes=(None, 1.6595, 1.6595),
        expected_metadata_type=dict,
        expected_resolution_levels=(0, 1, 2),
        reader_kwargs=dict(reconstruct_mosaic=False),
    )

    # Each per-site scene is the matching mosaic tile of the same well.
    mosaic = Reader(EXPERIMENT)
    tiles = Reader(EXPERIMENT, reconstruct_mosaic=False)
    for scene, expected in zip(("B02-s0", "B02-s1"), mosaic.data):
        tiles.set_scene(scene)
        assert tiles.position_index == int(scene[-1])
        np.testing.assert_array_equal(tiles.data, expected)


@pytest.mark.parametrize(
    "path, error_mentions",
    [
        pytest.param(RUN_ROOT, ["run root"], id="run_root"),
        pytest.param(RUN_ROOT / "autofocus", [], id="not_an_acquisition"),
        pytest.param(
            Z_STACK / "timepoint0" / "Wellscan_96well_TL_488_t0_B07_s0_w0_z0.tif",
            [],
            id="bare_plane",
        ),
        pytest.param(Z_STACK / "image_metadata_1.csv", [], id="manifest"),
        pytest.param(RUN_ROOT / "autofocus" / "focus_report.txt", [], id="text_file"),
    ],
)
def test_imagexpress_reader_unsupported(
    path: pathlib.Path, error_mentions: List[str]
) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError) as caught:
        Reader(path)

    for expected in error_mentions:
        assert expected in str(caught.value)


def test_missing_manifest_raises(tmp_path: pathlib.Path) -> None:
    acquisition = tmp_path / "experiment"
    shutil.copytree(EXPERIMENT, acquisition)
    (acquisition / "image_metadata_1.csv").unlink()

    with pytest.raises(exceptions.UnsupportedFileFormatError, match="manifest"):
        Reader(acquisition)


def test_stitched_mosaic_places_each_tile_at_its_position() -> None:
    reader = Reader(EXPERIMENT)
    tiles = reader.data
    stitched = reader.mosaic_data

    for (top, left), tile in zip(reader.get_mosaic_tile_positions(), tiles):
        np.testing.assert_array_equal(
            stitched[..., top : top + 64, left : left + 64], tile
        )


def test_resolution_levels() -> None:
    expected_shapes = [
        (2, 2, 2, 3, 64, 64),
        (2, 2, 2, 3, 32, 32),
        (2, 2, 2, 3, 16, 16),
    ]
    expected_pixel_sizes = [
        (3.0, 0.5817, 0.5817),
        (3.0, 1.1634, 1.1634),
        (3.0, 2.3268, 2.3268),
    ]
    reader = Reader(Z_STACK)
    assert reader.resolution_levels == (0, 1, 2)

    for level, (shape, pixel_sizes) in enumerate(
        zip(expected_shapes, expected_pixel_sizes)
    ):
        reader.set_resolution_level(level)
        assert reader.current_resolution_level == level
        assert reader.shape == shape
        assert reader.data.shape == shape
        assert reader.physical_pixel_sizes == pixel_sizes


def test_mosaic_tile_positions() -> None:
    reader = Reader(EXPERIMENT)
    positions = reader.get_mosaic_tile_positions()

    assert positions == [(0, 0), (2074, 0)]
    assert [
        reader.get_mosaic_tile_position(index) for index in range(len(positions))
    ] == positions

    # The stitched extent covers the last tile's origin plus one tile.
    assert reader.mosaic_xarray_dask_data.shape == (1, 1, 1, 2138, 64)
    assert reader.mosaic_xarray_dask_data.dims == tuple(
        dimensions.DEFAULT_DIMENSION_ORDER
    )
    assert reader.mosaic_data.shape == (1, 1, 1, 2138, 64)

    reader.set_resolution_level(1)
    assert reader.get_mosaic_tile_positions() == [(0, 0), (1037, 0)]


def test_missing_plane_raises(tmp_path: pathlib.Path) -> None:
    acquisition = tmp_path / "experiment"
    shutil.copytree(EXPERIMENT, acquisition)
    plane = sorted(acquisition.glob("timepoint0/*.tif"))[1]
    plane.unlink()

    reader = Reader(acquisition)
    assert reader.shape == (2, 1, 1, 1, 64, 64)

    with pytest.raises(FileNotFoundError):
        reader.data


def test_bioimage_routing() -> None:
    image = bioio.BioImage(descriptor(EXPERIMENT))
    assert isinstance(image.reader, Reader)
    assert image.scenes == ("B02", "B03")
    assert image.dims.order == dimensions.DEFAULT_DIMENSION_ORDER
    assert image.shape == (1, 1, 1, 2138, 64)
