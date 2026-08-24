#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import shutil
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import bioio
import numpy as np
import pytest
import tifffile
from bioio_base import dimensions, exceptions, test_utilities

from bioio_imagexpress import Reader, ReaderMetadata

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
    # One scene stands in for all four: per-scene pixel correctness is covered
    # by test_tiles_without_mosaic_match_mosaic_tiles.
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
        reader_kwargs=dict(mosaic=False),
    )


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(RUN_ROOT, id="run_root"),
        pytest.param(RUN_ROOT / "autofocus", id="not_an_acquisition"),
        pytest.param(
            Z_STACK / "timepoint0" / "Wellscan_96well_TL_488_t0_B07_s0_w0_z0.tif",
            id="bare_plane",
        ),
        pytest.param(Z_STACK / "image_metadata_1.csv", id="manifest"),
    ],
)
def test_imagexpress_reader_unsupported(path: pathlib.Path) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(path)


def test_imagexpress_reader_with_text_file(tmp_path: pathlib.Path) -> None:
    text_file = tmp_path / "example.txt"
    text_file.write_text("not an acquisition")

    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(text_file)


def test_run_root_error_names_sub_units() -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError) as caught:
        Reader(RUN_ROOT)

    assert "experiment" in str(caught.value)
    assert "experiment_montage" in str(caught.value)


def test_tiles_without_mosaic_match_mosaic_tiles() -> None:
    mosaic = Reader(EXPERIMENT)
    tiles = Reader(EXPERIMENT, mosaic=False)

    for scene, expected in zip(("B02-s0", "B02-s1"), mosaic.data):
        tiles.set_scene(scene)
        assert tiles.position_index == int(scene[-1])
        np.testing.assert_array_equal(tiles.data, expected)


def test_planes_come_from_the_files_that_name_them() -> None:
    # Every plane is its own TIFF, so the whole reader is an index: check each
    # position of the assembled array against the file its name points at, which
    # a transposed or reversed stack would still have the right shape for.
    reader = Reader(Z_STACK)
    reader.set_scene("B07")
    data = reader.data
    sites, timepoints, channels, zs = 2, 2, 2, 3

    assert data.shape == (sites, timepoints, channels, zs, 64, 64)

    for m, t, c, z in product(
        range(sites), range(timepoints), range(channels), range(zs)
    ):
        plane = (
            Z_STACK
            / f"timepoint{t}"
            / f"Wellscan_96well_TL_488_t{t}_B07_s{m}_w{c}_z{z}.tif"
        )
        np.testing.assert_array_equal(
            data[m, t, c, z],
            tifffile.imread(plane),
            err_msg=f"array position (m={m}, t={t}, c={c}, z={z}) is not {plane.name}",
        )


def test_stitched_mosaic_places_each_tile_at_its_position() -> None:
    reader = Reader(EXPERIMENT)
    tiles = reader.data
    stitched = reader.mosaic_data

    for (top, left), tile in zip(reader.get_mosaic_tile_positions(), tiles):
        np.testing.assert_array_equal(
            stitched[..., top : top + 64, left : left + 64], tile
        )


def test_ragged_acquisition_keeps_the_completed_tiles(tmp_path: pathlib.Path) -> None:
    # An acquisition aborted part way leaves one tile short of a time point the
    # others completed. That tile must not shrink the scene, and the planes it
    # never wrote must raise rather than come back as zeros.
    unit = tmp_path / "experiment_z_stack"
    shutil.copytree(Z_STACK, unit)

    # The first tile is the one aborted, so a scene sized off it alone would be
    # a time point short rather than merely raising.
    aborted = "_t1_B07_s0_w"
    manifest = unit / "image_metadata_1.csv"
    manifest.write_text(
        "\n".join(
            line
            for line in manifest.read_text(encoding="utf-8-sig").splitlines()
            if aborted not in line
        )
        + "\n"
    )
    for plane in (unit / "timepoint1").glob(f"*{aborted}*.tif"):
        plane.unlink()

    reader = Reader(unit)
    assert reader.scenes == ("B07", "B08")
    assert reader.shape == (2, 2, 2, 3, 64, 64)

    with pytest.raises(FileNotFoundError):
        reader.data

    # The well that completed is untouched by its neighbour's missing planes.
    reader.set_scene("B08")
    expected = Reader(Z_STACK)
    expected.set_scene("B08")
    np.testing.assert_array_equal(reader.data, expected.data)


def test_resolution_levels() -> None:
    # The level scaling math does not depend on the unit, so one unit suffices.
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

    # Tile origins are pixels, so they follow the resolution level.
    reader.set_resolution_level(1)
    assert reader.get_mosaic_tile_positions() == [(0, 0), (1037, 0)]


def test_single_tile_mosaic() -> None:
    reader = Reader(MONTAGE)

    assert reader.get_mosaic_tile_positions() == [(0, 0)]
    assert reader.mosaic_xarray_dask_data.shape == (1, 1, 1, 64, 64)


def test_mosaic_tile_positions_without_mosaic() -> None:
    reader = Reader(EXPERIMENT, mosaic=False)

    with pytest.raises(exceptions.UnexpectedShapeError):
        reader.get_mosaic_tile_positions()


def test_time_and_z_coords() -> None:
    reader = Reader(Z_STACK)
    coords = reader.xarray_dask_data.coords

    np.testing.assert_allclose(
        coords[dimensions.DimensionNames.Time].values, [0.0, 21599.7539999485]
    )
    np.testing.assert_allclose(
        coords[dimensions.DimensionNames.SpatialZ].values, [0.0, 3.0, 6.0]
    )


def test_unprocessed_metadata() -> None:
    reader = Reader(Z_STACK)
    reader.set_scene("B08")
    unprocessed = reader.xarray_dask_data.attrs["unprocessed"]

    assert sorted(unprocessed) == [
        "column",
        "jdce",
        "metaseries",
        "row",
        "sites",
        "stage_position_um",
        "tile_stage_positions_um",
        "well",
    ]
    assert unprocessed["well"] == "B08"
    assert unprocessed["row"] == "B"
    assert unprocessed["column"] == 8
    assert unprocessed["sites"] == [0, 1]
    assert reader.metadata is unprocessed


def test_scene_switching_does_not_leak_state() -> None:
    reader = Reader(Z_STACK)
    first = reader.data

    reader.set_scene("B08")
    assert reader.shape == (2, 2, 2, 3, 64, 64)
    assert reader.current_scene == "B08"

    reader.set_scene("B07")
    assert reader.shape == (2, 2, 2, 3, 64, 64)
    np.testing.assert_array_equal(reader.data, first)


@pytest.mark.parametrize(
    "dimension_order, selection",
    [
        ("YX", dict(M=0, T=1, C=1, Z=2)),
        ("MTZYX", dict(C=0)),
        ("MTCZYX", dict(Z=slice(0, 2))),
    ],
)
def test_get_image_data_selections(
    dimension_order: str, selection: Dict[str, Any]
) -> None:
    reader = Reader(Z_STACK)
    full = reader.xarray_data.data

    np.testing.assert_array_equal(
        reader.get_image_data(dimension_order, **selection),
        full[
            tuple(
                selection.get(dim, slice(None))
                for dim in dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES
            )
        ],
    )


def test_walk_fallback_without_manifest(tmp_path: pathlib.Path) -> None:
    acquisition = tmp_path / "experiment"
    shutil.copytree(EXPERIMENT, acquisition)
    for manifest in acquisition.glob("image_metadata_*.csv"):
        manifest.unlink()

    reader = Reader(acquisition)

    # Stage positions are manifest-only, so the wells cannot be stitched and the
    # reader falls back to one scene per acquisition position.
    assert reader.scenes == ("B02-s0", "B02-s1", "B03-s0", "B03-s1")
    assert reader.shape == (1, 1, 1, 64, 64)
    # Channel names still come from the descriptor.
    assert reader.channel_names == ["TL"]
    assert reader.physical_pixel_sizes == (None, 1.6595, 1.6595)
    # Timestamps are manifest-only too.
    assert dimensions.DimensionNames.Time not in reader.xarray_dask_data.coords
    assert reader.stage_position == (None, None)
    assert reader.total_time_duration is None
    np.testing.assert_array_equal(reader.data, Reader(EXPERIMENT, mosaic=False).data)

    # The default BioImage path must work too; it used to ask the degraded
    # mosaic for tile positions it could never have.
    image = bioio.BioImage(descriptor(acquisition))
    assert image.data.shape == (1, 1, 1, 64, 64)


def test_missing_plane_raises(tmp_path: pathlib.Path) -> None:
    acquisition = tmp_path / "experiment"
    shutil.copytree(EXPERIMENT, acquisition)
    # Not the scene's first plane, which the shape probe opens before any pixels.
    plane = sorted(acquisition.glob("timepoint0/*.tif"))[1]
    plane.unlink()

    reader = Reader(acquisition)
    assert reader.shape == (2, 1, 1, 1, 64, 64)

    with pytest.raises(FileNotFoundError):
        reader.data


def test_reader_metadata() -> None:
    assert ReaderMetadata.get_supported_extensions() == [".jdce"]
    assert ReaderMetadata.get_reader() is Reader


def test_bioimage_routing() -> None:
    # No reader= : naming the descriptor is what lets bioio pick this plugin on
    # its own, which is the whole reason the .jdce extension is claimed.
    image = bioio.BioImage(descriptor(EXPERIMENT))

    assert isinstance(image.reader, Reader)
    assert image.scenes == ("B02", "B03")
    assert image.dims.order == dimensions.DEFAULT_DIMENSION_ORDER
    assert image.shape == (1, 1, 1, 2138, 64)
