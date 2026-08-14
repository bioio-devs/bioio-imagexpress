#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reader behaviour against synthetic acquisitions."""

import logging
from pathlib import Path

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from bioio_base.dimensions import DimensionNames

from bioio_imagexpress import Reader

from .conftest import (
    PLANE_SIZE,
    TIME_INTERVAL_S,
    make_acquisition_unit,
    plane_value,
    tile_stage_position,
)

###############################################################################

# The pixel size the fixture descriptor records, needed to turn a tile's stage
# position back into the value the manifest carries.
PIXEL_SIZE_UM = 0.5817


def test_scenes_are_one_per_well(acquisition_unit: Path) -> None:
    """A well's sites are its mosaic tiles, so the well is the scene."""
    reader = Reader(acquisition_unit)

    assert reader.scenes == ("B02", "B03")


def test_scenes_are_one_per_well_and_site_without_mosaic(acquisition_unit: Path) -> None:
    """With mosaic off each acquisition position is its own scene."""
    reader = Reader(acquisition_unit, mosaic=False)

    assert reader.scenes == ("B02-s0", "B02-s1", "B03-s0", "B03-s1")


def test_dims_stack_tile_time_channel_and_z(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert reader.dims.order == "MTCZYX"
    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.dtype == np.uint16


def test_dims_stack_time_channel_and_z_without_mosaic(acquisition_unit: Path) -> None:
    """With mosaic off a scene is one tile, so there is no M dimension."""
    reader = Reader(acquisition_unit, mosaic=False)

    assert reader.dims.order == "TCZYX"
    assert reader.dims.shape == (2, 2, 3, 32, 32)
    assert reader.dtype == np.uint16


def test_planes_land_at_the_right_coordinates(acquisition_unit: Path) -> None:
    """
    Guards the stack order within one tile, which mosaic off makes the scene.
    Shape alone would not catch a transposed T and C.
    """
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")

    data = reader.get_image_data("TCZYX")

    for t in range(2):
        for channel in range(2):
            for z in range(3):
                assert data[t, channel, z, 0, 0] == plane_value("B03", 1, t, channel, z)


def test_tiles_land_on_the_mosaic_dimension_in_site_order(acquisition_unit: Path) -> None:
    """
    The default scene is a whole well, so M has to carry the sites in order --
    a swapped M would still have the right shape.
    """
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")

    data = reader.get_image_data("MTCZYX")

    for site in range(2):
        for t in range(2):
            for channel in range(2):
                for z in range(3):
                    assert data[site, t, channel, z, 0, 0] == plane_value(
                        "B03", site, t, channel, z
                    )


def test_delayed_and_immediate_agree(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert np.array_equal(reader.xarray_dask_data.compute(), reader.xarray_data)


def test_chunks_are_single_planes(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert reader.xarray_dask_data.data.chunksize == (1, 1, 1, 1, 32, 32)


def test_channel_names_come_from_the_descriptor(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert reader.channel_names == ["TL", "FITC"]


def test_physical_pixel_sizes(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)


def test_time_coords_are_seconds_from_the_start(acquisition_unit: Path) -> None:
    """
    The descriptor's TimeSchedule.Times[].Ms values are indices, not
    milliseconds, so the coords must come from the manifest's timestamps.
    """
    reader = Reader(acquisition_unit)

    coords = reader.xarray_dask_data.coords[DimensionNames.Time].values

    assert coords == pytest.approx([0.0, TIME_INTERVAL_S])


def test_z_coords_use_the_configured_step(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    coords = reader.xarray_dask_data.coords[DimensionNames.SpatialZ].values

    assert coords == pytest.approx([0.0, 3.0, 6.0])


def test_metadata_carries_plate_position(acquisition_unit: Path) -> None:
    """
    The scene is a whole well, so it names no single site: it lists the tiles it
    covers, and its stage position is the origin they are placed from.
    """
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")

    metadata = reader.xarray_dask_data.attrs["unprocessed"]
    tiles = [tile_stage_position(site, PLANE_SIZE, PIXEL_SIZE_UM) for site in (0, 1)]

    assert metadata["well"] == "B03"
    assert metadata["row"] == "B"
    assert metadata["column"] == 3
    assert metadata["site"] is None
    assert metadata["sites"] == [0, 1]
    assert metadata["stage_position_um"] == {"x": tiles[0][0], "y": tiles[0][1]}
    assert metadata["tile_stage_positions_um"] == [
        {"site": 0, "x": tiles[0][0], "y": tiles[0][1]},
        {"site": 1, "x": tiles[1][0], "y": tiles[1][1]},
    ]
    assert metadata["indexed_from"] == "image_metadata_csv"


def test_metadata_carries_plate_position_without_mosaic(acquisition_unit: Path) -> None:
    """With mosaic off the scene is one tile, and it names which one."""
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")

    metadata = reader.xarray_dask_data.attrs["unprocessed"]
    position_x, position_y = tile_stage_position(1, PLANE_SIZE, PIXEL_SIZE_UM)

    assert metadata["well"] == "B03"
    assert metadata["row"] == "B"
    assert metadata["column"] == 3
    assert metadata["site"] == 1
    assert metadata["stage_position_um"] == {"x": position_x, "y": position_y}
    assert metadata["indexed_from"] == "image_metadata_csv"


###############################################################################
# Input resolution


def test_accepts_a_jdce_file(acquisition_unit: Path) -> None:
    descriptor = next(acquisition_unit.glob("*.jdce"))

    reader = Reader(descriptor)

    assert reader.scenes == ("B02", "B03")


def test_run_root_exposes_every_unit_with_prefixed_ids(run_root: Path) -> None:
    reader = Reader(run_root)

    assert reader.scenes == (
        "experiment/B02",
        "experiment/B03",
        "experiment_montage/B02",
        "experiment_montage/B03",
    )


def test_run_root_prefixes_tile_scenes_too(run_root: Path) -> None:
    """The unit prefix is independent of what a scene covers."""
    reader = Reader(run_root, mosaic=False)

    assert reader.scenes == (
        "experiment/B02-s0",
        "experiment/B02-s1",
        "experiment/B03-s0",
        "experiment/B03-s1",
        "experiment_montage/B02-s0",
        "experiment_montage/B03-s0",
    )


def test_rejects_a_directory_that_is_not_an_acquisition(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(tmp_path / "empty")


def test_rejects_a_bare_tiff(acquisition_unit: Path) -> None:
    plane = next((acquisition_unit / "timepoint0").glob("*.tif"))

    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(plane)


def test_is_supported_image_matches_construction(
    acquisition_unit: Path, tmp_path: Path
) -> None:
    (tmp_path / "empty").mkdir()

    assert Reader.is_supported_image(acquisition_unit)
    assert not Reader.is_supported_image(tmp_path / "empty")


###############################################################################
# Heterogeneous units


def test_units_keep_their_own_shape_and_metadata(run_root: Path) -> None:
    """
    A run root mixes units that disagree about shape, channels and pixel size, so
    nothing derived from a scene may be cached across a scene change.
    """
    reader = Reader(run_root)

    def snapshot(scene_id: str):
        reader.set_scene(scene_id)
        return (
            reader.dims.shape,
            tuple(reader.channel_names),
            reader.physical_pixel_sizes,
        )

    # The montage unit is the instrument's own stitch, one tile per well, so its
    # M is 1 where the experiment unit's is one per site.
    experiment = ((2, 2, 2, 3, 32, 32), ("TL", "FITC"), (3.0, 0.5817, 0.5817))
    montage = ((1, 1, 1, 1, 16, 16), ("TL",), (3.0, 1.6595, 1.6595))

    assert snapshot("experiment/B02") == experiment
    assert snapshot("experiment_montage/B02") == montage
    # And in the other order, to catch state that only leaks one way.
    assert snapshot("experiment/B03") == experiment
    assert snapshot("experiment_montage/B03") == montage


def test_pixels_stay_scene_scoped(run_root: Path) -> None:
    reader = Reader(run_root)

    reader.set_scene("experiment_montage/B03")
    montage = reader.get_image_data("YX")
    reader.set_scene("experiment/B02")
    experiment = reader.get_image_data("YX", M=0, T=0, C=0, Z=0)

    assert montage.shape == (16, 16)
    assert experiment.shape == (32, 32)
    assert montage[0, 0] == plane_value("B03", 0, 0, 0, 0)
    assert experiment[0, 0] == plane_value("B02", 0, 0, 0, 0)


###############################################################################
# Resolution levels


def test_resolution_levels_expose_the_embedded_pyramid(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert reader.resolution_levels == (0, 1, 2)


def test_setting_a_resolution_level_rescales_shape_and_pixel_size(
    acquisition_unit: Path,
) -> None:
    reader = Reader(acquisition_unit)

    reader.set_resolution_level(1)
    assert reader.dims.shape == (2, 2, 2, 3, 16, 16)
    assert reader.physical_pixel_sizes == pytest.approx((3.0, 1.1634, 1.1634))

    reader.set_resolution_level(2)
    assert reader.dims.shape == (2, 2, 2, 3, 8, 8)

    reader.set_resolution_level(0)
    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)


def test_downsampled_levels_still_carry_plane_values(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)
    reader.set_resolution_level(1)

    data = reader.get_image_data("YX", T=1, C=0, Z=2)

    assert data.shape == (16, 16)
    assert np.all(data == plane_value("B02", 0, 1, 0, 2))


###############################################################################
# Degraded acquisitions


def test_missing_planes_are_zero_filled_and_reported(tmp_path: Path) -> None:
    """The scene spans the well, so a missing plane is reported as (m, t, c, z)."""
    unit = make_acquisition_unit(
        tmp_path / "ragged",
        skip_planes=[("B02", 0, 1, 1, 2)],
    )

    reader = Reader(unit)
    reader.set_scene("B02")

    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.xarray_dask_data.attrs["missing_planes"] == [(0, 1, 1, 2)]
    assert np.all(reader.get_image_data("YX", M=0, T=1, C=1, Z=2) == 0)
    # A neighbouring plane must be untouched.
    assert np.all(
        reader.get_image_data("YX", M=0, T=1, C=1, Z=1)
        == plane_value("B02", 0, 1, 1, 1)
    )
    # The tile that kept every plane must be untouched as well.
    assert np.all(
        reader.get_image_data("YX", M=1, T=1, C=1, Z=2)
        == plane_value("B02", 1, 1, 1, 2)
    )


def test_missing_planes_are_zero_filled_and_reported_without_mosaic(tmp_path: Path) -> None:
    """With mosaic off the scene is one tile, so the report is (t, c, z)."""
    unit = make_acquisition_unit(
        tmp_path / "ragged",
        skip_planes=[("B02", 0, 1, 1, 2)],
    )

    reader = Reader(unit, mosaic=False)
    reader.set_scene("B02-s0")

    assert reader.dims.shape == (2, 2, 3, 32, 32)
    assert reader.xarray_dask_data.attrs["missing_planes"] == [(1, 1, 2)]
    assert np.all(reader.get_image_data("YX", T=1, C=1, Z=2) == 0)
    # A neighbouring plane must be untouched.
    assert np.all(
        reader.get_image_data("YX", T=1, C=1, Z=1) == plane_value("B02", 0, 1, 1, 1)
    )


def test_reads_without_a_manifest(tmp_path: Path) -> None:
    """Filename-only indexing must produce the same array as the CSV path."""
    with_csv = Reader(make_acquisition_unit(tmp_path / "with", write_csv=True))
    without_csv = Reader(make_acquisition_unit(tmp_path / "without", write_csv=False))

    assert without_csv.scenes == with_csv.scenes
    assert without_csv.dims.shape == with_csv.dims.shape
    assert np.array_equal(without_csv.xarray_data.data, with_csv.xarray_data.data)
    assert without_csv.xarray_dask_data.attrs["unprocessed"]["indexed_from"] == (
        "filenames"
    )


def test_manifest_absence_only_costs_timestamps(tmp_path: Path) -> None:
    reader = Reader(make_acquisition_unit(tmp_path / "without", write_csv=False))

    # Channel names and pixel sizes come from the descriptor, so they survive.
    assert reader.channel_names == ["TL", "FITC"]
    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)
    assert DimensionNames.Time not in reader.xarray_dask_data.coords


def test_sidecars_and_os_junk_are_ignored(acquisition_unit: Path) -> None:
    """The fixture writes .statistics.json and Thumbs.db beside every plane."""
    reader = Reader(acquisition_unit)

    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)


def test_a_manifest_row_without_its_file_reads_as_zeros(tmp_path: Path, caplog) -> None:
    """
    An aborted run leaves a manifest describing planes never written.

    Indexing is manifest-driven, so the plane is indexed and only found missing at
    read time: zeros plus a warning, rather than a silently shorter array. It does
    not appear in ``missing_planes``, which tracks planes the index has no path
    for at all.

    Read tile by tile, since the plane that went missing belongs to one of them.
    """
    unit = make_acquisition_unit(tmp_path / "aborted")
    (unit / "timepoint1" / "Fixture_t1_B03_s1_w1_z2.tif").unlink()

    reader = Reader(unit, mosaic=False)
    reader.set_scene("B03-s1")

    assert reader.dims.shape == (2, 2, 3, 32, 32)
    assert reader.xarray_dask_data.attrs["missing_planes"] == []

    with caplog.at_level(logging.WARNING):
        plane = reader.get_image_data("YX", T=1, C=1, Z=2)

    assert np.all(plane == 0)
    assert "Could not read plane" in caplog.text
    # A neighbouring plane must be untouched.
    assert np.all(
        reader.get_image_data("YX", T=1, C=1, Z=1) == plane_value("B03", 1, 1, 1, 1)
    )


def test_verify_planes_drops_manifest_rows_for_absent_files(tmp_path: Path) -> None:
    """
    ``verify_planes=True`` walks the directories and confirms every row, which is
    what a part-transferred copy wants: the plane is dropped from the index and
    reported, rather than read back as zeros.

    Read tile by tile, so the report is the (t, c, z) of the scene's own stack.
    """
    unit = make_acquisition_unit(tmp_path / "aborted")
    (unit / "timepoint1" / "Fixture_t1_B03_s1_w1_z2.tif").unlink()

    reader = Reader(unit, verify_planes=True, mosaic=False)
    reader.set_scene("B03-s1")

    assert reader.xarray_dask_data.attrs["missing_planes"] == [(1, 1, 2)]


###############################################################################
# bioio-base contract


def test_reader_satisfies_the_base_contract(acquisition_unit: Path) -> None:
    test_utilities.check_local_file_not_open(Reader(acquisition_unit))
    test_utilities.check_can_serialize_image_container(Reader(acquisition_unit))
