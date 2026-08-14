#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests against committed fixtures cropped from the real acquisitions.

The synthetic fixtures in ``conftest.py`` control every variable and are the right
place to test edge cases. These resources are the opposite: unedited MetaXpress
output (real TIFF tags, real MetaSeries XML, real ``.jdce``, real CSV) trimmed to
a few hundred KB, so they catch assumptions that the synthetic writer happens to
share with the reader.

Regenerate with ``python scripts/make_test_resources.py``.
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from bioio_base import test_utilities

from bioio_imagexpress import Reader

###############################################################################

RESOURCES = Path(__file__).parent / "resources"
Z_STACK = RESOURCES / "experiment_z_stack"
RUN_ROOT = RESOURCES / "run_root"

###############################################################################


def test_reads_a_real_z_stack_acquisition() -> None:
    """
    By default the sites of a well are treated as mosaic tiles, so a scene is a
    whole well and the two sites of the reference acquisition become the M axis.
    """
    reader = Reader(Z_STACK)

    assert reader.scenes == ("B07", "B08")
    assert reader.dims.order == "MTCZYX"
    assert reader.dims.shape == (2, 2, 2, 3, 64, 64)
    assert reader.dtype == np.uint16
    assert reader.xarray_dask_data.attrs["missing_planes"] == []


def test_reads_a_real_z_stack_acquisition_without_mosaic() -> None:
    """Same acquisition with ``mosaic=False``: one scene per (well, site)."""
    reader = Reader(Z_STACK, mosaic=False)

    assert reader.scenes == ("B07-s0", "B07-s1", "B08-s0", "B08-s1")
    assert reader.dims.order == "TCZYX"
    assert reader.dims.shape == (2, 2, 3, 64, 64)
    assert reader.dtype == np.uint16
    assert reader.xarray_dask_data.attrs["missing_planes"] == []


def test_channel_names_from_a_real_descriptor() -> None:
    reader = Reader(Z_STACK)

    assert reader.channel_names == ["TL", "FITC"]


def test_pixel_sizes_from_a_real_descriptor() -> None:
    """
    The 10x objective calibration and 3 um Z step of the reference acquisition.
    Note the TIFF resolution tags are unset in real output, so these can only come
    from the descriptor.
    """
    reader = Reader(Z_STACK)

    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)


def test_time_coords_from_a_real_manifest() -> None:
    reader = Reader(Z_STACK)

    coords = reader.xarray_dask_data.coords["T"].values

    assert coords[0] == 0.0
    assert coords[1] == pytest.approx(21599.75, abs=1.0)


def test_real_planes_carry_metaseries_metadata() -> None:
    reader = Reader(Z_STACK)

    metaseries = reader.xarray_dask_data.attrs["unprocessed"]["metaseries"]

    assert metaseries["ApplicationName"] == "MetaXpress Acquire"
    assert metaseries["PlaneInfo"]["spatial-calibration-x"] == 0.5817


def test_real_stage_positions_are_captured() -> None:
    reader = Reader(Z_STACK)

    metadata = reader.xarray_dask_data.attrs["unprocessed"]

    assert metadata["well"] == "B07"
    assert metadata["barcode"] == "3500009044"
    assert metadata["objective"] == "10X Plan Apo Lambda D"
    assert metadata["stage_position_um"]["x"] == pytest.approx(67696.78)


def test_pixels_match_the_source_tiff() -> None:
    """
    Reads one acquisition position, so it runs with ``mosaic=False`` to name the
    site explicitly instead of leaning on the M axis defaulting to tile 0.
    """
    reader = Reader(Z_STACK, mosaic=False)

    plane = reader.get_image_data("YX", T=1, C=1, Z=2)
    expected = tifffile.imread(
        Z_STACK / "timepoint1" / "Wellscan_96well_TL_488_t1_B07_s0_w1_z2.tif"
    )

    assert np.array_equal(plane, expected)


def test_real_planes_expose_their_pyramid() -> None:
    reader = Reader(Z_STACK)

    assert reader.resolution_levels == (0, 1, 2)

    reader.set_resolution_level(1)
    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.physical_pixel_sizes == pytest.approx((3.0, 1.1634, 1.1634))


def test_sidecars_and_thumbs_db_are_skipped() -> None:
    """Both are present in the fixture, exactly as MetaXpress leaves them."""
    assert (Z_STACK / "timepoint0" / "Thumbs.db").exists()
    assert list(Z_STACK.glob("timepoint0/*.statistics.json"))

    assert Reader(Z_STACK).dims.shape == (2, 2, 2, 3, 64, 64)


###############################################################################


def test_real_run_root_exposes_both_units() -> None:
    """With mosaic on, each unit contributes one scene per well."""
    reader = Reader(RUN_ROOT)

    assert reader.scenes == (
        "experiment/B02",
        "experiment/B03",
        "experiment_montage/B02",
        "experiment_montage/B03",
    )


def test_real_run_root_exposes_both_units_without_mosaic() -> None:
    """
    With ``mosaic=False`` the sites stay separate scenes, which is what shows that
    the montage unit holds a single stitched site per well while the experiment it
    was stitched from holds two.
    """
    reader = Reader(RUN_ROOT, mosaic=False)

    assert reader.scenes == (
        "experiment/B02-s0",
        "experiment/B02-s1",
        "experiment/B03-s0",
        "experiment/B03-s1",
        "experiment_montage/B02-s0",
        "experiment_montage/B03-s0",
    )


def test_real_autofocus_folder_is_not_a_unit() -> None:
    assert (RUN_ROOT / "autofocus").is_dir()

    assert not any(s.startswith("autofocus") for s in Reader(RUN_ROOT).scenes)


def test_real_units_keep_distinct_pixel_sizes() -> None:
    """The 4x experiment and its montage differ from the 10x z-stack."""
    reader = Reader(RUN_ROOT)

    reader.set_scene("experiment/B02")
    assert reader.physical_pixel_sizes.X == 1.6595
    assert reader.channel_names == ["TL"]
    assert reader.dims.shape == (2, 1, 1, 1, 64, 64)


def test_accepts_a_real_jdce_file() -> None:
    descriptor = next(Z_STACK.glob("*.jdce"))

    assert Reader(descriptor).scenes == ("B07", "B08")


def test_accepts_a_real_jdce_file_without_mosaic() -> None:
    descriptor = next(Z_STACK.glob("*.jdce"))

    assert Reader(descriptor, mosaic=False).scenes == (
        "B07-s0",
        "B07-s1",
        "B08-s0",
        "B08-s1",
    )


def test_real_acquisition_satisfies_the_base_contract() -> None:
    test_utilities.check_local_file_not_open(Reader(Z_STACK))
    test_utilities.check_can_serialize_image_container(Reader(Z_STACK))
