#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests against real acquisitions.

These reference ~312 GB of data on the Allen network mount, so they skip cleanly
wherever it is not available -- CI included. They exist to catch drift between the
synthetic fixtures and what MetaXpress actually writes.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bioio_imagexpress import Reader

###############################################################################

ALLEN = Path(
    os.environ.get("BIOIO_IMAGEXPRESS_TEST_ROOT", "/Users/brian.whitney/allen")
)

Z_STACK = (
    ALLEN
    / "aics/lumenoid/crispri/CRISPRi_parallel_synthesis/ImageXpress"
    / "3500009044_zstacks/experiment_z_stack"
)

RUN_ROOT = (
    ALLEN
    / "aics/microscopes/Antoine/Transfer ImageXpress to other scopes"
    / "4X_Lumenoid_Cellvis_96_8-7-2026_20260807_150709"
)


ALLEN_HTTPS = os.environ.get(
    "BIOIO_IMAGEXPRESS_TEST_URL", "https://vast-files.int.allencell.org"
)

# The mount and the endpoint serve the same tree: <mount>/aics/FOO == <url>/FOO.
Z_STACK_URL = (
    f"{ALLEN_HTTPS}/lumenoid/crispri/CRISPRi_parallel_synthesis/ImageXpress"
    "/3500009044_zstacks/experiment_z_stack"
)
Z_STACK_DESCRIPTOR = (
    "3500009044_20260806_iCRISPRi_parallel_synthesis_trial_lumenoid_growth_TB"
    "_z_stack.jdce"
)


def _require_url(url: str) -> str:
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("aiohttp")

    try:
        if not fsspec.filesystem("https").exists(url):
            pytest.skip(f"Reference acquisition not reachable: {url}")
    except Exception as exc:  # off the Allen network
        pytest.skip(f"Reference endpoint unavailable ({exc})")

    return url


def _require(path: Path) -> Path:
    try:
        available = path.is_dir()
    except OSError:  # the mount is present but unresponsive
        available = False

    if not available:
        pytest.skip(f"Reference acquisition not available: {path}")

    return path


###############################################################################


@pytest.mark.integration
def test_z_stack_acquisition() -> None:
    reader = Reader(_require(Z_STACK))

    # 30 wells, each a 2x2 mosaic of 4 tiles, 9 time points x 2 channels x 11 z.
    assert len(reader.scenes) == 30
    assert reader.scenes[0] == "B07"
    assert reader.dims.shape == (4, 9, 2, 11, 2304, 2304)
    assert reader.dims.order == "MTCZYX"
    assert reader.dtype == np.uint16
    assert reader.channel_names == ["TL", "FITC"]
    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)
    assert reader.resolution_levels == (0, 1, 2)
    assert reader.xarray_dask_data.attrs["missing_planes"] == []


@pytest.mark.integration
def test_z_stack_time_coords_are_six_hourly() -> None:
    reader = Reader(_require(Z_STACK))

    coords = reader.xarray_dask_data.coords["T"].values

    assert len(coords) == 9
    assert coords[0] == 0.0
    assert np.allclose(np.diff(coords), 21600, atol=5)


@pytest.mark.integration
def test_z_stack_plane_matches_the_source_tiff() -> None:
    tifffile = pytest.importorskip("tifffile")
    root = _require(Z_STACK)

    reader = Reader(root)
    plane = reader.get_image_data("YX", T=3, C=1, Z=5)
    expected = tifffile.imread(
        root / "timepoint3" / "Wellscan_96well_TL_488_t3_B07_s0_w1_z5.tif"
    )

    assert np.array_equal(plane, expected)


@pytest.mark.integration
def test_run_root_exposes_both_units() -> None:
    reader = Reader(_require(RUN_ROOT))

    # 60 wells of `experiment`, plus the same 60 wells already stitched.
    assert len(reader.scenes) == 120
    assert reader.scenes[0] == "experiment/B02"

    units = {scene.split("/")[0] for scene in reader.scenes}
    assert units == {"experiment", "experiment_montage"}

    # The `autofocus` folder holds no planes and must not become a unit.
    assert "autofocus" not in units


@pytest.mark.integration
def test_run_root_units_keep_distinct_shapes() -> None:
    reader = Reader(_require(RUN_ROOT))

    reader.set_scene("experiment/B02")
    assert reader.dims.shape == (4, 1, 1, 1, 2304, 2304)
    assert reader.physical_pixel_sizes.X == 1.6595

    # The montage unit is already stitched, so its well is a single tile that is
    # bigger than one of `experiment`'s -- and differs in size well to well.
    reader.set_scene("experiment_montage/B02")
    assert reader.dims.shape[:4] == (1, 1, 1, 1)
    assert reader.dims.Y > 4000 and reader.dims.X > 4000


@pytest.mark.integration
def test_z_stack_standard_metadata() -> None:
    reader = Reader(_require(Z_STACK))

    metadata = reader.standard_metadata

    assert metadata.binning == "1x1"
    assert (metadata.row, metadata.column) == ("B", "7")
    # The scene is the whole well, not one of its acquisition positions.
    assert metadata.position_index is None
    assert metadata.objective == "10X Plan Apo Lambda D"
    assert metadata.imaged_by == "moldev"
    # The descriptor's own stamp, so instrument local time rather than the
    # manifest's Unix epoch.
    assert metadata.imaging_datetime.tzinfo is None
    assert metadata.imaging_datetime.year == 2026
    assert metadata.stage_position_x == pytest.approx(67696.78)
    assert metadata.timelapse is True
    # 9 time points, six-hourly.
    assert metadata.timelapse_interval.total_seconds() == pytest.approx(21600, abs=5)
    assert metadata.total_time_duration.total_seconds() == pytest.approx(
        8 * 21600, abs=40
    )
    assert reader.metadata["indexed_from"] == "image_metadata_csv"


@pytest.mark.integration
def test_z_stack_region_read_opens_only_the_named_planes() -> None:
    """
    The whole point of `_read_indexed`: one plane of a 198 plane scene must cost
    one file open, not 198. Without the seam this read takes ~7 minutes.
    """
    import time

    from bioio_imagexpress import reader as reader_module

    root = _require(Z_STACK)
    reader = Reader(root)
    reader.dims  # the shape probe reads one representative plane; do it up front

    opened = []
    original = reader_module._read_plane

    def spy(fs, path, shape, dtype, level):
        opened.append(path)
        return original(fs, path, shape, dtype, level)

    reader_module._read_plane = spy
    try:
        start = time.monotonic()
        plane = reader.get_image_data("YX", T=3, C=1, Z=5)
        elapsed = time.monotonic() - start
    finally:
        reader_module._read_plane = original

    assert plane.shape == (2304, 2304)
    assert len(opened) == 1
    assert opened[0].endswith("Wellscan_96well_TL_488_t3_B07_s0_w1_z5.tif")
    assert elapsed < 30, f"Single-plane read took {elapsed:.1f}s"


@pytest.mark.integration
def test_z_stack_region_read_matches_the_source_tiffs() -> None:
    """A multi-plane selection must land each file at the right coordinate."""
    tifffile = pytest.importorskip("tifffile")
    root = _require(Z_STACK)

    reader = Reader(root)
    data = reader.get_image_data("CZYX", T=3, C=[0, 1], Z=[5, 7])

    assert data.shape == (2, 2, 2304, 2304)
    for channel in (0, 1):
        for position, z in enumerate((5, 7)):
            expected = tifffile.imread(
                root
                / "timepoint3"
                / f"Wellscan_96well_TL_488_t3_B07_s0_w{channel}_z{z}.tif"
            )
            assert np.array_equal(data[channel, position], expected)


###############################################################################
# Remote, over plain HTTPS


@pytest.mark.integration
def test_z_stack_over_https_indexes_without_listing() -> None:
    """
    The endpoint serves files but returns 403 for directory listings, so this only
    works because indexing is manifest-driven. Naming the descriptor is the way
    in: it names the manifest, and the manifest names every plane.
    """
    url = _require_url(f"{Z_STACK_URL}/{Z_STACK_DESCRIPTOR}")

    reader = Reader(url)

    assert len(reader.scenes) == 30
    assert reader.scenes[0] == "B07"
    assert reader.dims.shape == (4, 9, 2, 11, 2304, 2304)
    assert reader.channel_names == ["TL", "FITC"]
    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)
    assert reader.metadata["indexed_from"] == "image_metadata_csv"


@pytest.mark.integration
def test_z_stack_over_https_standard_metadata_matches_the_mount() -> None:
    """The transport must not change a single field."""
    url = _require_url(f"{Z_STACK_URL}/{Z_STACK_DESCRIPTOR}")

    assert Reader(url).standard_metadata == Reader(_require(Z_STACK)).standard_metadata


@pytest.mark.integration
def test_z_stack_over_https_plane_matches_the_mount() -> None:
    tifffile = pytest.importorskip("tifffile")
    url = _require_url(f"{Z_STACK_URL}/{Z_STACK_DESCRIPTOR}")
    root = _require(Z_STACK)

    plane = Reader(url).get_image_data("YX", T=3, C=1, Z=5)
    expected = tifffile.imread(
        root / "timepoint3" / "Wellscan_96well_TL_488_t3_B07_s0_w1_z5.tif"
    )

    assert np.array_equal(plane, expected)


@pytest.mark.integration
def test_z_stack_over_https_directory_form_is_rejected_clearly() -> None:
    """
    Pointing at the directory cannot work here -- the descriptor's name is
    unknowable without a listing. The error must say what to do instead.
    """
    from bioio_base import exceptions

    _require_url(f"{Z_STACK_URL}/{Z_STACK_DESCRIPTOR}")

    with pytest.raises(
        (exceptions.UnsupportedFileFormatError, FileNotFoundError)
    ) as raised:
        Reader(Z_STACK_URL)

    if isinstance(raised.value, exceptions.UnsupportedFileFormatError):
        assert "name the '.jdce' descriptor directly" in str(raised.value)


@pytest.mark.integration
def test_indexing_a_large_acquisition_opens_no_tiffs() -> None:
    """
    Construction must stay cheap: 306 GB and 23,760 planes should cost directory
    listings and one CSV parse, not 23,760 file opens.
    """
    import time

    root = _require(Z_STACK)

    start = time.monotonic()
    reader = Reader(root)
    scenes = reader.scenes
    elapsed = time.monotonic() - start

    assert len(scenes) == 30
    assert elapsed < 120, f"Indexing took {elapsed:.1f}s; expected well under 2 min"
