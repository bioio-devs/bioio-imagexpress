#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The ``_read_indexed`` seam: ``get_image_data`` must open only the files it needs.

Every plane is a separate file here, so a sub-region read that materializes the
whole scene first is not a minor inefficiency -- it is the difference between one
file and the couple of hundred in a scene.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
from bioio_base import transforms
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER,
    DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
)

from bioio_imagexpress import Reader
from bioio_imagexpress import reader as reader_module

from .conftest import make_acquisition_unit, plane_value

###############################################################################


@pytest.fixture
def opened_planes(monkeypatch) -> List[str]:
    """Records the plane files each read actually opens."""
    opened: List[str] = []
    original = reader_module._read_plane

    def spy(fs, path, shape, dtype, level):
        opened.append(path)
        return original(fs, path, shape, dtype, level)

    monkeypatch.setattr(reader_module, "_read_plane", spy)

    return opened


###############################################################################
# Equivalence with the unoptimized path


SELECTIONS = [
    ("YX", {"T": 1, "C": 0, "Z": 2}),
    ("YX", {}),
    ("ZYX", {"T": 0, "C": 1}),
    ("TCZYX", {}),
    ("CZYX", {"T": 1, "C": [0, 1]}),
    ("TZYX", {"T": slice(0, 2), "Z": 1}),
    ("TCZYX", {"Z": slice(0, 3, 2)}),
    ("TCZYX", {"T": range(2)}),
    ("CYX", {"T": 0, "Z": 0, "C": (0, 1)}),
    # Non-contiguous, so this stays a list rather than reducing to a slice.
    ("TCZYX", {"Z": [0, 2]}),
    # Spatial crops are applied after each plane is read.
    ("YX", {"T": 0, "C": 0, "Z": 0, "Y": slice(4, 20), "X": slice(8, 12)}),
    # Bounded rather than open-ended: bioio-base's compute_dim_specs takes
    # abs(slice.start) while validating, so slice(None, ...) raises there --
    # on the unoptimized path too, so it is not this seam's to fix.
    ("ZYX", {"T": 1, "C": 1, "Y": slice(0, 32, 4)}),
    # Dimension shuffles must survive the seam.
    ("XYZ", {"T": 0, "C": 0}),
    ("ZCT", {"Y": 0, "X": 0}),
    # M is the tile dimension with mosaic on; with mosaic off it is a dimension
    # the data does not have, and must still be padded in.
    ("MTCZYX", {}),
]

# Selections that only mean something once the tiles of a well share a scene.
MOSAIC_SELECTIONS = SELECTIONS + [
    ("MTCZYX", {"M": 1}),
    # An integer tile spec drops M, exactly as an integer T or Z drops its axis.
    ("TCZYX", {"M": 1}),
    ("MYX", {"T": 1, "C": 0, "Z": 2}),
    ("MCZYX", {"M": [0, 1], "T": 0}),
]


@pytest.mark.parametrize("order_out, selection", MOSAIC_SELECTIONS)
def test_indexed_read_matches_a_full_read(acquisition_unit: Path, order_out, selection) -> None:
    """
    The seam is an optimization, so it must be indistinguishable from reading the
    whole scene and slicing it.

    With mosaic on the scene is a whole well, so the seam has a tile dimension to
    resolve as well as T, C and Z.
    """
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")

    indexed = reader.get_image_data(order_out, **selection)
    expected = transforms.reshape_data(
        data=reader.xarray_data.data,
        given_dims=DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
        return_dims=order_out,
        **selection,
    )

    assert indexed.shape == expected.shape
    assert indexed.dtype == expected.dtype
    assert np.array_equal(indexed, expected)


@pytest.mark.parametrize("order_out, selection", SELECTIONS)
def test_indexed_read_matches_a_full_read_without_mosaic(
    acquisition_unit: Path, order_out, selection
) -> None:
    """
    The same equivalence for a single-tile scene, which is what ``mosaic=False``
    gives: no M on the data, so the seam resolves T, C and Z alone.
    """
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")

    indexed = reader.get_image_data(order_out, **selection)
    expected = transforms.reshape_data(
        data=reader.xarray_data.data,
        given_dims=DEFAULT_DIMENSION_ORDER,
        return_dims=order_out,
        **selection,
    )

    assert indexed.shape == expected.shape
    assert indexed.dtype == expected.dtype
    assert np.array_equal(indexed, expected)


def test_indexed_read_lands_on_the_right_planes(acquisition_unit: Path) -> None:
    """
    Shape equality would not catch a transposed T and C -- or a tile stacked at
    the wrong M -- so check the values.
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


def test_indexed_read_lands_on_the_right_planes_without_mosaic(acquisition_unit: Path) -> None:
    """
    The same value check for a single-tile scene: with mosaic off the site comes
    from the scene id rather than from an M spec.
    """
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")

    data = reader.get_image_data("TCZYX")

    for t in range(2):
        for channel in range(2):
            for z in range(3):
                assert data[t, channel, z, 0, 0] == plane_value("B03", 1, t, channel, z)


###############################################################################
# What actually gets opened


def test_a_single_plane_opens_a_single_file(
    acquisition_unit: Path, opened_planes: List[str]
) -> None:
    """Naming one tile of the well must not drag in the well's other tiles."""
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")
    opened_planes.clear()  # the shape probe reads one representative plane

    plane = reader.get_image_data("YX", M=1, T=1, C=0, Z=2)

    assert len(opened_planes) == 1
    assert opened_planes[0].endswith("Fixture_t1_B03_s1_w0_z2.tif")
    assert np.all(plane == plane_value("B03", 1, 1, 0, 2))


def test_a_single_plane_opens_a_single_file_without_mosaic(
    acquisition_unit: Path, opened_planes: List[str]
) -> None:
    """The same, for the scene-per-site layout where the tile is the scene."""
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")
    opened_planes.clear()  # the shape probe reads one representative plane

    plane = reader.get_image_data("YX", T=1, C=0, Z=2)

    assert len(opened_planes) == 1
    assert opened_planes[0].endswith("Fixture_t1_B03_s1_w0_z2.tif")
    assert np.all(plane == plane_value("B03", 1, 1, 0, 2))


def test_only_the_selected_planes_are_opened(
    acquisition_unit: Path, opened_planes: List[str]
) -> None:
    """A 2x2x2x3 well scene holds 24 planes; this selection names 8 of them."""
    reader = Reader(acquisition_unit)
    opened_planes.clear()

    reader.get_image_data("MCZYX", T=0, C=[0, 1], Z=[0, 2])

    assert len(opened_planes) == 8
    assert all("_t0_" in path for path in opened_planes)
    assert not any("_z1." in path for path in opened_planes)


def test_a_spatial_crop_still_opens_only_one_plane(
    acquisition_unit: Path, opened_planes: List[str]
) -> None:
    """
    A MetaXpress plane has no tiling to exploit, so the crop happens in memory --
    but it must not drag in any other plane.
    """
    reader = Reader(acquisition_unit)
    opened_planes.clear()

    region = reader.get_image_data("YX", T=0, C=0, Z=0, Y=slice(0, 8), X=slice(0, 8))

    assert len(opened_planes) == 1
    assert region.shape == (8, 8)


def test_a_full_read_opens_every_plane(
    acquisition_unit: Path, opened_planes: List[str]
) -> None:
    """The seam must not accidentally drop planes from an unrestricted read."""
    reader = Reader(acquisition_unit)
    opened_planes.clear()

    reader.get_image_data("MTCZYX")

    assert len(opened_planes) == 2 * 2 * 2 * 3


###############################################################################
# Interaction with the rest of the reader


def test_indexed_read_respects_the_resolution_level(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)
    reader.set_resolution_level(1)

    plane = reader.get_image_data("YX", T=1, C=0, Z=2)

    assert plane.shape == (16, 16)
    assert np.all(plane == plane_value("B02", 0, 1, 0, 2))


def test_indexed_read_follows_the_current_scene(run_root: Path) -> None:
    reader = Reader(run_root)

    reader.set_scene("experiment_montage/B03")
    montage = reader.get_image_data("YX", T=0, C=0, Z=0)
    reader.set_scene("experiment/B02")
    experiment = reader.get_image_data("YX", T=0, C=0, Z=0)

    assert montage.shape == (16, 16)
    assert experiment.shape == (32, 32)
    assert montage[0, 0] == plane_value("B03", 0, 0, 0, 0)
    assert experiment[0, 0] == plane_value("B02", 0, 0, 0, 0)


def test_indexed_read_zero_fills_a_missing_plane(tmp_path: Path) -> None:
    unit = make_acquisition_unit(
        tmp_path / "ragged",
        skip_planes=[("B02", 0, 1, 1, 2)],
    )
    reader = Reader(unit)
    reader.set_scene("B02")

    assert np.all(reader.get_image_data("YX", T=1, C=1, Z=2) == 0)
    # A neighbouring plane must be untouched.
    assert np.all(
        reader.get_image_data("YX", T=1, C=1, Z=1) == plane_value("B02", 0, 1, 1, 1)
    )
    # The hole is one tile's, so the same coordinate on the next tile survives.
    assert np.all(
        reader.get_image_data("YX", M=1, T=1, C=1, Z=2)
        == plane_value("B02", 1, 1, 1, 2)
    )


def test_indexed_read_handles_sparse_coordinates(tmp_path: Path) -> None:
    """
    An aborted acquisition leaves gaps, so the on-disk Z values need not be
    ``0..n``. Specs index the coordinates that exist, not raw plane numbers.
    """
    unit = make_acquisition_unit(
        tmp_path / "sparse",
        wells=["B02"],
        sites=[0],
        channels=["TL"],
        t_count=1,
        skip_planes=[("B02", 0, 0, 0, 1)],
    )
    reader = Reader(unit)

    # Z=1 on disk is gone, so the second surviving coordinate is z2. The leading
    # 1 is M: the well was imaged at a single site, so its mosaic is one tile.
    assert reader.dims.shape == (1, 1, 1, 2, 32, 32)
    assert np.all(
        reader.get_image_data("YX", T=0, C=0, Z=1) == plane_value("B02", 0, 0, 0, 2)
    )


def test_indexed_read_of_an_empty_selection(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    empty = reader.get_image_data("TCZYX", Z=slice(0, 0))

    assert empty.shape == (2, 2, 0, 32, 32)


def test_dask_path_is_unaffected(acquisition_unit: Path) -> None:
    """``get_image_dask_data`` slices the graph and so never needed the seam."""
    reader = Reader(acquisition_unit)

    lazy = reader.get_image_dask_data("YX", T=1, C=0, Z=2)

    assert lazy.shape == (32, 32)
    assert np.array_equal(lazy.compute(), reader.get_image_data("YX", T=1, C=0, Z=2))
