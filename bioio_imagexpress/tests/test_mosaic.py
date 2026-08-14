#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mosaic support: a well is one scene, its acquisition positions are its tiles.

MetaXpress images a well as a grid of overlapping fields and ships its own stitch
of exactly those as the ``experiment_montage`` unit, so the sites are mosaic tiles
rather than independent scenes. That is what the reader does by default, and this
module is where that default is pinned down: which scenes exist, where each tile
lands, and what the stitched well looks like. ``mosaic=False`` restores the older
one-scene-per-position reading, so the two are checked against each other here as
well -- they must describe the same planes.
"""

from pathlib import Path

import numpy as np
import pytest
from bioio_base import exceptions

from bioio_imagexpress import Reader

from .conftest import (
    PLANE_SIZE,
    make_acquisition_unit,
    plane_value,
    tile_pixel_offset,
)

###############################################################################

# The full 2x2 grid the `mosaic_unit` fixture acquires, in M order.
MOSAIC_SITES = (0, 1, 2, 3)

# Tiles overlap by 10%, and the stitch resolves the overlap last-tile-wins, so a
# tile's own corner may belong to its neighbour. Its centre never does.
TILE_CENTRE = PLANE_SIZE // 2


###############################################################################
# Scenes


def test_scenes_are_wells(mosaic_unit: Path):
    """The whole point of the default: four acquisition positions, one scene."""
    reader = Reader(mosaic_unit)

    assert reader.scenes == ("B02",)


def test_scenes_are_wells_and_sites_without_mosaic(mosaic_unit: Path):
    """With mosaic off each acquisition position is addressable on its own."""
    reader = Reader(mosaic_unit, mosaic=False)

    assert reader.scenes == ("B02-s0", "B02-s1", "B02-s2", "B02-s3")


def test_both_readings_cover_the_same_planes(acquisition_unit: Path):
    """
    Mosaic only regroups the planes, it does not choose different ones: tile M of
    the well scene has to be byte-identical to the scene that tile reads as with
    mosaic off. Guards against the M axis being built from the wrong sites.
    """
    well = Reader(acquisition_unit)
    tiles = Reader(acquisition_unit, mosaic=False)
    well.set_scene("B03")

    for site in range(2):
        tiles.set_scene(f"B03-s{site}")

        assert np.array_equal(
            well.get_image_data("TCZYX", M=site),
            tiles.get_image_data("TCZYX"),
        )


###############################################################################
# Dimensions


def test_tiles_stack_on_the_mosaic_dimension(mosaic_unit: Path):
    """M is the first axis and carries one entry per acquisition position."""
    reader = Reader(mosaic_unit)

    assert reader.dims.order == "MTCZYX"
    assert reader.dims.shape == (4, 1, 1, 1, 32, 32)
    assert reader.dims.M == len(MOSAIC_SITES)


def test_there_is_no_mosaic_dimension_without_mosaic(mosaic_unit: Path):
    """A scene is a single tile with mosaic off, so M would have nothing to hold."""
    reader = Reader(mosaic_unit, mosaic=False)

    assert reader.dims.order == "TCZYX"
    assert reader.dims.shape == (1, 1, 1, 32, 32)


def test_mosaic_tile_dims_report_one_tile(mosaic_unit: Path):
    """The stitched well is larger than a tile, so YX here must stay the tile's."""
    reader = Reader(mosaic_unit)

    tile_dims = reader.mosaic_tile_dims

    assert tile_dims.order == "YX"
    assert tile_dims.shape == (32, 32)


def test_mosaic_tile_dims_are_absent_without_mosaic(mosaic_unit: Path):
    """bioio-base reports None rather than raising when there are no tiles."""
    reader = Reader(mosaic_unit, mosaic=False)

    assert reader.mosaic_tile_dims is None


###############################################################################
# Tile positions


def test_tile_positions_form_the_acquisition_grid(mosaic_unit: Path):
    """
    The manifest carries absolute stage coordinates far from the origin, so the
    reader has to difference them against the top-left tile and divide by pixel
    size. At 32 px and 10% overlap that grid is one step of 29 px on each axis.
    """
    reader = Reader(mosaic_unit)

    positions = reader.get_mosaic_tile_positions()

    assert positions == [(0, 0), (29, 0), (29, 29), (0, 29)]
    # ...which is exactly where the fixture put the tiles.
    assert positions == [tile_pixel_offset(site, PLANE_SIZE) for site in MOSAIC_SITES]


def test_a_single_tile_position_indexes_the_full_list(mosaic_unit: Path):
    """The singular accessor must agree with the list for every tile, not just M=0."""
    reader = Reader(mosaic_unit)

    positions = reader.get_mosaic_tile_positions()

    assert [reader.get_mosaic_tile_position(i) for i in range(4)] == positions


def test_tile_positions_require_a_tile_dimension(mosaic_unit: Path):
    """With mosaic off the scene is one tile, so there is no grid to describe."""
    reader = Reader(mosaic_unit, mosaic=False)

    with pytest.raises(exceptions.UnexpectedShapeError):
        reader.get_mosaic_tile_positions()


###############################################################################
# Stitching


def test_the_stitched_well_spans_the_grid(mosaic_unit: Path):
    """
    Stitching drops M and grows YX to the grid's extent: the last tile starts at
    29 and is 32 px wide, so the well is 61 px on each axis rather than 64 -- the
    3 px of overlap is shared, not repeated.
    """
    reader = Reader(mosaic_unit)

    stitched = reader.mosaic_xarray_data

    assert stitched.dims == ("T", "C", "Z", "Y", "X")
    assert stitched.shape == (1, 1, 1, 61, 61)


def test_every_tile_lands_at_its_own_position(mosaic_unit: Path):
    """
    Shape alone would pass on a mosaic assembled in the wrong order, so each tile
    is identified by the value packed into its pixels. Probed at the tile centre:
    the outer 3 px of each tile are overlap, which a later tile overwrites.
    """
    reader = Reader(mosaic_unit)

    stitched = reader.mosaic_xarray_data

    for site, (top, left) in enumerate(reader.get_mosaic_tile_positions()):
        assert stitched.data[
            0, 0, 0, top + TILE_CENTRE, left + TILE_CENTRE
        ] == plane_value("B02", site, 0, 0, 0)


def test_delayed_and_immediate_mosaics_agree(mosaic_unit: Path):
    """Stitching is done twice, once per backend, so the two can drift apart."""
    reader = Reader(mosaic_unit)

    assert np.array_equal(
        reader.mosaic_xarray_dask_data.compute(),
        reader.mosaic_xarray_data,
    )


def test_stitching_requires_a_tile_dimension(mosaic_unit: Path):
    """
    Both stitched surfaces refuse a tile-less scene. bioio-base guards these two
    itself, and raises its own InvalidDimensionOrderingError rather than the
    UnexpectedShapeError this reader raises from get_mosaic_tile_positions.
    """
    reader = Reader(mosaic_unit, mosaic=False)

    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        reader.mosaic_xarray_data

    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        reader.mosaic_xarray_dask_data


###############################################################################
# Already-stitched wells


def test_a_single_site_well_is_its_own_mosaic(tmp_path: Path):
    """
    ``experiment_montage`` is MetaXpress's own stitch and images one position per
    well. It needs no stage metadata to place: a lone tile is the origin, and the
    stitched well is that tile unchanged.
    """
    unit = make_acquisition_unit(
        tmp_path / "montage",
        wells=["B02"],
        sites=[0],
        channels=["TL"],
        t_count=1,
        z_count=1,
    )

    reader = Reader(unit)

    assert reader.dims.M == 1
    assert reader.get_mosaic_tile_positions() == [(0, 0)]
    assert reader.mosaic_xarray_data.shape == (1, 1, 1, 32, 32)
    assert np.array_equal(reader.mosaic_data, reader.data[0])


###############################################################################
# Resolution levels


def test_tile_placement_scales_with_resolution_level(mosaic_unit: Path):
    """
    Positions are pixel offsets, so they only stay right if they are recomputed
    against the level's pixel size: at level 1 both the step and the stitched well
    are half of what level 0 reports. 29 -> 14 rather than 15, since the offset is
    rounded from the stage distance and not from the level 0 offset.
    """
    reader = Reader(mosaic_unit)
    reader.set_resolution_level(1)

    assert reader.get_mosaic_tile_positions() == [(0, 0), (14, 0), (14, 14), (0, 14)]

    stitched = reader.mosaic_xarray_data

    assert stitched.shape == (1, 1, 1, 30, 30)
    # The downsampled planes keep their packed value, so tile identity still holds.
    for site, (top, left) in enumerate(reader.get_mosaic_tile_positions()):
        assert stitched.data[0, 0, 0, top + 8, left + 8] == plane_value(
            "B02", site, 0, 0, 0
        )


###############################################################################
# Acquisitions without stage positions


def test_a_mosaic_without_stage_positions_says_how_to_read_it(tmp_path: Path):
    """
    Filename-only indexing loses the stage coordinates, and nothing else in the
    acquisition says where a site sat, so a multi-tile well cannot be placed. The
    error has to name the way out, since ``mosaic=False`` reads it fine.
    """
    unit = make_acquisition_unit(
        tmp_path / "no_manifest",
        wells=["B02"],
        sites=[0, 1],
        channels=["TL"],
        t_count=1,
        z_count=1,
        write_csv=False,
    )

    reader = Reader(unit)

    with pytest.raises(ValueError) as error:
        reader.get_mosaic_tile_positions()

    assert "mosaic=False" in str(error.value)


def test_tiles_still_read_without_stage_positions(tmp_path: Path):
    """
    Only the placement is lost. The tiles themselves are indexed from filenames,
    so M still carries them in site order and the pixels still come back.
    """
    unit = make_acquisition_unit(
        tmp_path / "no_manifest",
        wells=["B02"],
        sites=[0, 1],
        channels=["TL"],
        t_count=1,
        z_count=1,
        write_csv=False,
    )

    reader = Reader(unit)

    assert reader.dims.shape == (2, 1, 1, 1, 32, 32)
    for site in range(2):
        assert np.all(
            reader.get_image_data("YX", M=site, T=0, C=0, Z=0)
            == plane_value("B02", site, 0, 0, 0)
        )


###############################################################################
# Scene changes


def test_a_scene_change_re_resolves_tile_positions(run_root: Path):
    """
    A run root mixes a two-tile unit with an already-stitched one at a different
    pixel size, so a tile layout cached across a scene change would place the
    wrong grid -- and it would still have a plausible shape.
    """
    reader = Reader(run_root)

    def layout(scene_id: str):
        reader.set_scene(scene_id)
        return (
            reader.get_mosaic_tile_positions(),
            reader.mosaic_xarray_data.shape,
        )

    experiment = ([(0, 0), (29, 0)], (2, 2, 3, 61, 32))
    montage = ([(0, 0)], (1, 1, 1, 16, 16))

    assert layout("experiment/B02") == experiment
    assert layout("experiment_montage/B02") == montage
    # And in the other order, to catch state that only leaks one way.
    assert layout("experiment/B03") == experiment
    assert layout("experiment_montage/B03") == montage
