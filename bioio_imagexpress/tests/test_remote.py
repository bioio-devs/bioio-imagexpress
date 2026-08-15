#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pytest
from fsspec.implementations.http import HTTPFileSystem

from bioio_imagexpress import Reader

from .conftest import LOCAL_RESOURCES_DIR, descriptor


@pytest.mark.parametrize(
    "unit, expected_scenes, expected_shape, expected_channels",
    [
        ("experiment_z_stack", ("B07", "B08"), (2, 2, 2, 3, 64, 64), ["TL", "FITC"]),
        ("run_root/experiment", ("B02", "B03"), (2, 1, 1, 1, 64, 64), ["TL"]),
    ],
)
def test_reader_over_http_matches_local(
    local_http_server: str,
    unit: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_channels: List[str],
) -> None:
    # Named by its descriptor, an acquisition served over http has to resolve to
    # the same image as the directory read off disk -- pixels and metadata both.
    name = descriptor(LOCAL_RESOURCES_DIR / unit).name
    local = Reader(LOCAL_RESOURCES_DIR / unit / name)
    over_http = Reader(f"{local_http_server}/{unit}/{name}")

    assert over_http.scenes == local.scenes == expected_scenes
    assert over_http.dims.order == local.dims.order == "MTCZYX"
    assert over_http.shape == local.shape == expected_shape
    assert over_http.dtype == local.dtype == np.dtype("uint16")
    assert over_http.channel_names == local.channel_names == expected_channels
    assert over_http.physical_pixel_sizes == local.physical_pixel_sizes
    np.testing.assert_array_equal(over_http.data, local.data)
    # The delayed path opens each plane from inside the dask graph, so it reaches
    # the server independently of the eager path above.
    np.testing.assert_array_equal(over_http.dask_data.compute(), local.data)


def test_reader_over_http_places_mosaic_tiles(local_http_server: str) -> None:
    # Tiles are placed from the manifest's stage positions, which are read over
    # http like everything else.
    unit = "run_root/experiment"
    name = descriptor(LOCAL_RESOURCES_DIR / unit).name
    local = Reader(LOCAL_RESOURCES_DIR / unit / name)
    over_http = Reader(f"{local_http_server}/{unit}/{name}")

    assert over_http.get_mosaic_tile_positions() == [(0, 0), (2074, 0)]
    assert over_http.get_mosaic_tile_positions() == local.get_mosaic_tile_positions()
    assert over_http.get_mosaic_tile_position(1) == (2074, 0)


def test_reader_over_http_reads_one_scene_per_site(local_http_server: str) -> None:
    unit = "run_root/experiment"
    name = descriptor(LOCAL_RESOURCES_DIR / unit).name
    url = f"{local_http_server}/{unit}/{name}"
    local = Reader(LOCAL_RESOURCES_DIR / unit / name, mosaic=False)
    over_http = Reader(url, mosaic=False)
    over_http.set_scene("B03-s1")
    local.set_scene("B03-s1")

    assert over_http.scenes == ("B02-s0", "B02-s1", "B03-s0", "B03-s1")
    assert over_http.dims.order == "TCZYX"
    assert over_http.shape == (1, 1, 1, 64, 64)
    assert over_http.position_index == 1
    np.testing.assert_array_equal(over_http.data, local.data)


def test_server_refuses_directory_listing(local_http_server: str) -> None:
    # The point of the '.jdce' entry point: this server serves files but 404s
    # every directory, so nothing above could have come from a listing.
    unit = "experiment_z_stack"
    directory = f"{local_http_server}/{unit}"

    with pytest.raises(FileNotFoundError):
        HTTPFileSystem().ls(directory)

    # A directory is not a fetchable resource here, so it cannot be opened at all.
    # The descriptor inside it reads fine, which is the contrast being drawn.
    with pytest.raises(FileNotFoundError):
        Reader(directory)

    name = descriptor(LOCAL_RESOURCES_DIR / unit).name

    assert Reader(f"{directory}/{name}").scenes == ("B07", "B08")
