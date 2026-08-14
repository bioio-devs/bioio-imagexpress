#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plugin-level metadata and how bioio routes to this reader."""

from pathlib import Path

import pytest
from bioio_base import exceptions

from bioio_imagexpress import Reader, ReaderMetadata

###############################################################################


def test_supported_extensions() -> None:
    assert ReaderMetadata.get_supported_extensions() == ["jdce"]


def test_get_reader_returns_the_reader() -> None:
    assert ReaderMetadata.get_reader() is Reader


###############################################################################


def test_bioimage_routes_a_descriptor_on_its_own(acquisition_unit: Path) -> None:
    """
    The primary entry point. `bioio` matches plugins on the path suffix, so
    claiming `.jdce` is what lets it pick this reader without being told.

    Mosaic is on by default, so a scene is a whole well and what `bioio` hands
    back is the stitch of its tiles. The fixture's two sites are one column of
    the site grid at 10% overlap, so Y grows from 32 to 32 + 29 while the
    reader's own tiles stay 32x32 on a leading M axis.
    """
    BioImage = pytest.importorskip("bioio").BioImage
    descriptor = next(acquisition_unit.glob("*.jdce"))

    image = BioImage(descriptor)

    assert isinstance(image.reader, Reader)
    assert image.dims.order == "TCZYX"
    assert image.dims.shape == (2, 2, 3, 61, 32)
    assert image.reader.dims.order == "MTCZYX"
    assert image.reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert image.channel_names == ["TL", "FITC"]
    assert image.scenes == ("B02", "B03")


def test_bioimage_routes_a_descriptor_with_mosaic_off(acquisition_unit: Path) -> None:
    """
    The same routing, with the tiles left apart. `mosaic=False` has to survive
    the trip through `bioio` down to the reader, and then a scene is one well
    and site again and nothing is stitched.
    """
    BioImage = pytest.importorskip("bioio").BioImage
    descriptor = next(acquisition_unit.glob("*.jdce"))

    image = BioImage(descriptor, mosaic=False)

    assert isinstance(image.reader, Reader)
    assert image.dims.order == "TCZYX"
    assert image.dims.shape == (2, 2, 3, 32, 32)
    assert image.channel_names == ["TL", "FITC"]
    assert image.scenes == ("B02-s0", "B02-s1", "B03-s0", "B03-s1")


def test_bioimage_reads_a_directory_when_named_explicitly(acquisition_unit: Path) -> None:
    """
    A directory carries no suffix, so it still has to be routed by hand. It
    reads the same as the descriptor does: one scene per well, stitched.
    """
    BioImage = pytest.importorskip("bioio").BioImage

    image = BioImage(acquisition_unit, reader=Reader)

    assert image.dims.shape == (2, 2, 3, 61, 32)
    assert image.scenes == ("B02", "B03")


def test_bioimage_cannot_route_a_directory_on_its_own(acquisition_unit: Path) -> None:
    """
    Pins the limitation so it cannot regress silently: extension routing has
    nothing to match on a directory name. If this ever starts passing, `bioio`
    gained directory dispatch and the README guidance should be revisited.
    """
    BioImage = pytest.importorskip("bioio").BioImage

    with pytest.raises(exceptions.UnsupportedFileFormatError):
        BioImage(acquisition_unit)


def test_a_bare_tiff_is_left_to_bioio_tifffile(acquisition_unit: Path) -> None:
    """Claiming `.jdce` must not have widened what this plugin takes."""
    plane = next((acquisition_unit / "timepoint0").glob("*.tif"))

    assert not Reader.is_supported_image(plane)
    assert ".tif" not in ReaderMetadata.get_supported_extensions()
