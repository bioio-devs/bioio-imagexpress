#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manifest-driven indexing.

The ``.jdce`` descriptor names its ``image_metadata_*.csv`` manifests, and those
name every plane's subfolder and filename. Indexing therefore needs two file reads
and no directory listing -- which is what makes an acquisition readable over a
filesystem that serves files but refuses to list.
"""

import json
from pathlib import Path
from typing import List

import fsspec
import numpy as np
import pytest
from bioio_base import exceptions
from fsspec.implementations.local import LocalFileSystem

from bioio_imagexpress import Reader, parsers

from .conftest import PLANE_SIZE, make_acquisition_unit, tile_stage_position

###############################################################################


class NoListingFileSystem(LocalFileSystem):
    """
    A filesystem that serves files but refuses to list, like an HTTP endpoint with
    nginx autoindex off.

    ``isdir`` is answered too -- HEAD-style existence checks are what a read-only
    HTTP endpoint does support; it is enumeration it will not do.
    """

    def ls(self, path, detail=True, **kwargs) -> None:
        raise PermissionError(f"403 Forbidden: listing is not permitted ({path})")


@pytest.fixture
def unlistable(monkeypatch) -> List[str]:
    """Installs the no-listing filesystem and records what was opened."""
    fs = NoListingFileSystem()
    opened: List[str] = []

    original_open = fs.open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(fs, "open", spy)
    monkeypatch.setattr(
        fsspec, "filesystem", lambda protocol, **kwargs: fs, raising=True
    )
    monkeypatch.setattr(
        "bioio_base.io.pathlike_to_fs",
        lambda image, enforce_exists=False, fs_kwargs=None: (fs, str(image)),
    )

    return opened


###############################################################################
# The manifest is the index


def test_indexing_opens_only_the_descriptor_and_manifest(
    acquisition_unit: Path, monkeypatch
) -> None:
    """
    Forty-eight planes on disk, but indexing must touch neither them nor a listing.
    """
    fs = LocalFileSystem()
    opened: List[str] = []
    original = fs.open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fs, "open", spy)
    monkeypatch.setattr(
        "bioio_base.io.pathlike_to_fs",
        lambda image, enforce_exists=False, fs_kwargs=None: (fs, str(image)),
    )

    reader = Reader(acquisition_unit)
    # One scene per well: with mosaic on the two sites are tiles of one scene.
    assert reader.scenes == ("B02", "B03")

    # Deduplicated: fsspec services a text-mode open with a second binary one.
    distinct = list(dict.fromkeys(Path(p).name for p in opened))
    assert distinct == ["Fixture.jdce", "image_metadata_1.csv"]


def test_descriptor_names_its_manifests(acquisition_unit: Path) -> None:
    descriptor = next(acquisition_unit.glob("*.jdce"))

    jdce = parsers.parse_jdce(descriptor.read_text())

    assert jdce.metadata_files == ["image_metadata_1.csv"]


def test_resolve_metadata_csvs_prefers_the_descriptor(acquisition_unit: Path) -> None:
    fs = LocalFileSystem()
    jdce = parsers.parse_jdce(next(acquisition_unit.glob("*.jdce")).read_text())

    resolved = parsers.resolve_metadata_csvs(fs, str(acquisition_unit), jdce)

    assert [Path(p).name for p in resolved] == ["image_metadata_1.csv"]


def test_resolve_metadata_csvs_falls_back_to_listing(acquisition_unit: Path) -> None:
    """A descriptor that does not name its manifests must still resolve them."""
    fs = LocalFileSystem()
    jdce = parsers.JdceMetadata()  # no metadata_files

    resolved = parsers.resolve_metadata_csvs(fs, str(acquisition_unit), jdce)

    assert [Path(p).name for p in resolved] == ["image_metadata_1.csv"]


###############################################################################
# Reading without any directory listing


def test_reads_through_a_filesystem_that_cannot_list(
    acquisition_unit: Path, unlistable: List[str]
) -> None:
    descriptor = next(acquisition_unit.glob("*.jdce"))

    reader = Reader(descriptor)

    assert reader.scenes == ("B02", "B03")
    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.channel_names == ["TL", "FITC"]
    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)
    # Timestamps and stage positions come from the manifest, so they survive.
    # With mosaic on the scene's stage position is its first tile's.
    assert (
        reader.standard_metadata.stage_position_x
        == tile_stage_position(0, PLANE_SIZE, 0.5817)[0]
    )
    assert reader.standard_metadata.total_time_duration is not None


def test_pixels_read_through_a_filesystem_that_cannot_list(
    acquisition_unit: Path, unlistable: List[str]
) -> None:
    """The tile a plane belongs to is composed from the manifest, not listed."""
    from .conftest import plane_value

    reader = Reader(next(acquisition_unit.glob("*.jdce")))
    reader.set_scene("B03")

    plane = reader.get_image_data("YX", M=1, T=1, C=0, Z=2)

    assert np.all(plane == plane_value("B03", 1, 1, 0, 2))


def test_tile_pixels_read_through_a_filesystem_that_cannot_list(
    acquisition_unit: Path, unlistable: List[str]
) -> None:
    """The same plane, reached with mosaic off, where the site is its own scene."""
    from .conftest import plane_value

    reader = Reader(next(acquisition_unit.glob("*.jdce")), mosaic=False)
    reader.set_scene("B03-s1")

    plane = reader.get_image_data("YX", T=1, C=0, Z=2)

    assert np.all(plane == plane_value("B03", 1, 1, 0, 2))


def test_a_directory_is_still_rejected_when_listing_is_impossible(
    acquisition_unit: Path, unlistable: List[str]
) -> None:
    """
    Pointing at the directory cannot work without listing -- the descriptor's name
    is unknowable. The error must say so rather than fail obscurely.
    """
    with pytest.raises(exceptions.UnsupportedFileFormatError) as raised:
        Reader(acquisition_unit)

    assert "name the '.jdce' descriptor directly" in str(raised.value)


###############################################################################
# A composed path is not a promise the file is there


MISSING_PLANES = [
    # The first plane of the scene: what the shape probe reaches for first.
    "timepoint0/Fixture_t0_B02_s0_w0_z0.tif",
    # And one in the middle, which was always handled.
    "timepoint1/Fixture_t1_B02_s0_w1_z2.tif",
]


@pytest.mark.parametrize("victim", MISSING_PLANES)
def test_one_absent_plane_never_costs_the_whole_scene(tmp_path: Path, victim: str) -> None:
    """
    Manifest-driven indexing composes plane paths without confirming them, so any
    one of them may be missing on a part-copied acquisition. Losing the plane the
    shape probe happens to pick must not take down `dims`, metadata and pixels for
    every other plane in the scene -- here the whole well, tiles included.
    """
    unit = make_acquisition_unit(tmp_path / "partial")
    (unit / victim).unlink()

    reader = Reader(unit)
    reader.set_scene("B02")

    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.dtype == np.uint16
    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)
    assert reader.standard_metadata.objective == "10X Plan Apo Lambda D"
    assert reader.resolution_levels == (0, 1, 2)

    # The absent plane reads as zeros; its neighbours are untouched.
    from .conftest import plane_value

    assert np.all(
        reader.get_image_data("YX", M=0, T=0, C=0, Z=1)
        == plane_value("B02", 0, 0, 0, 1)
    )


@pytest.mark.parametrize("victim", MISSING_PLANES)
def test_one_absent_plane_never_costs_the_whole_tile(tmp_path: Path, victim: str) -> None:
    """The same partial acquisition with mosaic off, where the scene is one site."""
    unit = make_acquisition_unit(tmp_path / "partial")
    (unit / victim).unlink()

    reader = Reader(unit, mosaic=False)
    reader.set_scene("B02-s0")

    assert reader.dims.shape == (2, 2, 3, 32, 32)
    assert reader.dtype == np.uint16
    assert reader.physical_pixel_sizes == (3.0, 0.5817, 0.5817)
    assert reader.standard_metadata.objective == "10X Plan Apo Lambda D"
    assert reader.resolution_levels == (0, 1, 2)

    # The absent plane reads as zeros; its neighbours are untouched.
    from .conftest import plane_value

    assert np.all(
        reader.get_image_data("YX", T=0, C=0, Z=1) == plane_value("B02", 0, 0, 0, 1)
    )


def test_a_scene_with_no_readable_plane_still_raises(tmp_path: Path) -> None:
    """Degrading is for partial loss; a scene with nothing behind it is an error."""
    unit = make_acquisition_unit(
        tmp_path / "gone", wells=["B02"], sites=[0], channels=["TL"], t_count=1
    )
    for plane in (unit / "timepoint0").glob("*.tif"):
        plane.unlink()

    reader = Reader(unit)

    with pytest.raises(FileNotFoundError):
        reader.dims


###############################################################################
# Backends refuse to list in more ways than OSError


class DeniedListingFileSystem(LocalFileSystem):
    """
    Refuses to list with an exception that is *not* an OSError.

    s3fs turns a denied ``ListBucket`` into ``PermissionError`` (which is one),
    but fsspec's HTTP backend raises ``aiohttp.ClientResponseError``, which
    descends from ``Exception`` alone. The discovery helpers have to survive both.
    """

    class NotAnOSError(Exception):
        pass

    def ls(self, path, detail=True, **kwargs) -> None:
        raise self.NotAnOSError("403 Forbidden")

    def isdir(self, path) -> None:
        raise self.NotAnOSError("403 Forbidden")


def test_discovery_survives_a_non_oserror_refusal(acquisition_unit: Path, monkeypatch) -> None:
    """`discover_units` must answer, not raise, when the backend refuses."""
    fs = DeniedListingFileSystem()
    descriptor = str(next(acquisition_unit.glob("*.jdce")))

    # The descriptor still resolves: it needs existence, not listing.
    assert parsers.discover_units(fs, descriptor) == [
        parsers.DiscoveredUnit("", str(acquisition_unit), descriptor)
    ]
    # And a directory degrades to "nothing here" rather than propagating.
    assert parsers.discover_units(fs, str(acquisition_unit)) == []
    assert parsers._names_in(fs, str(acquisition_unit)) == []


def test_is_supported_image_answers_rather_than_raising(acquisition_unit: Path) -> None:
    """
    bioio calls this hook while probing every registered plugin, so a raise here
    would break routing for unrelated formats, not just this one.
    """
    fs = DeniedListingFileSystem()

    assert Reader._is_supported_image(fs, str(next(acquisition_unit.glob("*.jdce"))))
    assert not Reader._is_supported_image(fs, str(acquisition_unit))


###############################################################################
# Fallbacks


def test_falls_back_to_walking_without_a_manifest(tmp_path: Path) -> None:
    unit = make_acquisition_unit(tmp_path / "without", write_csv=False)

    reader = Reader(unit)

    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.xarray_dask_data.attrs["unprocessed"]["indexed_from"] == "filenames"


def test_falls_back_to_walking_when_the_named_manifest_is_gone(tmp_path: Path) -> None:
    """The descriptor names a manifest that was never copied."""
    unit = make_acquisition_unit(tmp_path / "no_csv")
    (unit / "image_metadata_1.csv").unlink()

    reader = Reader(unit)

    assert reader.dims.shape == (2, 2, 2, 3, 32, 32)
    assert reader.xarray_dask_data.attrs["unprocessed"]["indexed_from"] == "filenames"


def test_manifest_index_matches_the_verified_index(tmp_path: Path) -> None:
    """
    On an intact acquisition the fast path and the walk must agree exactly.
    """
    unit = make_acquisition_unit(tmp_path / "intact")

    fast = Reader(unit)
    verified = Reader(unit, verify_planes=True)

    assert fast.scenes == verified.scenes
    assert fast.dims.shape == verified.dims.shape
    assert np.array_equal(fast.xarray_data.data, verified.xarray_data.data)
    assert fast.standard_metadata == verified.standard_metadata


def test_a_planeless_descriptor_is_not_claimed(tmp_path: Path) -> None:
    """A protocol definition with no planes behind it is not an acquisition."""
    empty = tmp_path / "protocol"
    empty.mkdir()
    (empty / "Protocol.jdce").write_text(json.dumps({"ImageStack": {}}))

    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(empty / "Protocol.jdce")
