#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterator, List, Tuple

import fsspec
import pytest
from bioio import BioImage
from bioio_base import exceptions, io, reader

from bioio_imagexpress import Reader

from .conftest import LOCAL_RESOURCES_DIR

###############################################################################

EXPERIMENT_DIR = LOCAL_RESOURCES_DIR / "experiment_z_stack"

_MEM_PATH = "/experiment_z_stack"
_MEM_URL = f"memory://{_MEM_PATH}"

###############################################################################


@pytest.fixture(scope="module")
def mem_fs() -> Iterator[fsspec.AbstractFileSystem]:
    mem = fsspec.filesystem("memory")
    mem.put(str(EXPERIMENT_DIR), _MEM_PATH, recursive=True)
    yield mem
    mem.rm(_MEM_PATH, recursive=True)


def test_bioimage_detects_acquisition_directory() -> None:
    img = BioImage(EXPERIMENT_DIR)
    assert isinstance(img.reader, Reader)


def test_bioimage_reads_remote_acquisition_directory(
    mem_fs: fsspec.AbstractFileSystem,
) -> None:
    img = BioImage(_MEM_URL)
    assert isinstance(img.reader, Reader)
    assert img.dims.order == "TCZYX"
    assert img.shape == (2, 2, 3, 64, 64)
    assert img.scenes == ("B07-s0", "B07-s1", "B08-s0", "B08-s1")


@pytest.mark.parametrize(
    "url, expected_uri",
    [
        (
            "https://s3.us-west-2.amazonaws.com/my-bucket/experiment_z_stack",
            "s3://my-bucket/experiment_z_stack",
        ),
        (
            "https://my-bucket.s3.us-west-2.amazonaws.com/experiment_z_stack",
            "s3://my-bucket/experiment_z_stack",
        ),
        (
            "https://my.dotted.bucket.s3.amazonaws.com/experiment_z_stack",
            "s3://my.dotted.bucket/experiment_z_stack",
        ),
    ],
)
def test_s3_https_url_is_read_as_anonymous_s3(
    mem_fs: fsspec.AbstractFileSystem,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    expected_uri: str,
) -> None:
    # Arrange
    calls: List[Tuple[Any, Dict[str, Any]]] = []
    pathlike_to_fs = io.pathlike_to_fs

    def fake_pathlike_to_fs(
        uri: Any, enforce_exists: bool, fs_kwargs: Dict[str, Any]
    ) -> Any:
        calls.append((uri, fs_kwargs))
        return pathlike_to_fs(_MEM_URL, enforce_exists, {})

    monkeypatch.setattr(io, "pathlike_to_fs", fake_pathlike_to_fs)
    monkeypatch.setattr(reader, "pathlike_to_fs", fake_pathlike_to_fs)

    # Act
    supported = Reader.is_supported_image(url)
    Reader(url)

    # Assert
    assert supported
    assert calls == [(expected_uri, {"anon": True})] * 2


@pytest.mark.parametrize("method", [Reader, Reader.is_supported_image])
@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/experiment_z_stack",  # not S3, nothing to list with
        "http://example.org/experiment_z_stack/",  # plain http is not S3 either
        "https://s3.example.org/my-bucket/experiment_z_stack",  # s3. host but not AWS
        "https://example.org/stack?name=run.jdce",  # .jdce is only in the query
    ],
)
def test_unlistable_http_url_raises_specific_error(method: Any, url: str) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="can't be listed"):
        method(url)


@pytest.mark.parametrize("method", [Reader, Reader.is_supported_image])
def test_http_descriptor_url_is_not_rejected_as_unlistable(method: Any) -> None:
    # Unreachable host: the descriptor is looked up, not refused up front
    with pytest.raises(Exception) as error:
        method("http://127.0.0.1:9/experiment_z_stack/run.jdce")

    assert not isinstance(error.value, exceptions.UnsupportedFileFormatError)
