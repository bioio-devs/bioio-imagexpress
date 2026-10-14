#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterator

import fsspec
import pytest
from bioio import BioImage

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
