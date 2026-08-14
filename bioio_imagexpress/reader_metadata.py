#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import bioio_base.reader_metadata

###############################################################################


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    """
    Notes
    -----
    Defines metadata for the reader itself (not the image read),
    such as supported file extensions.
    """

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Return a list of file extensions this plugin supports reading.

        An ImageXpress acquisition is a directory of per-plane TIFFs, and the
        ``.jdce`` descriptor is the one file that identifies it. Claiming that
        extension is what lets ``bioio`` route to this reader on its own, since it
        matches plugins on the path's suffix::

            BioImage("/path/to/experiment/Acquisition.jdce")

        Naming the descriptor is also the only form that works on a filesystem
        which serves files but cannot list directories, so it is the reader's
        primary entry point rather than a special case.

        Bare ``.tif`` files are deliberately not claimed -- they belong to
        bioio-tifffile. An acquisition *directory* still reads, but has to be
        routed by hand, because its name is arbitrary and carries no suffix::

            BioImage("/path/to/experiment", reader=bioio_imagexpress.Reader)
        """
        return ["jdce"]

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        """
        Return the reader this plugin represents
        """
        from .reader import Reader

        return Reader
