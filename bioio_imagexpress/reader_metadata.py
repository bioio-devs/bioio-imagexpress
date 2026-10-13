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

        An ImageXpress acquisition is a directory of per-plane TIFFs; the
        ``.jdce`` descriptor is the one file that identifies it, and naming it is
        what lets ``bioio`` route to this reader on its own.
        """
        return [".jdce"]

    @staticmethod
    def supports_directory_images() -> bool:
        """
        An acquisition directory denotes the image, so ``bioio`` should also probe
        this reader on directories that have no matching extension.
        """
        return True

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        """
        Return the reader this plugin represents
        """
        from .reader import Reader

        return Reader
