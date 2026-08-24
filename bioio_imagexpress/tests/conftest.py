#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib

LOCAL_RESOURCES_DIR = pathlib.Path(__file__).parent / "resources"


def descriptor(directory: pathlib.Path) -> pathlib.Path:
    """The acquisition's ``.jdce``, which is its routable form."""
    return next(directory.glob("*.jdce"))
