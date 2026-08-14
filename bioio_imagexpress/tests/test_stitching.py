#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tile registration, on its own.

``bioio_imagexpress.stitching`` exists because an ImageXpress acquisition's only
micron-to-pixel scale, the descriptor's ``ObjectiveCalibration``, disagrees with
the real image scale by a few percent -- enough to tear a seam across a well. The
reader-facing consequences are in ``test_mosaic``; here the arithmetic is pinned
down against pictures whose true offsets are known by construction, including the
ones it is supposed to refuse to answer.
"""

from typing import List, Optional, Tuple

import numpy as np
import pytest

from bioio_imagexpress import stitching

###############################################################################

TILE = 64
STEP = 48  # what the tiles are really cut at: 25% overlap


def picture(seed: int = 0, span: int = TILE + STEP) -> np.ndarray:
    """A textured field large enough for a 2x2 grid of tiles to be cut from it."""
    return np.random.default_rng(seed).integers(
        100, 4000, size=(span, span), dtype=np.uint16
    )


def tile(field: np.ndarray, top: int, left: int) -> np.ndarray:
    return field[top : top + TILE, left : left + TILE].astype(np.float32)


def grid(field: np.ndarray, step: int = STEP) -> List[Optional[np.ndarray]]:
    """The four tiles of a serpentine 2x2 acquisition, in M order."""
    return [
        tile(field, 0, 0),
        tile(field, step, 0),
        tile(field, step, step),
        tile(field, 0, step),
    ]


TRUE_POSITIONS = [(0, 0), (STEP, 0), (STEP, STEP), (0, STEP)]

# What a descriptor 20% out about pixel size would claim: the same grid, spaced
# wrongly. Every test that corrects a layout starts from this.
WRONG_STEP = round(STEP * 1.2)
WRONG_POSITIONS = [(0, 0), (WRONG_STEP, 0), (WRONG_STEP, WRONG_STEP), (0, WRONG_STEP)]


###############################################################################
# One pair


def test_a_pair_is_placed_where_it_actually_overlaps() -> None:
    """The whole point: two crops of one picture, found at the offset they were cut."""
    field = picture()

    measured = stitching.register_pair(
        tile(field, 0, 0), tile(field, STEP, 0), guess=(float(WRONG_STEP), 0.0)
    )

    assert measured is not None
    top, left, score = measured
    assert (top, left) == (STEP, 0)
    assert score > 0.99


@pytest.mark.parametrize(
    "offset", [(STEP, 0), (0, STEP), (STEP, 0), (-STEP, 0), (0, -STEP)]
)
def test_a_pair_is_placed_the_same_way_up_and_down(offset: Tuple[int, int]) -> None:
    """
    Phase correlation only knows a shift modulo the tile, so a tile a quarter of
    the way down and one three quarters of the way up produce the same peak. The
    sign has to come from scoring the overlap, and it has to come out right in
    every direction.
    """
    field = picture(span=TILE + STEP)
    top, left = offset
    first = tile(field, max(0, -top), max(0, -left))
    second = tile(field, max(0, top), max(0, left))

    measured = stitching.register_pair(first, second, guess=(float(top), float(left)))

    assert measured is not None
    assert measured[:2] == offset


def test_unrelated_tiles_are_refused() -> None:
    """
    An empty corner of a well has nothing to match on. Something will always be
    the best offset, so the correlation floor is what stops that something from
    being reported as if it meant anything.
    """
    measured = stitching.register_pair(
        picture(seed=1, span=TILE)[:TILE, :TILE].astype(np.float32),
        picture(seed=2, span=TILE)[:TILE, :TILE].astype(np.float32),
        guess=(float(STEP), 0.0),
    )

    assert measured is None


def test_a_flat_tile_is_refused() -> None:
    """A saturated or unexposed plane correlates with everything and means nothing."""
    field = picture()

    measured = stitching.register_pair(
        tile(field, 0, 0),
        np.zeros((TILE, TILE), dtype=np.float32),
        guess=(float(STEP), 0.0),
    )

    assert measured is None


def test_a_pair_too_far_from_the_guess_is_refused() -> None:
    """
    The stage is trusted to a quarter of a tile. A registration further out than
    that is likelier to be a repeating structure matching itself than a real
    placement, and the stage's own answer is the safer one.
    """
    field = picture()

    measured = stitching.register_pair(
        tile(field, 0, 0), tile(field, STEP, 0), guess=(0.0, 0.0)
    )

    assert measured is None


###############################################################################
# A whole layout


def test_a_wrongly_scaled_layout_is_corrected() -> None:
    """The reason this module exists, at fixture scale."""
    positions = stitching.refine_tile_positions(WRONG_POSITIONS, grid(picture()))

    assert positions == TRUE_POSITIONS


def test_a_layout_that_was_already_right_survives() -> None:
    """Registering an accurate layout must not shuffle it."""
    positions = stitching.refine_tile_positions(TRUE_POSITIONS, grid(picture()))

    assert positions == TRUE_POSITIONS


def test_an_unregistrable_tile_follows_the_ones_that_registered() -> None:
    """
    A blank tile cannot be placed by measurement, but the pairs that did register
    know how far out the metadata's scale was -- so it is placed by the guess,
    rescaled by what they found, rather than by the guess as given.
    """
    tiles = grid(picture())
    tiles[3] = np.zeros((TILE, TILE), dtype=np.float32)

    positions = stitching.refine_tile_positions(WRONG_POSITIONS, tiles)

    assert positions == TRUE_POSITIONS


def test_a_layout_nothing_registers_in_keeps_the_metadata() -> None:
    """
    An empty well is not an error. The stage placement is the best available
    answer and is returned unchanged.
    """
    tiles: List[Optional[np.ndarray]] = [
        np.zeros((TILE, TILE), dtype=np.float32) for _ in range(4)
    ]

    positions = stitching.refine_tile_positions(WRONG_POSITIONS, tiles)

    assert positions == WRONG_POSITIONS


def test_an_unreadable_tile_is_skipped_rather_than_fatal() -> None:
    """A plane that would not open is None, and only costs its own placement."""
    tiles = grid(picture())
    tiles[2] = None

    positions = stitching.refine_tile_positions(WRONG_POSITIONS, tiles)

    assert positions[:2] + positions[3:] == [(0, 0), (STEP, 0), (0, STEP)]


def test_a_tile_is_placed_through_the_tiles_between_it_and_the_first() -> None:
    """
    Tile 2 is diagonal from tile 0: they share only a corner, far too little to
    register on, so it can only be reached by way of another tile. With tile 1
    unreadable the one remaining route is 0 to 3 to 2, and the pair that closes
    it is only usable once the pair before it has been used.
    """
    tiles = grid(picture())
    tiles[1] = None

    positions = stitching.refine_tile_positions(WRONG_POSITIONS, tiles)

    assert positions[2] == (STEP, STEP)


def test_a_lone_tile_is_its_own_layout() -> None:
    """Nothing to register against, and nothing that needs registering."""
    assert stitching.refine_tile_positions([(0, 0)], [picture(span=TILE)]) == [(0, 0)]
