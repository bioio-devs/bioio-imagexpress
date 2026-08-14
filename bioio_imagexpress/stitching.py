#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Placing mosaic tiles by registering them against each other.

Stage coordinates say where a tile was acquired in microns; turning that into a
pixel offset needs a micron-per-pixel scale, and the only one an ImageXpress
acquisition carries is ``ObjectiveCalibration.PixelWidth`` in the ``.jdce``
descriptor. That number does not match the real image scale. Measured against
both reference acquisitions, one plane per tile, by normalized cross-correlation
of the overlap:

======================  =========  ============  ==========  =============
unit                    objective  descriptor    stage step  measured step
======================  =========  ============  ==========  =============
``experiment``          4X         1.6595 um/px  2073.6 px   2115 px
``experiment_z_stack``  10X        0.5817 um/px  2073.6 px   1862 px
======================  =========  ============  ==========  =============

The stage itself is fine -- the measured steps are constant to within 3 px
across wells, and the cross-axis drift is under 3 px, so the grid really is
square and regular. It is the scale that is wrong, by +2.0% on one unit and
-10.2% on the other. Two hundred pixels of a 2304 px tile placed wrongly is a
torn seam you can see across the well, and since the error differs in size and
in sign between units acquired on the same instrument, no constant can repair
it. It has to be measured off the pixels.

So: take the descriptor's placement as a starting guess, phase-correlate each
overlapping pair of tiles to find where they actually agree, and rebuild the
layout from the pairs that correlate convincingly. Tiles that never correlate
-- an empty corner of a well has nothing to match on -- keep their guessed
position, rescaled by the median correction the other pairs agreed on.

This is deliberately translation-only. MetaXpress's own ``experiment_montage``
is not reproduced pixel for pixel, and no attempt is made to blend the seam:
the overlap is resolved last-tile-wins by the caller.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

###############################################################################

log = logging.getLogger(__name__)

# A pair is only believed if its overlap correlates this well. Measured pairs
# that are genuinely overlapping score 0.72 to 0.998 on the reference data;
# pairs with nothing in common (a blank corner tile) score below 0.13, so
# anything in between is a judgement call this never has to make.
MIN_CORRELATION = 0.3

# How far, as a fraction of the tile, a measured offset may sit from the
# descriptor's guess before it is dismissed as a mis-registration. The two
# reference units need 0.092 and 0.018 of a tile.
MAX_DRIFT = 0.25

# Overlap smaller than this fraction of a tile is not worth registering on --
# and it is what keeps the diagonal pair of a 2x2 grid, which meets only at a
# corner, from being treated as a neighbour.
MIN_OVERLAP = 0.02

# Radius of the integer search around the phase-correlation peak. The peak is
# already integer-accurate; this only recovers the odd pixel lost when one tile
# is dimmer than the other.
REFINE_RADIUS = 3

Position = Tuple[int, int]
Offset = Tuple[float, float]


###############################################################################


def refine_tile_positions(
    positions: Sequence[Position],
    tiles: Sequence[Optional[np.ndarray]],
    min_correlation: float = MIN_CORRELATION,
    max_drift: float = MAX_DRIFT,
) -> List[Position]:
    """
    Re-place tiles by registering their overlaps, starting from ``positions``.

    Parameters
    ----------
    positions: Sequence[Position]
        The descriptor's placement, as ``(top, left)`` pixel offsets in tile
        order. Used as the starting guess, as the sanity check on every
        measurement, and as the answer for any tile that cannot be registered.
    tiles: Sequence[Optional[np.ndarray]]
        One representative 2D plane per tile, in the same order. ``None`` for a
        tile whose plane could not be read; it keeps its guessed position.
    min_correlation: float
        Reject a pair whose overlap correlates below this.
    max_drift: float
        Reject a measurement further than this fraction of a tile from the guess.

    Returns
    -------
    positions: List[Position]
        ``(top, left)`` pixel offsets, referenced to the top-left-most tile.
    """
    if len(positions) < 2:
        return list(positions)

    edges = _register_neighbours(positions, tiles, min_correlation, max_drift)
    if not edges:
        log.warning(
            "No tile pair could be registered; placing %d tiles from stage "
            "metadata alone, which is accurate to a few percent of a tile.",
            len(positions),
        )
        return list(positions)

    # Everything below is relative to the first tile, so that a tile placed by
    # measurement and a tile placed by guesswork share one origin.
    guess = [(top - positions[0][0], left - positions[0][1]) for top, left in positions]
    scale = _median_scale(edges, guess)
    placed: Dict[int, Offset] = {0: (0.0, 0.0)}
    _propagate(edges, placed)

    if len(placed) < len(positions):
        log.debug(
            "%d of %d tiles registered; the rest keep their stage placement, "
            "rescaled by %.4f.",
            len(placed),
            len(positions),
            scale,
        )

    resolved = [
        placed.get(tile, (guess[tile][0] * scale, guess[tile][1] * scale))
        for tile in range(len(positions))
    ]
    top_left = min(top for top, _ in resolved), min(left for _, left in resolved)

    return [
        (int(round(top - top_left[0])), int(round(left - top_left[1])))
        for top, left in resolved
    ]


def register_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    guess: Offset,
    min_correlation: float = MIN_CORRELATION,
    max_drift: float = MAX_DRIFT,
    min_overlap: float = MIN_OVERLAP,
) -> Optional[Tuple[int, int, float]]:
    """
    Find where ``moving`` sits relative to ``reference``, or None if unsure.

    Returns ``(top, left, correlation)``: the offset of ``moving``'s top-left
    corner from ``reference``'s, and how well the two agree there.

    Phase correlation is cyclic, so its peak names a shift only modulo the tile
    size -- a tile a fifth of the way down is indistinguishable from one four
    fifths of the way up. The four candidates that peak implies are scored on
    the overlap they actually produce, and the best one wins.
    """
    if reference.shape != moving.shape:
        return None

    height, width = reference.shape
    min_area = min_overlap * height * width
    peak_top, peak_left = _phase_peak(reference, moving)
    candidates = [
        (top, left)
        for top in (peak_top, peak_top - height)
        for left in (peak_left, peak_left - width)
    ]

    best = max(
        (_correlation(reference, moving, top, left, min_area), top, left)
        for top, left in candidates
    )
    score, top, left = max(
        (_correlation(reference, moving, y, x, min_area), y, x)
        for y in range(best[1] - REFINE_RADIUS, best[1] + REFINE_RADIUS + 1)
        for x in range(best[2] - REFINE_RADIUS, best[2] + REFINE_RADIUS + 1)
    )

    if score < min_correlation:
        return None

    drift = max(abs(top - guess[0]) / height, abs(left - guess[1]) / width)
    if drift > max_drift:
        log.debug(
            "Rejecting a registration %.2f tiles from where the stage put it.",
            drift,
        )
        return None

    return top, left, score


###############################################################################


def _register_neighbours(
    positions: Sequence[Position],
    tiles: Sequence[Optional[np.ndarray]],
    min_correlation: float,
    max_drift: float,
) -> List[Tuple[float, int, int, int, int]]:
    """
    Register every pair of tiles the guess says should overlap.

    Returned as ``(correlation, reference, moving, top, left)`` tuples, best
    first, so that the layout can be grown from the most convincing pairs.
    """
    edges = []
    for first, second in combinations(range(len(positions)), 2):
        reference, moving = tiles[first], tiles[second]
        if reference is None or moving is None or reference.ndim != 2:
            continue

        guess = (
            float(positions[second][0] - positions[first][0]),
            float(positions[second][1] - positions[first][1]),
        )
        if not _guessed_to_overlap(guess, reference.shape):
            continue

        measured = register_pair(
            reference.astype(np.float32),
            moving.astype(np.float32),
            guess,
            min_correlation=min_correlation,
            max_drift=max_drift,
        )
        if measured is not None:
            top, left, score = measured
            edges.append((score, first, second, top, left))

    return sorted(edges, reverse=True)


def _guessed_to_overlap(guess: Offset, shape: Tuple[int, ...]) -> bool:
    """Whether two tiles offset by ``guess`` share enough area to register on."""
    height, width = shape
    overlap = max(0.0, height - abs(guess[0])) * max(0.0, width - abs(guess[1]))

    return overlap >= MIN_OVERLAP * height * width


def _median_scale(
    edges: Sequence[Tuple[float, int, int, int, int]],
    guess: Sequence[Offset],
) -> float:
    """
    How much the guessed spacing has to shrink or grow to match what was measured.

    Only the axis a pair actually steps along votes: a horizontal neighbour says
    nothing useful about vertical scale, and its near-zero vertical offset would
    swamp the median with noise if it were allowed to.
    """
    ratios = []
    for _, first, second, top, left in edges:
        for measured, guessed in (
            (top, guess[second][0] - guess[first][0]),
            (left, guess[second][1] - guess[first][1]),
        ):
            if abs(guessed) > 1.0:
                ratios.append(measured / guessed)

    return float(np.median(ratios)) if ratios else 1.0


def _propagate(
    edges: Sequence[Tuple[float, int, int, int, int]],
    placed: Dict[int, Offset],
) -> None:
    """
    Grow the layout outwards from the tiles already placed, best pair first.

    Repeated until nothing more can be attached, since a strong pair may join two
    tiles that are only later connected to the first one.
    """
    growing = True
    while growing:
        growing = False
        for _, first, second, top, left in edges:
            if first in placed and second not in placed:
                anchor = placed[first]
                placed[second] = (anchor[0] + top, anchor[1] + left)
                growing = True
            elif second in placed and first not in placed:
                anchor = placed[second]
                placed[first] = (anchor[0] - top, anchor[1] - left)
                growing = True


def _phase_peak(reference: np.ndarray, moving: np.ndarray) -> Tuple[int, int]:
    """
    The cyclic shift between two planes, by phase correlation.

    Whitening the cross-power spectrum is what makes this robust to the two
    tiles being at different brightness, which neighbouring fields routinely are.
    """
    first = np.fft.rfft2(reference - reference.mean())
    second = np.fft.rfft2(moving - moving.mean())
    cross = first * np.conj(second)
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1.0
    correlation = np.fft.irfft2(cross / magnitude, s=reference.shape)
    top, left = np.unravel_index(np.argmax(correlation), correlation.shape)

    return int(top), int(left)


def _correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    top: int,
    left: int,
    min_area: float,
) -> float:
    """
    Normalized cross-correlation of the overlap ``moving`` at ``(top, left)`` has
    with ``reference``, or -1 when there is too little of it to judge.
    """
    height, width = reference.shape
    first_top, last_top = max(0, top), min(height, top + height)
    first_left, last_left = max(0, left), min(width, left + width)

    if (last_top - first_top) * (last_left - first_left) < min_area:
        return -1.0

    patch = reference[first_top:last_top, first_left:last_left]
    against = moving[
        first_top - top : last_top - top, first_left - left : last_left - left
    ]
    patch = patch - patch.mean()
    against = against - against.mean()
    norm = np.sqrt(float((patch * patch).sum()) * float((against * against).sum()))

    return -1.0 if norm == 0.0 else float((patch * against).sum() / norm)
