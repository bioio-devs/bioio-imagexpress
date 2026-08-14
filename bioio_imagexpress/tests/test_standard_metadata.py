#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The metadata surfaces: the standard field set, and the source files as JSON."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bioio_base.standard_metadata import StandardMetadata

from bioio_imagexpress import Reader

from .conftest import (
    EPOCH,
    PLANE_SIZE,
    TIME_INTERVAL_S,
    make_acquisition_unit,
    tile_stage_position,
)

###############################################################################

# The fixture's pixel size, which sets the stage distance between tile centres.
PIXEL_SIZE = 0.5817

TILE_0_X, TILE_0_Y = tile_stage_position(0, PLANE_SIZE, PIXEL_SIZE)
TILE_1_X, TILE_1_Y = tile_stage_position(1, PLANE_SIZE, PIXEL_SIZE)


def test_every_field_the_format_can_supply_is_populated(acquisition_unit: Path) -> None:
    """
    ImageXpress carries all but one standard field. Asserting on the whole set
    catches a field silently dropping to None as the parsers change.

    A default reader mosaics a well's sites into one scene, so the fields here
    describe the whole well: the tile axis joins the dimension order, there is no
    single acquisition position to index, and the stage position is the first
    tile's -- the origin the stitch is laid out from.
    """
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")

    metadata = reader.standard_metadata

    assert metadata == StandardMetadata(
        binning="1x1",
        column="3",
        dimensions_present="MTCZYX",
        image_size_c=2,
        image_size_t=2,
        image_size_x=32,
        image_size_y=32,
        image_size_z=3,
        imaged_by="fixture_operator",
        imaging_datetime=datetime(2026, 8, 4, 10, 31, 10),
        objective="10X Plan Apo Lambda D",
        pixel_size_x=0.5817,
        pixel_size_y=0.5817,
        pixel_size_z=3.0,
        position_index=None,
        row="B",
        stage_position_x=TILE_0_X,
        stage_position_y=TILE_0_Y,
        timelapse=True,
        timelapse_interval=timedelta(seconds=TIME_INTERVAL_S),
        total_time_duration=timedelta(seconds=TIME_INTERVAL_S),
    )


def test_every_field_the_format_can_supply_is_populated_per_tile(
    acquisition_unit: Path,
) -> None:
    """
    The same whole-set assertion with mosaic off, where a scene is one
    acquisition position: no tile axis, and the tile's own index and stage
    position rather than the well's origin.
    """
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")

    metadata = reader.standard_metadata

    assert metadata == StandardMetadata(
        binning="1x1",
        column="3",
        dimensions_present="TCZYX",
        image_size_c=2,
        image_size_t=2,
        image_size_x=32,
        image_size_y=32,
        image_size_z=3,
        imaged_by="fixture_operator",
        imaging_datetime=datetime(2026, 8, 4, 10, 31, 10),
        objective="10X Plan Apo Lambda D",
        pixel_size_x=0.5817,
        pixel_size_y=0.5817,
        pixel_size_z=3.0,
        position_index=1,
        row="B",
        stage_position_x=TILE_1_X,
        stage_position_y=TILE_1_Y,
        timelapse=True,
        timelapse_interval=timedelta(seconds=TIME_INTERVAL_S),
        total_time_duration=timedelta(seconds=TIME_INTERVAL_S),
    )


def _scene_snapshot(reader: Reader, scene_id: str):
    """The fields a scene change has to move, as a comparable tuple."""
    reader.set_scene(scene_id)
    metadata = reader.standard_metadata

    return (
        metadata.row,
        metadata.column,
        metadata.position_index,
        metadata.pixel_size_x,
        metadata.image_size_c,
        metadata.timelapse,
        metadata.total_time_duration,
    )


def test_fields_follow_the_current_scene(run_root: Path) -> None:
    """
    A run root mixes units, so nothing here may survive a scene change.

    Scenes are wells by default, and a well is not a single acquisition
    position, so ``position_index`` is None throughout -- it is kept in the
    snapshot to catch a stale tile index leaking in from a previous scene.
    """
    reader = Reader(run_root)

    experiment = ("B", "3", None, 0.5817, 2, True, timedelta(seconds=TIME_INTERVAL_S))
    montage = ("B", "2", None, 1.6595, 1, False, None)

    assert _scene_snapshot(reader, "experiment/B03") == experiment
    assert _scene_snapshot(reader, "experiment_montage/B02") == montage
    # And back, to catch state that only leaks one way.
    assert _scene_snapshot(reader, "experiment/B03") == experiment


def test_fields_follow_the_current_scene_per_tile(run_root: Path) -> None:
    """
    The same crossing of units with mosaic off, where each scene is one tile and
    ``position_index`` has to move with it too.
    """
    reader = Reader(run_root, mosaic=False)

    experiment = ("B", "3", 1, 0.5817, 2, True, timedelta(seconds=TIME_INTERVAL_S))
    montage = ("B", "2", 0, 1.6595, 1, False, None)

    assert _scene_snapshot(reader, "experiment/B03-s1") == experiment
    assert _scene_snapshot(reader, "experiment_montage/B02-s0") == montage
    # And back, to catch state that only leaks one way.
    assert _scene_snapshot(reader, "experiment/B03-s1") == experiment


def test_to_dict_uses_the_readable_labels(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    as_dict = reader.standard_metadata.to_dict()

    assert as_dict["Binning"] == "1x1"
    assert as_dict["Row"] == "B"
    assert as_dict["Column"] == "2"
    assert as_dict["Objective"] == "10X Plan Apo Lambda D"


###############################################################################
# reader.metadata stays format-native


def test_metadata_is_the_formats_own_and_nothing_else(acquisition_unit: Path) -> None:
    """
    The normalized field set lives on `standard_metadata`; `metadata` must not
    duplicate it.
    """
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")

    metadata = reader.metadata

    assert "standard" not in metadata
    assert metadata["well"] == "B03"
    assert metadata["column"] == 3
    assert metadata["jdce"]["ImageStack"]["PlateId"] == "3500000000"
    assert metadata["indexed_from"] == "image_metadata_csv"


def test_metadata_does_not_mutate_the_array_attrs(acquisition_unit: Path) -> None:
    """`metadata` composes a new dict; the xarray attrs are left alone."""
    reader = Reader(acquisition_unit)

    reader.metadata

    assert "image_metadata" not in reader.xarray_dask_data.attrs["unprocessed"]


###############################################################################
# The manifest, as JSON-able Python


def test_metadata_carries_the_manifest_rows_for_this_scene(acquisition_unit: Path) -> None:
    """
    The CSV manifest becomes one dict per row with every column kept -- the
    per-plane record the reduced index rows throw away.

    A default scene is the whole well, so its rows cover every tile of it.
    """
    reader = Reader(acquisition_unit)
    reader.set_scene("B03")

    records = reader.metadata["image_metadata"]

    # 2 site x 2 t x 2 c x 3 z for this scene, out of 48 rows for the unit.
    assert len(records) == 24
    assert {row["Well"] for row in records} == {"B - 3"}
    assert {row["Field"] for row in records} == {"0", "1"}
    assert {row["ImageFileName"] for row in records} == {
        f"Fixture_t{t}_B03_s{s}_w{c}_z{z}.tif"
        for s in range(2)
        for t in range(2)
        for c in range(2)
        for z in range(3)
    }
    # Columns the index drops must survive.
    assert records[0]["ExposureTimeMs"] == "10"
    assert records[0]["PositionZUm"] is not None


def test_metadata_carries_the_manifest_rows_for_one_tile(acquisition_unit: Path) -> None:
    """
    With mosaic off a scene is one acquisition position, so the rows must be
    narrowed to that tile rather than to its well.
    """
    reader = Reader(acquisition_unit, mosaic=False)
    reader.set_scene("B03-s1")

    records = reader.metadata["image_metadata"]

    # 2 t x 2 c x 3 z for this scene, out of 48 rows for the unit.
    assert len(records) == 12
    assert {row["Well"] for row in records} == {"B - 3"}
    assert {row["Field"] for row in records} == {"1"}
    assert {row["ImageFileName"] for row in records} == {
        f"Fixture_t{t}_B03_s1_w{c}_z{z}.tif"
        for t in range(2)
        for c in range(2)
        for z in range(3)
    }
    # Columns the index drops must survive.
    assert records[0]["ExposureTimeMs"] == "10"
    assert records[0]["PositionZUm"] is not None


def test_manifest_rows_keep_every_value_as_written(acquisition_unit: Path) -> None:
    """
    Cells stay strings: coercing would not round-trip a checksum or a zero-padded
    id. Empty cells become None.
    """
    reader = Reader(acquisition_unit)

    row = reader.metadata["image_metadata"][0]

    assert all(value is None or isinstance(value, str) for value in row.values())
    assert row["Row"] == "2"


def test_metadata_is_json_serializable(acquisition_unit: Path) -> None:
    """
    The point of the conversion: both source files land as JSON-able Python.
    """
    reader = Reader(acquisition_unit)
    metadata = reader.metadata

    payload = {key: metadata[key] for key in ("jdce", "image_metadata")}

    assert json.loads(json.dumps(payload)) == payload
    # The descriptor is JSON already -- only the extension is unusual.
    assert metadata["jdce"]["ImageStack"]["PlateId"] == "3500000000"


def test_manifest_rows_follow_the_current_scene(run_root: Path) -> None:
    reader = Reader(run_root)

    reader.set_scene("experiment/B02")
    assert {r["Field"] for r in reader.metadata["image_metadata"]} == {"0", "1"}

    reader.set_scene("experiment_montage/B03")
    records = reader.metadata["image_metadata"]
    assert {r["Well"] for r in records} == {"B - 3"}
    assert len(records) == 1  # this unit is 1 site x 1 t x 1 c x 1 z


def test_manifest_rows_are_empty_without_a_manifest(tmp_path: Path) -> None:
    """A filename-indexed unit has no manifest to expose."""
    reader = Reader(make_acquisition_unit(tmp_path / "without", write_csv=False))

    assert reader.metadata["image_metadata"] == []
    # The descriptor is still there.
    assert reader.metadata["jdce"]["ImageStack"]["PlateId"] == "3500000000"


###############################################################################
# Timing


def test_durations_come_from_the_manifest(acquisition_unit: Path) -> None:
    reader = Reader(acquisition_unit)

    assert reader.total_time_duration == timedelta(seconds=TIME_INTERVAL_S)
    assert reader.time_interval == timedelta(seconds=TIME_INTERVAL_S)


def test_durations_are_none_for_a_single_time_point(tmp_path: Path) -> None:
    reader = Reader(make_acquisition_unit(tmp_path / "single", t_count=1))

    assert reader.total_time_duration is None
    assert reader.time_interval is None
    assert reader.standard_metadata.timelapse is False


def test_durations_are_none_without_a_manifest(tmp_path: Path) -> None:
    """Timestamps live only in the CSV; the filenames cannot supply them."""
    reader = Reader(make_acquisition_unit(tmp_path / "without", write_csv=False))

    assert reader.total_time_duration is None
    assert reader.time_interval is None
    # But the descriptor still knows this was a timelapse.
    assert reader.standard_metadata.timelapse is True


def test_interval_averages_over_more_than_two_time_points(tmp_path: Path) -> None:
    reader = Reader(make_acquisition_unit(tmp_path / "long", t_count=4))

    assert reader.total_time_duration == timedelta(seconds=3 * TIME_INTERVAL_S)
    assert reader.time_interval == timedelta(seconds=TIME_INTERVAL_S)


###############################################################################
# Acquisition datetime


def test_imaging_datetime_prefers_the_descriptor(acquisition_unit: Path) -> None:
    """
    The descriptor's stamp is instrument local time; the manifest's is a Unix
    epoch. They describe the same moment and must not be mixed up.
    """
    reader = Reader(acquisition_unit)

    assert reader.imaging_datetime == datetime(2026, 8, 4, 10, 31, 10)
    assert reader.imaging_datetime.tzinfo is None


def test_imaging_datetime_falls_back_to_the_manifest(tmp_path: Path) -> None:
    unit = make_acquisition_unit(tmp_path / "no_creation")
    descriptor = next(unit.glob("*.jdce"))
    descriptor.write_text(descriptor.read_text().replace('"Creation"', '"_Creation"'))

    reader = Reader(unit)

    # A Unix epoch, so this fallback is UTC-aware rather than local-naive.
    assert reader.imaging_datetime == datetime.fromtimestamp(EPOCH, tz=timezone.utc)


def test_imaging_datetime_is_none_without_either_source(tmp_path: Path) -> None:
    unit = make_acquisition_unit(tmp_path / "bare", write_csv=False)
    descriptor = next(unit.glob("*.jdce"))
    descriptor.write_text(descriptor.read_text().replace('"Creation"', '"_Creation"'))

    assert Reader(unit).imaging_datetime is None
