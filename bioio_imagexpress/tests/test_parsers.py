#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the format-parsing layer."""

import json
from datetime import datetime

import pytest

from bioio_imagexpress import parsers

from .conftest import build_jdce

###############################################################################


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "Wellscan_96well_TL_488_t0_B07_s0_w0_z0.tif",
            ("Wellscan_96well_TL_488", 0, "B07", 0, 0, 0),
        ),
        (
            "4X_Lumenoid_Cellvis_96_8-7-2026_t0_B02_s3_w0_z0.tif",
            ("4X_Lumenoid_Cellvis_96_8-7-2026", 0, "B02", 3, 0, 0),
        ),
        # Multi-digit indices must not be truncated.
        (
            "Proj_t12_H11_s10_w2_z10.tif",
            ("Proj", 12, "H11", 10, 2, 10),
        ),
        # Underscores and 't'/'s'/'w'/'z' inside the project name are common.
        (
            "TL_z_stack_w_test_t3_G11_s2_w1_z5.tif",
            ("TL_z_stack_w_test", 3, "G11", 2, 1, 5),
        ),
        ("Proj_t0_B02_s0_w0_z0.tiff", ("Proj", 0, "B02", 0, 0, 0)),
    ],
)
def test_parse_plane_name_accepts_real_forms(name, expected):
    parsed = parsers.parse_plane_name(name)

    assert parsed is not None
    assert (
        parsed.prefix,
        parsed.t,
        parsed.well,
        parsed.site,
        parsed.channel,
        parsed.z,
    ) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Wellscan_t0_B07_s0_w0_z0.tif.statistics.json",
        "Thumbs.db",
        "image_metadata_1.csv",
        "Fixture.jdce",
        # Missing the z token entirely.
        "Proj_t0_B02_s0_w0.tif",
        # Well must be a letter plus two digits.
        "Proj_t0_B2_s0_w0_z0.tif",
        "not-a-plane.tif",
    ],
)
def test_parse_plane_name_rejects_non_planes(name):
    assert parsers.parse_plane_name(name) is None


@pytest.mark.parametrize(
    "name",
    ["Thumbs.db", "thumbs.db", ".DS_Store", "x_t0_B02_s0_w0_z0.tif.statistics.json"],
)
def test_is_ignorable(name):
    assert parsers.is_ignorable(name)


def test_is_ignorable_passes_planes():
    assert not parsers.is_ignorable("Proj_t0_B02_s0_w0_z0.tif")


###############################################################################


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("B - 7", "B07"),
        ("B - 12", "B12"),
        ("B07", "B07"),
        ("b7", "B07"),
        ("  G - 11 ", "G11"),
    ],
)
def test_normalize_well(raw, expected):
    assert parsers.normalize_well(raw) == expected


@pytest.mark.parametrize("raw", ["", "not a well", "12", None])
def test_normalize_well_rejects_junk(raw):
    assert parsers.normalize_well(raw) is None


###############################################################################


def test_parse_jdce_extracts_acquisition_parameters():
    contents = json.dumps(build_jdce(["TL", "FITC"], z_count=11, t_count=9))

    metadata = parsers.parse_jdce(contents)

    assert metadata.channel_names == ["TL", "FITC"]
    assert metadata.pixel_size_x == 0.5817
    assert metadata.pixel_size_y == 0.5817
    assert metadata.z_step == 3.0
    assert metadata.objective == "10X Plan Apo Lambda D"
    assert metadata.barcode == "3500000000"
    assert metadata.binning == "1x1"
    assert metadata.operator == "fixture_operator"
    assert metadata.acquired_at == datetime(2026, 8, 4, 10, 31, 10)


def test_parse_jdce_ignores_the_creation_time_zone_offset():
    """
    MetaXpress writes 0 regardless of where the instrument is, so applying it
    would relabel an accurate local time as an inaccurate absolute one.
    """
    raw = build_jdce(["TL"], z_count=1, t_count=1)
    raw["ImageStack"]["Creation"]["TimeZoneOffset"] = 0

    acquired_at = parsers.parse_jdce(json.dumps(raw)).acquired_at

    assert acquired_at == datetime(2026, 8, 4, 10, 31, 10)
    assert acquired_at.tzinfo is None


def test_parse_jdce_falls_back_to_the_station_login():
    raw = build_jdce(["TL"], z_count=1, t_count=1)
    del raw["ImageStack"]["AutoLeadAcquisitionProtocol"]["ProjectInformation"]["User"]

    assert parsers.parse_jdce(json.dumps(raw)).operator == "moldev"


@pytest.mark.parametrize(
    "raw, expected",
    [("1 X 1", "1x1"), ("2 x 2", "2x2"), ("1x1", "1x1"), ("", None), (None, None)],
)
def test_parse_jdce_normalizes_binning(raw, expected):
    descriptor = build_jdce(["TL"], z_count=1, t_count=1)
    descriptor["ImageStack"]["AutoLeadAcquisitionProtocol"]["Camera"]["Binning"] = raw

    assert parsers.parse_jdce(json.dumps(descriptor)).binning == expected


@pytest.mark.parametrize("creation", [{}, {"Date": "not-a-date"}])
def test_parse_jdce_tolerates_an_unusable_creation_block(creation):
    raw = build_jdce(["TL"], z_count=1, t_count=1)
    raw["ImageStack"]["Creation"] = creation

    assert parsers.parse_jdce(json.dumps(raw)).acquired_at is None


def test_parse_jdce_orders_channels_by_index():
    raw = build_jdce(["TL", "FITC"], z_count=1, t_count=1)
    wavelengths = raw["ImageStack"]["AutoLeadAcquisitionProtocol"]["Wavelengths"]
    wavelengths.reverse()

    assert parsers.parse_jdce(json.dumps(raw)).channel_names == ["TL", "FITC"]


def test_parse_jdce_tolerates_a_sparse_descriptor():
    metadata = parsers.parse_jdce(json.dumps({"ImageStack": {}}))

    assert metadata.channel_names == []
    assert metadata.pixel_size_x is None
    assert metadata.z_step is None


def test_parse_jdce_falls_back_to_plate_z_step():
    raw = build_jdce(["TL"], z_count=5, t_count=1)
    protocol = raw["ImageStack"]["AutoLeadAcquisitionProtocol"]
    for wavelength in protocol["Wavelengths"]:
        wavelength["ZStep"] = 0.0
    protocol["PlateMap"]["ZDimensionParameters"]["Step"] = 1.5

    assert parsers.parse_jdce(json.dumps(raw)).z_step == 1.5


###############################################################################


def test_read_image_metadata_csv_maps_columns():
    contents = (
        "Well,Row,Column,Field,Wavelength,Timepoint,ZIndex,"
        "ImageSubFolderPath,ImageFileName,TimeStampSec,ExcitationEmissionFilter,"
        "PositionXUm,PositionYUm,PositionZUm\n"
        "B - 7,2,7,0,1,3,5,timepoint3,Proj_t3_B07_s0_w1_z5.tif,"
        "1785864670.188,FITC,67696.78,19757.0,20197.34\n"
    )

    rows = parsers.read_image_metadata_csv(contents)

    assert len(rows) == 1
    row = rows[0]
    assert (row.well, row.site, row.channel, row.t, row.z) == ("B07", 0, 1, 3, 5)
    assert row.subfolder == "timepoint3"
    assert row.timestamp_s == pytest.approx(1785864670.188)
    assert row.channel_name == "FITC"
    assert row.position_x_um == pytest.approx(67696.78)


def test_read_image_metadata_csv_falls_back_to_the_filename():
    # Coordinate columns blank -- the filename must still resolve the plane.
    contents = (
        "Well,Field,Wavelength,Timepoint,ZIndex,ImageSubFolderPath,ImageFileName\n"
        ",,,,,,Proj_t3_B07_s2_w1_z5.tif\n"
    )

    rows = parsers.read_image_metadata_csv(contents)

    assert len(rows) == 1
    assert (rows[0].well, rows[0].site, rows[0].channel, rows[0].t, rows[0].z) == (
        "B07",
        2,
        1,
        3,
        5,
    )
    # Subfolder must be inferred from the time point when the column is empty.
    assert rows[0].subfolder == "timepoint3"


def test_read_image_metadata_csv_drops_unresolvable_rows():
    contents = (
        "Well,Field,Wavelength,Timepoint,ZIndex,ImageSubFolderPath,ImageFileName\n"
        ",,,,,,mystery.tif\n"
    )

    assert parsers.read_image_metadata_csv(contents) == []
