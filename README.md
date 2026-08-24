# bioio-imagexpress

[![Build Status](https://github.com/bioio-devs/bioio-imagexpress/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-imagexpress/actions)
[![PyPI version](https://badge.fury.io/py/bioio-imagexpress.svg)](https://badge.fury.io/py/bioio-imagexpress)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for reading Molecular Devices ImageXpress (multifile) images

---


## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the generic use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base)

## Installation

**Stable Release:** `pip install bioio-imagexpress`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-imagexpress.git`

## Example Usage (see full documentation for more examples)

Install bioio-imagexpress alongside bioio:

`pip install bioio bioio-imagexpress`


This example shows a simple use case for just accessing the pixel data of the image
by explicitly passing this `Reader` into the `BioImage`. Passing the `Reader` into
the `BioImage` instance is optional as `bioio` will automatically detect installed
plug-ins and auto-select the most recently installed plug-in that supports the file
passed in.
```python
from bioio import BioImage
import bioio_imagexpress

img = BioImage("/path/to/experiment/Acquisition.jdce", reader=bioio_imagexpress.Reader)
img.data
```

## ImageXpress specifics

An ImageXpress acquisition is a **directory**, not a single file. Point at the
`.jdce` descriptor inside it so that `bioio` has a suffix to route on; the
acquisition directory itself also reads, but has to be passed `reader=` by hand.

```
<PlateBarcode>/<Protocol>_<timestamp>/     # run root
  experiment/                              # acquisition unit
      <name>.jdce                          # JSON descriptor
      image_metadata_1.csv                 # per-plane manifest
      timepoint0/
          <Project>_t0_<Well>_s<Site>_w<Channel>_z<Z>.tif
      timepoint1/ ...
  experiment_montage/ experiment_z_stack/  # further units
```

One `Reader` is one acquisition unit. A run root holds several; it raises
`UnsupportedFileFormatError` naming the units it found, so open them one at a time.

Scenes are wells (`"B07"`), and a well's sites are mosaic tiles on `M`, giving
`MTCZYX`. Pass `mosaic=False` for one scene per site (`"B07-s0"`, dims `TCZYX`).
An acquisition whose manifest is missing carries no stage positions to place
tiles with, so it falls back to per-site scenes with a warning.

```python
img = BioImage("/path/to/experiment/Acquisition.jdce")
img.scenes                          # ('B02', 'B03')
img.get_mosaic_tile_positions()     # [(0, 0), (2074, 0)]
img.mosaic_xarray_dask_data         # TCZYX, YX expanded to the stitched well
```

Tiles are placed from the manifest's stage positions scaled by the descriptor's
`ObjectiveCalibration` pixel size, which is the only micron-to-pixel scale an
acquisition carries. That scale does not match the real image scale, and the error
differs in size and sign between units taken on the same instrument - measured at
**+2%** on a 4X unit (~40 px of a 2304 px tile) and **-10%** on a 10X one (~210 px,
enough to collapse the nominal 10% overlap and duplicate a strip of sample at every
seam). Stitched output is therefore approximate. Where an acquisition includes an
`experiment_montage` unit, that is MetaXpress's own registered stitch and is the
better source; a unit with no montage sibling has no exact stitch available here.

A missing or unreadable plane raises rather than being filled with zeros.

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
