# bioio-imagexpress

[![Build Status](https://github.com/bioio-devs/bioio-imagexpress/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-imagexpress/actions)
[![Documentation](https://github.com/bioio-devs/bioio-imagexpress/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio-imagexpress)

A BioIO reader plugin for reading Molecular Devices ImageXpress (multifile) images.

This plugin is intended to be used in conjunction with [bioio](https://github.com/bioio-devs/bioio)
---

## Installation

**Stable Release:** `pip install bioio-imagexpress`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-imagexpress.git`

## Quickstart

An ImageXpress acquisition is a **directory**, not a single file.

Name the **`.jdce` descriptor** — it is the file that identifies an acquisition, and
the entry point everything else is built around:

```python
from bioio_imagexpress import Reader

r = Reader("/path/to/experiment/Acquisition.jdce")
r.scenes        # ('B02', 'B03', ...) -- one per well
r.dims          # <Dimensions [M: 4, T: 9, C: 2, Z: 11, Y: 2304, X: 2304]>
r.channel_names # ['TL', 'FITC']

r.set_scene("B03")
plane = r.get_image_data("YX", T=0, C=0, Z=0)
```

### Using it through `BioImage`

The plugin claims the `.jdce` extension, so `bioio` routes to it unaided:

```python
from bioio import BioImage

img = BioImage("/path/to/experiment/Acquisition.jdce")   # routed automatically
```

A *directory* also reads, but a directory name has no suffix for `bioio` to match
on, so that form has to be routed by hand:

```python
import bioio_imagexpress

img = BioImage("/path/to/experiment", reader=bioio_imagexpress.Reader)  # works
img = BioImage("/path/to/experiment")                                   # raises
```

### What it accepts

| Input | Result | Needs listing? |
| --- | --- | --- |
| A `.jdce` descriptor | the unit containing it, ids like `B07` | no |
| An acquisition unit (a directory holding a `.jdce` and `timepoint<N>/` folders) | its scenes, ids like `B07` | yes |
| A run root holding several units | every unit's scenes, ids like `experiment_z_stack/B07` | yes |

Only the descriptor form works on a filesystem that serves files but refuses to
list directories — see [Remote and object stores](#remote-and-object-stores).

### Layout

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

Scenes are one per well; time points, channels and Z steps stack into an `MTCZYX`
array. Planes are read lazily, one chunk per `YX` plane.

### Mosaic tiles

A well is imaged as a grid of overlapping fields — the `s0`, `s1`, … in the plane
filenames. These are **mosaic tiles, not independent acquisitions**: their stage
positions form a 2×2 grid at 0.90 of a field of view (10% overlap), and MetaXpress's
own `experiment_montage` unit is its stitch of exactly those tiles.

They therefore land on the `M` dimension, and a scene is a whole well:

```python
r = Reader(".../experiment/Acquisition.jdce")
r.scenes                       # ('B02', 'B03', ...) -- one per well
r.dims                         # <Dimensions [M: 4, T: 1, C: 1, Z: 1, Y: 2304, X: 2304]>
r.get_mosaic_tile_positions()  # [(0, 0), (2074, 0), (2074, 2074), (0, 2074)]
r.mosaic_xarray_data           # TCZYX, YX expanded to the stitched well
```

Turn it off when the tiles are the unit of analysis rather than the well:

```python
r = Reader(path, mosaic=False)
r.scenes                       # ('B02-s0', 'B02-s1', ...) -- one per well and tile
r.dims                         # <Dimensions [T: 1, C: 1, Z: 1, Y: 2304, X: 2304]>
```

Stitching places tiles from their manifest stage positions, with the later tile
winning in the overlap rather than blending. MetaXpress refines its own montage by
image registration — observed tile spacing differs from the stage-derived spacing by
a few percent — so `experiment_montage` is not reproduced pixel for pixel. Prefer
that unit where the acquisition has one; note that `experiment_z_stack` does not.

Indexing is manifest-driven and costs **two file reads, no directory listings**: the
`.jdce` descriptor names its `image_metadata_*.csv` manifests, and those name every
plane's subfolder and filename. The 306 GB / 23,760-plane reference acquisition
indexes in ~0.7 s locally, against ~8 s for the directory walk it replaced.

### Remote and object stores

Because indexing never lists a directory, an acquisition is readable over any
filesystem that can serve files — including read-only HTTP endpoints with directory
listing disabled. This is the same descriptor form as above, and `BioImage` routes
it automatically:

```python
Reader("https://host/path/experiment_z_stack/Acquisition.jdce")
BioImage("https://host/path/experiment_z_stack/Acquisition.jdce")
```

That same 306 GB acquisition indexes over plain HTTPS in ~1.3 s and reads a
2304×2304 plane in ~2.5 s, byte-identical to the mounted copy. Pointing at the
*directory* still requires a listable filesystem (local, S3, or an HTTP server with
autoindex on), because the descriptor's name has to be discovered.

### Notes

- **Resolution levels.** MetaXpress writes each plane as a small pyramid, exposed
  through `r.resolution_levels` and `r.set_resolution_level(n)`.
- **Region reading.** `get_image_data` resolves its `T`/`C`/`Z` selection to the exact
  plane files it names and opens only those, via the `_read_indexed` seam. On the
  reference z-stack (a 9x2x11 scene, 198 planes) reading one plane over the network
  mount goes from 414 s to 0.02 s. `Y`/`X` are cropped after each plane is read —
  a MetaXpress plane is strip-per-row with no tiling to exploit, so a spatial crop
  saves memory rather than I/O. The dask path already sliced its graph and is
  unchanged.
- **Heterogeneous run roots.** Units within one run root routinely differ in shape,
  channel count and pixel size, so shape and metadata always reflect the current
  scene.
- **Incomplete acquisitions.** Missing or unreadable planes are filled with zeros
  and listed in `r.xarray_dask_data.attrs["missing_planes"]` rather than failing
  the read. Because the manifest is trusted, a row whose plane was never written
  is found missing at *read* time — zeros plus a logged warning — rather than
  shortening the array. Pass `Reader(..., verify_planes=True)` to confirm every
  row against the files present instead; that costs a full directory walk and
  needs a listable filesystem, and is worth it for a part-transferred copy.
- **Standard metadata.** `r.standard_metadata` fills every `bioio-base` field the
  format carries — binning, plate row/column, site, objective, operator,
  acquisition datetime, stage position and timelapse timing — resolved against the
  current scene. `imaging_datetime` is the descriptor's own stamp, which is naive
  instrument local time; it falls back to the manifest's Unix timestamp
  (UTC-aware) only when the descriptor has no `Creation` block.
- **Metadata as JSON.** `r.metadata` carries both source files in full, as
  JSON-able Python, and stays format-native throughout:

  ```python
  r.metadata["jdce"]            # the descriptor, verbatim (it is JSON already)
  r.metadata["image_metadata"]  # this scene's manifest rows, one dict per row
  r.metadata["metaseries"]      # the plane TIFF's MetaSeries tags
  ```

  Every manifest column is kept, so the per-plane record — exposure, intensity
  statistics, incubation temperature, CO₂, O₂, field offsets, FOV uuid — survives
  the conversion. Cells stay as written rather than being coerced to numbers,
  since a checksum or zero-padded id would not round-trip; empty cells become
  `None`. Rows are read on demand and scoped to the current scene, so a 5 MB /
  24,000-row manifest costs nothing until asked for.
- **Format.** This reader targets the MetaXpress 2026+ `.jdce` export. The older
  `.HTD` / `TimePoint_N` / `ZStep_N` layout is not yet supported.

## Documentation

For full package documentation please visit [bioio-devs.github.io/bioio-imagexpress](https://bioio-devs.github.io/bioio-imagexpress).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
