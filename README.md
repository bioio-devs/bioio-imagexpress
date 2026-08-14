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

```python
from bioio_imagexpress import Reader 

r = Reader("my-image.ext")
r.dims
```

## Documentation

For full package documentation please visit [bioio-devs.github.io/bioio-imagexpress](https://bioio-devs.github.io/bioio-imagexpress).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
