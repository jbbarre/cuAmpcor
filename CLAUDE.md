# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PyCuAmpcor is a GPU (CUDA) implementation of amplitude cross-correlation (a.k.a. feature/speckle tracking) for
InSAR dense offset estimation, ported from the ROIPAC `ampcor.F` FORTRAN code. The computational core is C++/CUDA
(`src/`), exposed to Python via a pybind11 extension module (`pycuampcor`). It is normally consumed either
standalone or as a submodule of ISCE2.

## Build

Two build systems exist:

- **CMake** (primary, used for standalone/pip installs):
  ```bash
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
    -DCMAKE_BUILD_TYPE=Release
  make -j && make install
  ```
  or simply `pip install .` (uses scikit-build-core, see `pyproject.toml` / `CMakeLists.txt`).
  Debug builds (intermediate result dumps) are the CMake default; pass `-DCMAKE_BUILD_TYPE=Release` to disable.
  GPU architecture defaults to `52 60 70 75 80 86 90` if `-DCMAKE_CUDA_ARCHITECTURES` is not set.

- **SCons** (`SConscript`, `src/SConscript`): used only when building inside an ISCE2 checkout; it clones the
  ISCE2 environment (`envcontrib`) and is not meant to be invoked standalone.

`src/Makefile` is a legacy raw Makefile and is **stale** — it references source files that no longer exist
(`GDALImage.h`, `cuAmpcorChunk.*`) instead of the current `SlcImage.*` / `cuAmpcorProcessor*.*`. Do not use it as
a reference for the current source layout or as a build path; use CMake instead.

There is no automated test suite in this repository. Verifying changes to the CUDA/C++ core means building the
extension and running it against sample SLC data (e.g. via the `cuDenseOffsets` console command installed by
`pip install .`, or `python -m pycuampcor.examples.cuDenseOffsets`) or one of the other scripts under
`pycuampcor/examples/`.

CUDA toolkit (with cuFFT), pybind11, and GDAL (>=3.1.0 recommended for mmap-accelerated I/O) are required
dependencies.

## Architecture

**Two-language split**: `src/` is the CUDA/C++ computational core; `pycuampcor/__init__.py` just re-exports the
compiled `pycuampcor` extension (`from .pycuampcor import *`); `pycuampcor/examples/` holds user-facing Python
driver scripts, most importantly `cuDenseOffsets.py` (the general-purpose CLI entry point used in InSAR stacks)
and `plotOffsets.py` (offset field visualization). Each script's `main()` is also registered as a console-script
command in `[project.scripts]` in `pyproject.toml`, so after `pip install .` they're runnable directly as e.g.
`cuDenseOffsets ...` (no `.py`, no path) instead of `python -m pycuampcor.examples.cuDenseOffsets ...`. Add new
example scripts under `pycuampcor/examples/` (not a top-level `examples/`) with a callable `main()` if they should
be pip-installable the same way, and register them in `[project.scripts]`.

**Python binding** (`src/PyCuAmpcor.cpp`): a single pybind11 module wraps `cuAmpcorController`. Most parameters
are exposed as trivial property getters/setters via the `DEF_PARAM`/`DEF_PARAM_RENAME` macros that reach into
`controller.param->field` (the Python-facing name is often different from the internal C++ field name — check
this file, not just `cuAmpcorParameter.h`, to map one to the other).

**Core call chain**:
1. `cuAmpcorController` (`cuAmpcorController.{h,cpp}`) is the top-level orchestrator, constructed from Python.
   `runAmpcor()`: opens reference/secondary `SlcImage`s (GDAL-backed, optionally memory-mapped), allocates the
   output `cuArrays` (offset, SNR, covariance, peak value — sized to the *run* buffer, i.e.
   `numberChunk * numberWindowInChunk`, which may be larger than the actual requested window count), creates one
   `cuAmpcorProcessor` per CUDA stream, then iterates chunks of windows dispatching across streams to overlap
   kernel execution with data transfer. After all streams sync, results are extracted from the oversized run
   buffers into the correctly-sized output arrays, gross offsets are optionally merged in, and everything is
   written to output files.
2. `cuAmpcorParameter` (`cuAmpcorParameter.{h,cpp}`) is a plain parameter container plus derived-value
   calculator. Usage pattern: set the public fields (window sizes, search range, number of windows, etc.), call
   `setupParameters()` (dispatches to `_setupParameters_TwoPass()` or `_setupParameters_OnePass()` depending on
   `workflow`) to compute all derived sizes and allocate per-window arrays, then call one of the three
   `setStartPixels()` overloads to fill in reference/secondary starting pixels and gross offsets (static value,
   per-window array, or both fixed), and optionally `checkPixelInImageRange()` to validate before running.
3. `cuAmpcorProcessor` (`cuAmpcorProcessor.h` + `cuAmpcorProcessorTwoPass.cpp` / `cuAmpcorProcessorOnePass.cpp`)
   is an abstract per-chunk batch processor selected at runtime by `cuAmpcorProcessor::create(param->workflow, ...)`
   (0 = two-pass, 1 = one-pass). Each concrete `run(chunkIdxDown, chunkIdxAcross)` implements the full
   cross-correlation pipeline for one batch of windows on its assigned CUDA stream (see README section 5 for the
   step-by-step procedure: window extraction → raw cross-correlation → peak-centered secondary re-extraction →
   antialiasing oversampling → oversampled cross-correlation → correlation-surface oversampling/peak-finding →
   stats).

**Supporting CUDA/C++ modules** (`src/`): `cuArrays.{h,cpp}` / `cuArraysCopy.cu` / `cuArraysPadding.cu` (generic
device/host array container with copy/extract/pad helpers), `cuCorrFrequency.cu` / `cuCorrTimeDomain.cu`
(frequency- vs time-domain cross-correlation algorithms, selected by `param->algorithm`), `cuCorrNormalization.cu`
/ `cuCorrNormalizationSAT.cu` / `cuCorrNormalizer.{h,cpp}` (normalized correlation surface, SAT = summed-area
table variant), `cuOverSampler.{h,cpp}` / `cuSincOverSampler.cu` (FFT vs sinc oversampling), `cuDeramp.cu` (phase
deramp before oversampling), `cuOffset.cu` (peak position finding), `cuEstimateStats.cu` (SNR/variance around the
peak), `SlcImage.{h,cpp}` (GDAL-backed reference/secondary image reader, optional mmap I/O).

**Precision**: `data_types.h` defines `real_type`/`complex_type`/etc. as either `float` or `double` depending on
whether `CUAMPCOR_DOUBLE` is defined (currently commented out — float is the default; `isDoublePrecision()` on
the controller reflects this at runtime).

**Two workflows** (`param->workflow`): 0 = two-pass (default; cheap initial offset estimate without antialiasing
oversampling, then a refined pass with oversampling over a smaller search range — more efficient, adequate for
most cases) vs 1 = one-pass (antialiasing oversampling applied over the full search range up front — slower but
more accurate for noisy correlation surfaces). This is implemented as the two `cuAmpcorProcessor` subclasses
above, not a runtime branch within a single class.

**Naming gotcha** (recurs throughout parameter names and code comments): "row / height / down / azimuth / along
the track" all refer to the same (inner-most, x) dimension, and "column / width / across / range / along the
sight" all refer to the other (outer-most, y) dimension. Note this is the *opposite* convention from GDAL and
`ampcor.F`, which use y for rows and x for columns.
