# grossOffsets — velocity-model gross offsets for PyCuAmpcor

`grossOffsets` converts the **MEaSUREs InSAR-Based Antarctica Ice Velocity Map (v2)**
([doi:10.5067/D7GK8F5J8M8R](https://doi.org/10.5067/D7GK8F5J8M8R), NSIDC-0484) into
*per-window integer pixel offsets* in radar geometry. Feeding them to `cuDenseOffsets` as a
varying gross offset lets you use a much smaller search range on fast-moving ice.

Installed as a console command by `pip install .` (`[project.scripts]`), or run as
`python -m pycuampcor.grossOffsets`. Runtime dependencies: `numpy`, `pyproj`
(GDAL only for the optional raster geometry input / `--save-rasters`, matplotlib only for `--plot`;
no ISCE).

## Quick start

Needs `numpy` and `pyproj`; the one-time model preparation also needs `netCDF4`. Inputs: the MEaSUREs
v2 450 m NetCDF and the GAMMA `.par` of the reference and secondary images.

```bash
# 1. once: convert the NetCDF to a memory-mappable .npy (+ .json)
grossOffsets --prepare-model antarctica_ice_velocity_450m_v2.nc antarctica_vxvy_450m.npy

# 2. per pair: use the SAME window-grid options as the cuDenseOffsets command
grossOffsets --model antarctica_vxvy_450m.npy --ref-par ref.slc.par --sec-par sec.slc.par \
    --ww 64 --wh 64 --sw 20 --sh 20 --kw 64 --kh 64 --outdir out --outname grossOffsets.bin

# 3. pass the result to cuDenseOffsets (and reduce --sw/--sh accordingly)
cuDenseOffsets ... --gross 1 --gross-file out/grossOffsets.bin
```

Check that the printed `the number of windows: N by M` matches what `cuDenseOffsets` reports.
Details of every step follow; `grossOffsets -h` lists all options.

---

## 1. How it works

For every reference window on the grid `cuDenseOffsets` will use:

1. **Geometry.** Window-centre latitude/longitude, incidence angle, flight and look azimuth are
   computed analytically from the GAMMA parameter file of the reference image (`--ref-par`):
   slant→ground range on the WGS84 ellipsoid (height 0), along-track displacement along the
   geodesic with the given `heading`, cross-track displacement to the right
   (`azimuth_angle = +90`). Alternatively they can be read from ISCE-style `--lat/--lon/--los`
   rasters.
2. **Model.** lon/lat → EPSG:3031 (`pyproj.Transformer`, `always_xy`), velocity `(VX, VY)` in m/yr
   sampled bilinearly (or `--interp nearest`) from the 450 m model; no-data cells (fill value 0)
   and points outside the model are invalid.
3. Rotate `(VX, VY)` to East/North, project onto the horizontal look direction (× sin incidence) and
   the flight direction, multiply by `interval / 365.25` and divide by the **slant**-range and
   azimuth pixel spacings → pixels.
4. Round to int32. Invalid windows get **0** (no shift), never garbage values.

**Speed.** The model (450 m) is coarser than a typical window grid, so the chain above is evaluated
on a coarse grid (~`--decimate-m` 400 m spacing) and linearly interpolated to every window; a
production grid of 2.46 M windows takes ~0.5–3 s and < 0.5 GB. `--no-decimate` evaluates every window.

## 2. One-time model preparation

The NetCDF file (6.8 GB, many variables) is converted once into a 1.2 GB float32 file that is
memory-mapped; each run reads only the cells covering the scene. This step needs the `netCDF4`
package (`conda install netcdf4`), nothing else does:

```bash
grossOffsets --prepare-model antarctica_ice_velocity_450m_v2.nc antarctica_vxvy_450m.npy
```

writes `antarctica_vxvy_450m.npy` and `antarctica_vxvy_450m.json` (grid, CRS, source, time
coverage). Keep both on the shared file system and pass `--model antarctica_vxvy_450m.npy`.

## 3. Options

### Geometry (one of)

| Option | Description |
|--------|-------------|
| `--ref-par`, `--sec-par` | GAMMA `.par` of the reference / secondary image. Used: `date`, `range_samples`, `azimuth_lines`, `range_pixel_spacing` (slant), `azimuth_pixel_spacing`, `near_range_slc`, `center_range_slc`, `center_latitude/longitude`, `heading`, `incidence_angle`, `azimuth_angle`. The interval is the difference of the two `date`s. |
| `--lat`, `--lon`, `--los`, `--los_scheme` | raster alternative (float64, float64, 2-band float32; `--los` must be GDAL readable; flight direction in the ISCE2 convention). Needs `--rps`, `--aps` (slant range, azimuth, metres) and `--interval`. |
| `--interval`, `--rps`, `--aps` | override the `.par` values |

### Window grid — must match `cuDenseOffsets`

Same names and semantics as `cuDenseOffsets`: `--ww --wh --sw --sh --kw --kh --mm --nwa --nwd`,
`--startpixelac/--spa`, `--startpixeldw/--spd`, `--endpixelac/--epa`, `--endpixeldw/--epd`,
`--use-center`. Defaults: start = `mm + half search`, end = `size - mm - half search`,
`n = (end - start - win) // skip`, window centre = `start + win//2 + i*skip`. Paste the exact option
string of the `cuDenseOffsets` command. The script prints `the number of windows: N by M` in the
same format.

### Output / performance

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir`, `--outname` | `.`, `grossOffsets.bin` | created if missing |
| `--decimate-m`, `--no-decimate` | 400 | coarse-grid spacing, see above |
| `--overwrite` | off | recompute even if `<outname>` and its `.json` signature are up to date |
| `--save-rasters`, `--plot` | off | ENVI float/int rasters (GDAL), PNG preview (matplotlib) |
| `-q` | | only warnings and errors |

## 4. Outputs

| File | Content |
|------|---------|
| `<outname>` | **Input for `cuDenseOffsets --gross 1 --gross-file`.** Raw int32 `(nDown*nAcross, 2)`, columns = (azimuth, range), row-major window order. Written atomically (temp file + rename), so an interrupted job never leaves a partial file. |
| `<outname>.json` | grid, interval, min/max, number of invalid windows and an input signature used to skip identical re-runs |

Sign convention: positive azimuth offset = secondary feature appears later in azimuth
(along the flight direction); positive range offset = further from the sensor — same as
`cuDenseOffsets`.

Exit status is non-zero with a one-line `ERROR grossOffsets: ...` on stderr for any problem (missing
file, empty grid, no window on valid data), so `set -e` / `trap ... ERR` in job scripts work.

## 5. Examples

### 5.1 Slurm job (one job per pair), chained to cuDenseOffsets

```bash
GRID="--ww 192 --wh 48 --sw 32 --sh 8 --kw 34 --kh 10 \
      --startpixeldw 15 --startpixelac 24 --endpixeldw 13052 --endpixelac 64774"

grossOffsets --model /path/antarctica_vxvy_450m.npy \
    --ref-par 003543.par --sec-par 003718.par $GRID \
    --outname 003543-003718.gross.bin

cuDenseOffsets -r 003543.slc -s 003718.slc --op 003543-003718.offmap_1 $GRID_AS_CUDENSE \
    --gross 1 --gross-file 003543-003718.gross.bin ...
```

(`$GRID_AS_CUDENSE` = the same `--ww/--wh/--sw/--sh/--kw/--kh/--startpixel*/--endpixel*` options.
Drop `--rr/--aa`: with a varying gross file the static values are not used.)

### 5.2 ISCE rasters instead of `.par`

```bash
grossOffsets --model antarctica_vxvy_450m.npy \
    --lat lat.rdr --lon lon.rdr --los los.rdr --los_scheme bil \
    --rps 2.33 --aps 14.1 --interval 12 --ww 64 --wh 64 --sw 10 --sh 10 --kw 32 --kh 32 --mm 50
```

### 5.3 Checking the result

```python
import numpy as np
g = np.fromfile('003543-003718.gross.bin', dtype=np.int32).reshape(-1, 2)
az = g[:, 0].reshape(nDown, nAcross)      # nDown, nAcross as printed by the script
print(g.min(0), g.max(0))
```

### 5.4 From Python

```python
from pycuampcor.grossOffsets import main
status = main(['--model', 'antarctica_vxvy_450m.npy', '--ref-par', 'a.par', '--sec-par', 'b.par'])
```

## 6. Choosing the search range

The model gives a *mean* velocity (metadata: 1995-2016 composite), so the residual to be found by
correlation is temporal variability plus model error. Inspect the spread of the offsets (`.json`,
`--plot`) and choose `--sw/--sh` somewhat larger than the expected residual (typically 4–32 px).

## 7. Caveats

* **Geometry from `.par` is approximate**: ellipsoid height 0 (surface elevation of a few km shifts
  the ground position by a few km), spherical triangle at the local radius of curvature, along-track
  geodesic with constant `heading` at the scene centre, scene centre at
  `(azimuth_lines/2, range_samples/2)`. This is far below the scale on which glacier velocity varies
  and the 450 m model resolution for the purpose of a *gross* offset, but it is not a geocoding.
  Use the raster input if you have precise geometry. Left-looking sensors work only if the `.par`
  has `azimuth_angle = -90`.
* Only vertical-free surface motion is modelled (no vertical velocity, no orbit/DEM coregistration
  shift between the two images).
* Points where the model has no data (fill value 0, ice-free, ocean, outside the grid) get gross
  offset 0 and a warning with the percentage. Decimated evaluation can move the valid/invalid
  boundary by up to one coarse step (~400 m).
* The `--rps`/range pixel size must be the **slant**-range spacing (what the `.par` gives).
* Previously the script wrote `grossRange.off`, `grossAzimuth_int.off`, etc. and PNGs by default and
  needed ISCE and a GDAL NetCDF driver; use `--save-rasters`/`--plot` for those. The old
  `--model_file` (the NetCDF) is replaced by `--model` (prepared file); `--model_file` is still accepted as
  an alias but must point to the prepared `.npy`.
