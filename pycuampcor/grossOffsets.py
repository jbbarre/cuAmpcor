#!/usr/bin/env python3
"""Per-window gross (integer pixel) offsets from the MEaSUREs Antarctica ice velocity model.

The output ``grossOffsets.bin`` (int32, shape (nWindows, 2) = [azimuth, range], C order) is meant to be fed to
``cuDenseOffsets --gross 1 --gross-file grossOffsets.bin``. It must be generated with the SAME window-grid options
(--ww --wh --sw --sh --kw --kh --mm --startpixel* --endpixel* --use-center --nwa --nwd) as the cuDenseOffsets run.

Radar geometry comes either from GAMMA ISP image parameter files (``--ref-par``/``--sec-par``, default; the
geometry is computed analytically on the WGS84 ellipsoid) or, alternatively, from ISCE-style lat/lon/los rasters
(``--lat --lon --los``).

Velocity model: MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 (NSIDC-0484,
doi:10.5067/D7GK8F5J8M8R). The NetCDF file is converted ONCE with ``--prepare-model`` into a compact float32
memory-mappable file; per-pair runs only read the small part of it that covers the scene.

Original author: Minyan Zhong.
"""
import os
import sys
import json
import hashlib
import argparse
import datetime
import logging

import numpy as np

__version__ = '2.0'

log = logging.getLogger('grossOffsets')

# WGS84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)

DAYS_PER_YEAR = 365.25
INT32_MAX = np.iinfo(np.int32).max
INT32_MIN = np.iinfo(np.int32).min

EXAMPLE = '''
Prepare the model once (needs the python package netCDF4):
  grossOffsets.py --prepare-model antarctica_ice_velocity_450m_v2.nc antarctica_vxvy_450m.npy

Per pair (same grid options as the cuDenseOffsets command):
  grossOffsets.py --model antarctica_vxvy_450m.npy --ref-par 003543.par --sec-par 003718.par \\
      --ww 192 --wh 48 --sw 32 --sh 8 --kw 34 --kh 10 \\
      --startpixeldw 15 --startpixelac 24 --endpixeldw 13052 --endpixelac 64774 \\
      --outname 003543-003718.gross.bin
  cuDenseOffsets.py ... --gross 1 --gross-file 003543-003718.gross.bin
'''


# --------------------------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------------------------
def createParser():
    p = argparse.ArgumentParser(
        description='Generate per-window integer gross offsets from the Antarctica ice velocity model.',
        formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)

    g = p.add_argument_group('velocity model')
    g.add_argument('--prepare-model', nargs=2, metavar=('NC_FILE', 'OUT_NPY'), dest='prepare_model',
                   help='convert the MEaSUREs NetCDF file (VX, VY) to a compact float32 .npy (+ .json) and exit')
    g.add_argument('--model', '--model_file', dest='model', type=str,
                   help='prepared model (.npy, see --prepare-model)')
    g.add_argument('--interp', choices=('bilinear', 'nearest'), default='bilinear',
                   help='model sampling (default: %(default)s)')

    g = p.add_argument_group('radar geometry: GAMMA parameter files')
    g.add_argument('--ref-par', dest='ref_par', type=str, help='GAMMA .par of the reference image')
    g.add_argument('--sec-par', dest='sec_par', type=str,
                   help='GAMMA .par of the secondary image (only its date is used, unless --interval is given)')

    g = p.add_argument_group('radar geometry: ISCE-style rasters (alternative to .par)')
    g.add_argument('--lat', type=str, help='latitude file (float64, one value per pixel)')
    g.add_argument('--lon', type=str, help='longitude file (float64)')
    g.add_argument('--los', type=str,
                   help='two band float32 raster. band1: incidence angle; band2: satellite flight direction '
                        '(ISCE2 convention, counter-clockwise from East)')
    g.add_argument('--los_scheme', type=str, default='bil', help='interleave of --los: bil, bsq or bip')
    g.add_argument('--aps', '--azimuthPixelSize', dest='azimuthPixelSize', type=float,
                   help='azimuth pixel size [m] (overrides the .par value)')
    g.add_argument('--rps', '--rangePixelSize', dest='rangePixelSize', type=float,
                   help='SLANT range pixel size [m] (overrides the .par value)')
    g.add_argument('--interval', type=float,
                   help='days between reference and secondary (default: from the .par dates)')

    g = p.add_argument_group('window grid (identical meaning as in cuDenseOffsets)')
    g.add_argument('--ww', type=int, dest='winwidth', default=64, help='Window width (default: %(default)s).')
    g.add_argument('--wh', type=int, dest='winhgt', default=64, help='Window height (default: %(default)s).')
    g.add_argument('--sw', type=int, dest='srcwidth', default=20,
                   help='Half search range along width (default: %(default)s).')
    g.add_argument('--sh', type=int, dest='srchgt', default=20,
                   help='Half search range along height (default: %(default)s).')
    g.add_argument('--kw', type=int, dest='skipwidth', default=64, help='Skip across (default: %(default)s).')
    g.add_argument('--kh', type=int, dest='skiphgt', default=64, help='Skip down (default: %(default)s).')
    g.add_argument('--mm', type=int, dest='margin', default=0, help='Margin (default: %(default)s).')
    g.add_argument('--nwa', type=int, dest='numWinAcross', default=-1,
                   help='Number of windows across (default: auto).')
    g.add_argument('--nwd', type=int, dest='numWinDown', default=-1, help='Number of windows down (default: auto).')
    g.add_argument('--spa', '--startpixelac', dest='startpixelac', type=int, default=None,
                   help='Starting pixel across (default: margin + half search range).')
    g.add_argument('--spd', '--startpixeldw', dest='startpixeldw', type=int, default=None,
                   help='Starting pixel down (default: margin + half search range).')
    g.add_argument('--epa', '--endpixelac', dest='endpixelac', type=int, default=None,
                   help='Ending pixel across (default: width - margin - half search range).')
    g.add_argument('--epd', '--endpixeldw', dest='endpixeldw', type=int, default=None,
                   help='Ending pixel down (default: length - margin - half search range).')
    g.add_argument('--use-center', dest='use_center', type=int, default=0,
                   help='the start/end pixels given are chip centers (default: %(default)s).')

    g = p.add_argument_group('output / performance')
    g.add_argument('--outdir', type=str, default='.', help='output directory (default: %(default)s)')
    g.add_argument('--outname', type=str, default='grossOffsets.bin',
                   help='output name of the int32 gross offsets (default: %(default)s)')
    g.add_argument('--decimate-m', dest='decimate_m', type=float, default=400.0,
                   help='evaluate the model on a coarse grid with about this ground spacing [m] and interpolate '
                        'linearly to every window (default: %(default)s; the model is 450 m)')
    g.add_argument('--no-decimate', dest='decimate_m', action='store_const', const=0.0,
                   help='evaluate every window exactly (slower)')
    g.add_argument('--overwrite', action='store_true', help='recompute even if an up-to-date output exists')
    g.add_argument('--save-rasters', action='store_true',
                   help='also write float and int Cloud Optimized GeoTIFF rasters (needs GDAL >= 3.1)')
    g.add_argument('--plot', action='store_true', help='also write a PNG preview (needs matplotlib)')
    g.add_argument('-q', '--quiet', action='store_true')
    return p


def cmdLineParse(iargs=None):
    return createParser().parse_args(args=iargs)


# --------------------------------------------------------------------------------------------------------------
# velocity model
# --------------------------------------------------------------------------------------------------------------
def prepare_model(nc_file, out_npy, chunk_rows=512):
    """Convert the MEaSUREs NetCDF (VX, VY) to a (2, ny, nx) float32 .npy with y ascending and NaN for no data."""
    try:
        import netCDF4
    except ImportError:
        raise RuntimeError('--prepare-model needs the python package netCDF4 (conda install netcdf4 / '
                           'pip install netCDF4). It is only required for this one-time conversion.')
    if not os.path.exists(nc_file):
        raise FileNotFoundError('{} not found. Download the MEaSUREs InSAR-Based Antarctica Ice Velocity Map, '
                                'Version 2 (NSIDC-0484)'.format(nc_file))
    out_npy = str(out_npy)
    tmp = out_npy + '.tmp.{}'.format(os.getpid())
    with netCDF4.Dataset(nc_file) as ds:
        x = np.asarray(ds['x'][:], dtype=np.float64)
        y = np.asarray(ds['y'][:], dtype=np.float64)
        nx, ny = x.size, y.size
        dx = float(np.mean(np.diff(x)))
        dy = float(np.mean(np.diff(y)))
        if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy)) or dx <= 0 or abs(dy) != dx:
            raise ValueError('unexpected x/y axes (non uniform, x not ascending or non square cells)')
        flip = dy < 0
        y_asc = y[::-1] if flip else y
        for name in ('VX', 'VY'):
            ds[name].set_auto_maskandscale(False)
            if ds[name].shape != (ny, nx):
                raise ValueError('{} has shape {}, expected {}'.format(name, ds[name].shape, (ny, nx)))
        out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(2, ny, nx))
        for r0 in range(0, ny, chunk_rows):
            r1 = min(ny, r0 + chunk_rows)
            vx = np.asarray(ds['VX'][r0:r1, :], dtype=np.float32)
            vy = np.asarray(ds['VY'][r0:r1, :], dtype=np.float32)
            bad = ~np.isfinite(vx) | ~np.isfinite(vy) | ((vx == 0) & (vy == 0))  # _FillValue is 0
            vx[bad] = np.nan
            vy[bad] = np.nan
            if flip:
                d0, d1 = ny - r1, ny - r0
                out[0, d0:d1, :] = vx[::-1]
                out[1, d0:d1, :] = vy[::-1]
            else:
                out[0, r0:r1, :] = vx
                out[1, r0:r1, :] = vy
        out.flush()
        del out
        meta = {'x0': float(x[0]), 'y0': float(y_asc[0]), 'dx': dx, 'dy': dx, 'nx': nx, 'ny': ny,
                'crs': 'EPSG:3031', 'lon0': 0.0, 'units': 'm/yr', 'layout': '(2[VX,VY], ny[y ascending], nx)',
                'source': os.path.basename(nc_file),
                'time_coverage': [getattr(ds, 'time_coverage_start', None), getattr(ds, 'time_coverage_end', None)],
                'prepared_by': 'grossOffsets ' + __version__}
    os.replace(tmp, out_npy)
    _atomic_write_text(_model_meta_path(out_npy), json.dumps(meta, indent=1))
    log.info('model written: %s (%d x %d, %.0f m)', out_npy, nx, ny, dx)


def _model_meta_path(npy):
    return os.path.splitext(npy)[0] + '.json'


class VelocityModel:
    def __init__(self, path):
        if path.lower().endswith('.nc'):
            raise ValueError('{} is a NetCDF file: convert it once with --prepare-model'.format(path))
        if not os.path.exists(path):
            raise FileNotFoundError('model file {} not found'.format(path))
        with open(_model_meta_path(path)) as f:
            self.meta = json.load(f)
        self.path = path
        self.data = np.load(path, mmap_mode='r')
        m = self.meta
        if self.data.shape != (2, m['ny'], m['nx']):
            raise ValueError('model {} does not match its .json description'.format(path))
        self.x0, self.y0, self.d = m['x0'], m['y0'], m['dx']
        self.lon0 = m.get('lon0', 0.0)
        self.crs = m.get('crs', 'EPSG:3031')

    def signature(self):
        st = os.stat(self.path)
        return {'path': os.path.abspath(self.path), 'size': st.st_size, 'mtime': int(st.st_mtime)}

    def sample(self, x, y, method='bilinear'):
        """Velocity (vx, vy) [m/yr] at polar-stereographic coordinates; NaN where not available."""
        shape = x.shape
        fx = (np.ravel(x) - self.x0) / self.d
        fy = (np.ravel(y) - self.y0) / self.d
        vx = np.full(fx.shape, np.nan, dtype=np.float32)
        vy = np.full(fx.shape, np.nan, dtype=np.float32)
        ok = np.isfinite(fx) & np.isfinite(fy)
        if method == 'nearest':
            ix0 = np.where(ok, np.rint(np.where(ok, fx, 0)), -1).astype(np.int64)
            iy0 = np.where(ok, np.rint(np.where(ok, fy, 0)), -1).astype(np.int64)
            corners = [(0, 0, None)]
        else:
            ix0 = np.where(ok, np.floor(np.where(ok, fx, 0)), -2).astype(np.int64)
            iy0 = np.where(ok, np.floor(np.where(ok, fy, 0)), -2).astype(np.int64)
            tx, ty = fx - ix0, fy - iy0
            corners = [(0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                       (0, 1, (1 - tx) * ty), (1, 1, tx * ty)]
        nx, ny = self.meta['nx'], self.meta['ny']
        inside = ok & (ix0 >= -1) & (ix0 <= nx - 1) & (iy0 >= -1) & (iy0 <= ny - 1)
        if not inside.any():
            return vx.reshape(shape), vy.reshape(shape)
        c0 = max(int(ix0[inside].min()), 0)
        c1 = min(int(ix0[inside].max()) + 2, nx)
        r0 = max(int(iy0[inside].min()), 0)
        r1 = min(int(iy0[inside].max()) + 2, ny)
        if c1 <= c0 or r1 <= r0:
            return vx.reshape(shape), vy.reshape(shape)
        sub = np.asarray(self.data[:, r0:r1, c0:c1])
        accx = np.zeros(fx.shape, np.float64)
        accy = np.zeros(fx.shape, np.float64)
        wsum = np.zeros(fx.shape, np.float64)
        for cx, cy, w in corners:
            ix, iy = ix0 + cx, iy0 + cy
            m = inside & (ix >= c0) & (ix < c1) & (iy >= r0) & (iy < r1)
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                continue
            ax = sub[0, iy[idx] - r0, ix[idx] - c0]
            ay = sub[1, iy[idx] - r0, ix[idx] - c0]
            good = np.isfinite(ax) & np.isfinite(ay)
            idx, ax, ay = idx[good], ax[good], ay[good]
            ww = np.ones(idx.size) if w is None else w[idx]
            accx[idx] += ww * ax
            accy[idx] += ww * ay
            wsum[idx] += ww
        # keep a point only if at least half of its interpolation weight has valid data
        good = wsum >= 0.5
        vx[good] = (accx[good] / wsum[good]).astype(np.float32)
        vy[good] = (accy[good] / wsum[good]).astype(np.float32)
        return vx.reshape(shape), vy.reshape(shape)


# --------------------------------------------------------------------------------------------------------------
# window grid (mirrors cuDenseOffsets.py)
# --------------------------------------------------------------------------------------------------------------
def window_grid(a, length, width):
    """Window-centre pixel coordinates (rows [down], cols [across]) as cuDenseOffsets defines them."""
    hh, hw = a.winhgt // 2, a.winwidth // 2
    sd = a.startpixeldw - (hh if a.use_center else 0) if a.startpixeldw is not None else a.margin + a.srchgt
    sa = a.startpixelac - (hw if a.use_center else 0) if a.startpixelac is not None else a.margin + a.srcwidth
    ed = a.endpixeldw + (hh if a.use_center else 0) if a.endpixeldw is not None else length - a.margin - a.srchgt
    ea = a.endpixelac + (hw if a.use_center else 0) if a.endpixelac is not None else width - a.margin - a.srcwidth
    nd = a.numWinDown if a.numWinDown > 0 else (ed - sd - a.winhgt) // a.skiphgt
    na = a.numWinAcross if a.numWinAcross > 0 else (ea - sa - a.winwidth) // a.skipwidth
    if nd <= 0 or na <= 0:
        raise ValueError('empty window grid ({} x {}) for an image of {} x {}'.format(nd, na, length, width))
    if min(a.skiphgt, a.skipwidth, a.winhgt, a.winwidth) <= 0:
        raise ValueError('window and skip sizes must be positive')
    rows = sd + hh + a.skiphgt * np.arange(nd, dtype=np.int64)
    cols = sa + hw + a.skipwidth * np.arange(na, dtype=np.int64)
    if sd < 0 or sa < 0 or rows[-1] - hh + a.winhgt > length or cols[-1] - hw + a.winwidth > width:
        log.warning('Warning: the window grid is not fully inside the %d x %d image', length, width)
    info = {'start_down': int(sd), 'start_across': int(sa), 'n_down': int(nd), 'n_across': int(na),
            'win_height': a.winhgt, 'win_width': a.winwidth, 'skip_down': a.skiphgt, 'skip_across': a.skipwidth,
            'half_search_down': a.srchgt, 'half_search_across': a.srcwidth, 'length': int(length), 'width': int(width)}
    return rows, cols, info


# --------------------------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------------------------
def read_par(path):
    """GAMMA parameter file -> {key: [tokens]}."""
    par = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            toks = val.split()
            if toks:
                par[key.strip()] = toks
    return par


def scene_from_par(path):
    par = read_par(path)

    def num(key, cast=float):
        try:
            return cast(par[key][0])
        except (KeyError, ValueError, IndexError):
            raise ValueError('{}: missing or invalid "{}"'.format(path, key))

    try:
        y, mo, d, h, mi, s = [float(t) for t in par['date'][:6]]
        date = datetime.datetime(int(y), int(mo), int(d), int(h), int(mi)) + datetime.timedelta(seconds=s)
    except (KeyError, ValueError, IndexError):
        raise ValueError('{}: missing or invalid "date"'.format(path))
    sc = {'width': num('range_samples', int), 'length': num('azimuth_lines', int),
          'dr': num('range_pixel_spacing'), 'daz': num('azimuth_pixel_spacing'),
          'near': num('near_range_slc'), 'center_range': num('center_range_slc'),
          'lat0': num('center_latitude'), 'lon0': num('center_longitude'),
          'heading': num('heading'), 'inc': num('incidence_angle'), 'azangle': num('azimuth_angle'),
          'date': date}
    if str(par.get('image_geometry', ['SLANT_RANGE'])[0]).upper() != 'SLANT_RANGE':
        raise ValueError('{}: only SLANT_RANGE images are supported'.format(path))
    if abs(abs(sc['azangle']) - 90.0) > 5.0:
        raise ValueError('{}: unexpected azimuth_angle {} (expected +-90)'.format(path, sc['azangle']))
    return sc


def _earth_radius(lat_deg):
    """Gaussian mean radius of curvature of the WGS84 ellipsoid [m]."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    w = 1.0 - _E2 * s2
    return np.sqrt(_A * (1.0 - _E2) / w ** 1.5 * _A / np.sqrt(w))


def par_geometry(sc, rows, cols):
    """Latitude/longitude/incidence/azimuths of the given window centres from a GAMMA par description.

    Spherical-triangle slant/ground range relation at the local radius of curvature, ellipsoid height 0, scene
    centre at (azimuth_lines/2, range_samples/2); along-track displacement follows the geodesic with the given
    heading, cross-track displacement is perpendicular (right looking for azimuth_angle=+90).
    """
    import pyproj
    geod = pyproj.Geod(a=_A, f=_F)
    re = float(_earth_radius(sc['lat0']))
    rc = sc['center_range']
    rs2 = re ** 2 + rc ** 2 + 2.0 * re * rc * np.cos(np.radians(sc['inc']))
    rs = np.sqrt(rs2)

    def ground(rng):
        return re * np.arccos(np.clip((re ** 2 + rs2 - rng ** 2) / (2.0 * re * rs), -1.0, 1.0))

    rng = sc['near'] + cols * sc['dr']
    theta = np.arccos(np.clip((rs2 - re ** 2 - rng ** 2) / (2.0 * re * rng), -1.0, 1.0))
    cross = ground(rng) - ground(rc)                       # [m], negative towards near range
    along = (rows - sc['length'] / 2.0) * sc['daz']        # [m], negative before the scene centre
    n = rows.size
    lon1, lat1, baz = geod.fwd(np.full(n, sc['lon0']), np.full(n, sc['lat0']), np.full(n, sc['heading']), along)
    flight = (np.asarray(baz) - 180.0) % 360.0
    look = flight + np.sign(sc['azangle']) * 90.0
    lon1, lat1, look_b = np.broadcast_arrays(np.asarray(lon1)[:, None], np.asarray(lat1)[:, None], look[:, None])
    cross_b = np.broadcast_to(cross[None, :], (n, cols.size))
    lon2, lat2, _ = geod.fwd(np.ascontiguousarray(np.broadcast_to(lon1, cross_b.shape)).ravel(),
                             np.ascontiguousarray(np.broadcast_to(lat1, cross_b.shape)).ravel(),
                             np.ascontiguousarray(np.broadcast_to(look_b, cross_b.shape)).ravel(),
                             np.ascontiguousarray(cross_b).ravel())
    shape = cross_b.shape
    return {'lon': np.asarray(lon2).reshape(shape), 'lat': np.asarray(lat2).reshape(shape),
            'theta': theta[None, :], 'flight': np.radians(flight)[:, None], 'look': np.radians(look)[:, None],
            'invalid': None}


class RasterGeometry:
    """ISCE-style lat/lon/los rasters (float64, float64, 2 x float32)."""

    def __init__(self, a):
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(a.los)
        if ds is None:
            raise ValueError('cannot open {}'.format(a.los))
        self.width, self.length = ds.RasterXSize, ds.RasterYSize
        ds = None
        self.scheme = a.los_scheme.lower()
        if self.scheme not in ('bil', 'bsq', 'bip'):
            raise ValueError('--los_scheme must be bil, bsq or bip')
        n = self.width * self.length
        for fname, nbytes in ((a.lat, 8 * n), (a.lon, 8 * n), (a.los, 8 * n)):
            if os.path.getsize(fname) != nbytes:
                raise ValueError('{}: size {} does not match {} x {} pixels'.format(
                    fname, os.path.getsize(fname), self.width, self.length))
        shp = {'bil': (self.length, 2, self.width), 'bsq': (2, self.length, self.width),
               'bip': (self.length, self.width, 2)}[self.scheme]
        self.lat = np.memmap(a.lat, dtype='float64', mode='r', shape=(self.length, self.width))
        self.lon = np.memmap(a.lon, dtype='float64', mode='r', shape=(self.length, self.width))
        los = np.memmap(a.los, dtype='float32', mode='r', shape=shp)
        self.inc, self.azi = {'bil': (los[:, 0, :], los[:, 1, :]), 'bsq': (los[0], los[1]),
                              'bip': (los[:, :, 0], los[:, :, 1])}[self.scheme]

    def __call__(self, rows, cols):
        ix = np.ix_(rows, cols)
        inc = np.asarray(self.inc[ix], dtype=np.float64)
        azi = np.asarray(self.azi[ix], dtype=np.float64)
        flight = 90.0 - azi                       # compass angle (clockwise from North) from ISCE (CCW from East)
        return {'lon': np.asarray(self.lon[ix]), 'lat': np.asarray(self.lat[ix]), 'theta': np.radians(inc),
                'flight': np.radians(flight), 'look': np.radians(flight + 90.0), 'invalid': inc == 0}


# --------------------------------------------------------------------------------------------------------------
# computation
# --------------------------------------------------------------------------------------------------------------
def coarse_indices(n, step):
    if step <= 1 or n <= 2:
        return np.arange(n)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _interp_axis(arr, xc, x, axis):
    if xc.size < 2:
        return np.take(arr, np.zeros(x.size, dtype=np.int64), axis=axis)
    i1 = np.clip(np.searchsorted(xc, x), 1, xc.size - 1)
    i0 = i1 - 1
    shape = [1] * arr.ndim
    shape[axis] = -1
    w = ((x - xc[i0]) / (xc[i1] - xc[i0])).astype(np.float32).reshape(shape)
    return np.take(arr, i0, axis=axis) * (1.0 - w) + np.take(arr, i1, axis=axis) * w


def upsample(vals, ir, ic, nr, nc):
    """Separable linear interpolation of a NaN-holding coarse field to all windows (weights treat NaN as missing)."""
    w = np.isfinite(vals).astype(np.float32)
    v = np.where(w > 0, vals, 0).astype(np.float32)
    xr, xc = np.arange(nr), np.arange(nc)
    out = []
    for arr in (v, w):
        t = _interp_axis(arr, ic.astype(np.float64), xc, 1)
        out.append(_interp_axis(t, ir.astype(np.float64), xr, 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        res = out[0] / out[1]
    res[out[1] < 0.5] = np.nan
    return res


def project(vx, vy, geo, lon0, dr, daz, interval_days):
    """Polar stereographic velocity -> expected offsets [pixels] in azimuth and (slant) range."""
    lam = np.radians(geo['lon'] - lon0)
    ve = vx * np.cos(lam) - vy * np.sin(lam)
    vn = vx * np.sin(lam) + vy * np.cos(lam)
    f = interval_days / DAYS_PER_YEAR
    v_rng = (ve * np.sin(geo['look']) + vn * np.cos(geo['look'])) * np.sin(geo['theta'])  # + = away from sensor
    v_az = ve * np.sin(geo['flight']) + vn * np.cos(geo['flight'])                        # + = along flight
    return v_az * f / daz, v_rng * f / dr


def estimate_offsets(a, model, geom, rows, cols, dr, daz, inc_hint_deg, interval_days):
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', model.crs, always_xy=True)
    if a.decimate_m > 0:
        step_r = max(1, int(a.decimate_m // (a.skiphgt * daz)))
        step_c = max(1, int(a.decimate_m // (a.skipwidth * dr / np.sin(np.radians(inc_hint_deg)))))
    else:
        step_r = step_c = 1
    ir, ic = coarse_indices(rows.size, step_r), coarse_indices(cols.size, step_c)
    log.info('evaluating %d x %d of %d x %d windows (steps %d, %d)', ir.size, ic.size, rows.size, cols.size,
             step_r, step_c)
    geo = geom(rows[ir], cols[ic])
    shape = geo['lon'].shape
    x, y = tr.transform(np.ravel(geo['lon']), np.ravel(geo['lat']))
    vx, vy = model.sample(np.asarray(x).reshape(shape), np.asarray(y).reshape(shape), a.interp)
    az, rg = project(vx.astype(np.float64), vy.astype(np.float64), geo, model.lon0, dr, daz, interval_days)
    if geo['invalid'] is not None:
        az[geo['invalid']] = np.nan
        rg[geo['invalid']] = np.nan
    if ir.size != rows.size or ic.size != cols.size:
        az = upsample(az, ir, ic, rows.size, cols.size)
        rg = upsample(rg, ir, ic, rows.size, cols.size)
    return az.astype(np.float32), rg.astype(np.float32)


def to_int32(x):
    """Round; invalid (NaN) and out-of-range values become 0 (no gross shift) - never garbage integers."""
    ok = np.isfinite(x) & (np.abs(x) < 2 ** 30)
    out = np.zeros(x.shape, dtype=np.int32)
    out[ok] = np.rint(x[ok]).astype(np.int32)
    return out, ok


# --------------------------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------------------------
def _atomic_write_text(path, text):
    tmp = '{}.tmp.{}'.format(path, os.getpid())
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_bin(path, arr):
    tmp = '{}.tmp.{}'.format(path, os.getpid())
    try:
        with open(tmp, 'wb') as f:
            arr.tofile(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_rasters(outdir, stem, az_f, rg_f, az_i, rg_i):
    """Write the gross-offset grids as Cloud Optimized GeoTIFFs (GDAL COG driver, GDAL >= 3.1)."""
    from osgeo import gdal
    mem = gdal.GetDriverByName('MEM')
    cog = gdal.GetDriverByName('COG')
    if cog is None:
        raise RuntimeError('GDAL COG driver not available (needs GDAL >= 3.1)')
    for name, arr, typ in (('grossAzimuth', az_f, gdal.GDT_Float32), ('grossRange', rg_f, gdal.GDT_Float32),
                           ('grossAzimuth_int', az_i, gdal.GDT_Int32), ('grossRange_int', rg_i, gdal.GDT_Int32)):
        src = mem.Create('', arr.shape[1], arr.shape[0], 1, typ)
        src.GetRasterBand(1).WriteArray(arr)
        out = os.path.join(outdir, '{}_{}.tif'.format(stem, name))
        dst = cog.CreateCopy(out, src, options=['COMPRESS=DEFLATE', 'BLOCKSIZE=256'])
        if dst is None:
            raise RuntimeError('failed to write ' + out)
        dst = None
        src = None


def save_plot(path, az_f, rg_f):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    for ax, arr, title in zip(axs, (az_f, rg_f), ('gross azimuth offset', 'gross range offset')):
        lim = np.nanpercentile(np.abs(arr), 99) if np.isfinite(arr).any() else 1.0
        im = ax.imshow(arr, cmap='coolwarm', vmin=-lim, vmax=lim, aspect='auto', interpolation='nearest')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label='pixel')
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _signature(a, sc, info, interval, dr, daz, model):
    d = {'version': __version__, 'grid': info, 'interval': round(float(interval), 9), 'dr': dr, 'daz': daz,
         'interp': a.interp, 'decimate_m': a.decimate_m, 'model': model.signature()}
    if sc is not None:
        d['scene'] = {k: str(v) for k, v in sc.items()}
    else:
        d['rasters'] = [{'path': os.path.abspath(p), 'size': os.path.getsize(p), 'mtime': int(os.path.getmtime(p))}
                        for p in (a.lat, a.lon, a.los)] + [a.los_scheme]
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def run(a):
    model = VelocityModel(a.model)
    sc_ref = sc_sec = None
    if a.ref_par:
        if a.lat or a.lon or a.los:
            raise ValueError('use either --ref-par or --lat/--lon/--los, not both')
        sc_ref = scene_from_par(a.ref_par)
        length, width = sc_ref['length'], sc_ref['width']
        dr = a.rangePixelSize or sc_ref['dr']
        daz = a.azimuthPixelSize or sc_ref['daz']
        inc_hint = sc_ref['inc']
        if a.interval is not None:
            interval = a.interval
        elif a.sec_par:
            sc_sec = scene_from_par(a.sec_par)
            interval = (sc_sec['date'] - sc_ref['date']).total_seconds() / 86400.0
        else:
            raise ValueError('give --sec-par or --interval')
        geom = lambda r, c: par_geometry(sc_ref, r, c)
    elif a.lat and a.lon and a.los:
        if not (a.rangePixelSize and a.azimuthPixelSize and a.interval is not None):
            raise ValueError('raster geometry needs --rps, --aps and --interval')
        rg_geom = RasterGeometry(a)
        length, width = rg_geom.length, rg_geom.width
        dr, daz, interval, inc_hint = a.rangePixelSize, a.azimuthPixelSize, a.interval, 38.0
        geom = rg_geom
    else:
        raise ValueError('radar geometry missing: give --ref-par (GAMMA) or --lat --lon --los')

    rows, cols, info = window_grid(a, length, width)
    nd, na = rows.size, cols.size
    print('image length: {}  width: {}'.format(length, width))
    print('the number of windows: {} by {}'.format(nd, na))
    print('the center pixel of the first window is : ({}, {})'.format(rows[0], cols[0]))
    log.info('interval %.4f days, pixel size %.4f (range) x %.4f (azimuth) m', interval, dr, daz)

    out = os.path.join(a.outdir, a.outname)
    os.makedirs(a.outdir, exist_ok=True)
    sig = _signature(a, sc_ref, info, interval, dr, daz, model)
    side = out + '.json'
    if not a.overwrite and os.path.exists(out) and os.path.exists(side) and os.path.getsize(out) == nd * na * 8:
        try:
            with open(side) as f:
                if json.load(f).get('signature') == sig:
                    log.info('%s is up to date, skipping (use --overwrite to recompute)', out)
                    return 0
        except (OSError, ValueError):
            pass

    az_f, rg_f = estimate_offsets(a, model, geom, rows, cols, dr, daz, inc_hint, interval)
    az_i, ok_a = to_int32(az_f)
    rg_i, ok_r = to_int32(rg_f)
    valid = ok_a & ok_r
    az_i[~valid] = 0
    rg_i[~valid] = 0
    if not valid.any():
        raise ValueError('no window falls on valid velocity data: wrong inputs or scene outside the model/ice')
    n_inv = int((~valid).sum())
    if n_inv:
        log.warning('Warning: %d of %d windows (%.1f %%) have no velocity data; their gross offset is set to 0',
                    n_inv, valid.size, 100.0 * n_inv / valid.size)

    data = np.empty((nd * na, 2), dtype=np.int32)
    data[:, 0] = az_i.ravel()
    data[:, 1] = rg_i.ravel()
    _atomic_write_bin(out, data)
    stats = {'azimuth_min': int(az_i.min()), 'azimuth_max': int(az_i.max()),
             'range_min': int(rg_i.min()), 'range_max': int(rg_i.max()), 'invalid_windows': n_inv}
    _atomic_write_text(side, json.dumps({'signature': sig, 'version': __version__, 'grid': info,
                                         'interval_days': interval, 'stats': stats,
                                         'layout': 'int32 (n_down*n_across, 2) = [azimuth, range]'}, indent=1))
    log.info('gross azimuth offset [%d, %d] pixels, range [%d, %d] pixels -> %s', stats['azimuth_min'],
             stats['azimuth_max'], stats['range_min'], stats['range_max'], out)

    stem = os.path.splitext(out)[0]
    if a.save_rasters:
        save_rasters(a.outdir, os.path.basename(stem), az_f, rg_f, az_i, rg_i)
    if a.plot:
        save_plot(stem + '.png', az_f, rg_f)
    return 0


def main(iargs=None):
    a = cmdLineParse(iargs)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format='%(message)s')
    try:
        if a.prepare_model:
            prepare_model(*a.prepare_model)
            return 0
        if not a.model:
            raise ValueError('--model is required (create it once with --prepare-model)')
        return run(a)
    except (ValueError, OSError, RuntimeError, ImportError) as e:
        log.error('ERROR grossOffsets: %s', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
