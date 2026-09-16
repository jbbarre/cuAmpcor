#!/usr/bin/env python3
"""Convert cuAmpcor (GPU) output rasters into the legacy ROI_PAC ampcor.F (CPU) text table.

cuAmpcor writes its results as flat BIP binary rasters (offset, snr, cov, ...) with no
per-window location column, since the window grid geometry is regular and fully described
by a handful of scalar parameters. The legacy ampcor.F CPU tool instead emits one text line
per window with the format (Fortran statement label 151):

    range_loc range_offset azimuth_loc azimuth_offset SNR    cov_xx     cov_yy     cov_xy

    151 format(1x,i7,1x,f9.3,1x,i7,1x,f11.3,1x,f10.5,1x,f10.6,1x,f10.6,1x,f10.6)

This script reads the cuAmpcor output rasters, reconstructs the azimuth/range pixel location
of each window from the run's geometry parameters, and writes out a text table that
reproduces that exact layout: no header line, identical field widths, and the same
asterisk-overflow behavior Fortran uses when a value doesn't fit its field -- this matters
because downstream GAMMA tooling may rely on fixed column positions, and a value that
overflows its Python format silently shifts every later column on that line.
"""

import argparse
import os
import re
import numpy as np


ISCE_DATATYPE_TO_NUMPY = {
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
}


def parse_isce_xml(xmlPath):
    """Extract width/length/bands/dtype from an ISCE image .xml sidecar, if present."""
    if not os.path.isfile(xmlPath):
        return None
    with open(xmlPath) as f:
        content = f.read()

    def findProperty(name):
        m = re.search(
            r'<property\s+name="{}"\s*>\s*<value>(.*?)</value>'.format(name),
            content, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    width = findProperty('WIDTH')
    length = findProperty('LENGTH')
    bands = findProperty('NUMBER_BANDS')
    dtype = findProperty('DATA_TYPE')
    if width is None or length is None:
        return None
    return {
        'width': int(width),
        'length': int(length),
        'bands': int(bands) if bands is not None else None,
        'dtype': ISCE_DATATYPE_TO_NUMPY.get(dtype.upper()) if dtype else None,
    }


def readBip(path, bands, width=None, length=None, dtype=None):
    """Read a BIP raster, using its .xml sidecar (if any) to fill in missing width/length/dtype."""
    meta = parse_isce_xml(path + '.xml')
    if meta is not None:
        width = width or meta['width']
        length = length or meta['length']
        dtype = dtype or meta['dtype']

    if width is None or length is None:
        raise ValueError(
            f'Cannot determine width/length for {path} (no .xml sidecar found); '
            'pass --width/--length explicitly.')
    # cuAmpcor's offset/snr/cov rasters are always single precision (float/float2/float3
    # in the CUDA source -- there is no double-precision build path), so default here
    # and treat anything else as a user error rather than silently reinterpreting bytes.
    dtype = dtype or np.float32
    if dtype not in (np.float32,):
        raise ValueError(
            f'{path}: dtype {dtype} was requested, but cuAmpcor output rasters are always '
            'float32. Do not pass --double unless you know your build actually differs.')

    data = np.fromfile(path, dtype=dtype)
    expected = width * length * bands
    if data.size != expected:
        raise ValueError(
            f'{path}: expected {expected} elements ({length}x{width}x{bands} {dtype.__name__}), '
            f'got {data.size}. Check --width/--length.')
    return data.reshape(length, width, bands), width, length


def fortranField(value, width, decimals):
    """Format a float the way Fortran's Fw.d edit descriptor does: right-justified in a
    field of `width` characters, or `width` asterisks if the formatted number (including
    sign and decimal point) would not fit. This preserves column alignment for any
    downstream fixed-column reader, exactly like the legacy ampcor.F output does.
    """
    text = '{:.{prec}f}'.format(value, prec=decimals)
    if len(text) > width:
        return '*' * width
    return text.rjust(width)


def fortranIntField(value, width):
    """Format an int right-justified in `width` chars, or asterisks if it overflows."""
    text = '{:d}'.format(value)
    if len(text) > width:
        return '*' * width
    return text.rjust(width)


def createParser():
    parser = argparse.ArgumentParser(
        description='Convert cuAmpcor (GPU) output rasters into the legacy ampcor.F (CPU) text format.',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--prefix', type=str,
                         help='cuAmpcor output prefix, e.g. the --outprefix given to cuDenseOffsets '
                              '(offset/snr/cov filenames are derived as prefix+".bip"/"_snr.bip"/"_cov.bip").')
    parser.add_argument('--offset', type=str, help='Path to the offset .bip file (2 bands: down, across). '
                         'Overrides the name derived from --prefix.')
    parser.add_argument('--snr', type=str, help='Path to the SNR .bip file (1 band). Overrides --prefix-derived name.')
    parser.add_argument('--cov', type=str, help='Path to the covariance .bip file (3 bands: cov_xx, cov_yy, cov_xy). '
                         'Overrides --prefix-derived name.')
    parser.add_argument('--gross', type=str, default=None,
                         help='Path to the gross offset .bip file (2 bands). If given, it is added to the '
                              'offset before output (use when mergeGrossOffset was NOT enabled at run time -- '
                              'legacy ampcor.F always reports the TOTAL offset, gross + residual, so this is '
                              'required for parity unless you ran cuDenseOffsets with --mg 1).')

    parser.add_argument('--width', type=int, default=None, help='Number of windows across (columns). '
                         'Auto-detected from a .xml sidecar if present.')
    parser.add_argument('--length', type=int, default=None, help='Number of windows down (rows). '
                         'Auto-detected from a .xml sidecar if present.')

    geom = parser.add_argument_group(
        'Window geometry',
        'Used to reconstruct azimuth_loc/range_loc pixel coordinates (window CENTER pixel, matching '
        'legacy i_centerxi/i_centeryi). These should match the values passed to cuDenseOffsets '
        '(--startpixeldw/ac, --kh/--kw, --wh/--ww).')
    geom.add_argument('--start-down', type=int, default=None,
                       help='referenceStartPixelDownStatic: first window start pixel, down/azimuth direction.')
    geom.add_argument('--start-across', type=int, default=None,
                       help='referenceStartPixelAcrossStatic: first window start pixel, across/range direction.')
    geom.add_argument('--skip-down', type=int, default=None, help='skipSampleDown between windows.')
    geom.add_argument('--skip-across', type=int, default=None, help='skipSampleAcross between windows.')
    geom.add_argument('--win-height', type=int, default=None, help='windowSizeHeight (down direction).')
    geom.add_argument('--win-width', type=int, default=None, help='windowSizeWidth (across direction).')

    qc = parser.add_argument_group(
        'Quality filtering',
        'Legacy ampcor.F only ever emits a line for a window that passes BOTH its SNR threshold '
        'AND its covariance threshold, and is not an edge/no-data window. cuAmpcor emits every '
        'window unconditionally, so replicate the checks explicitly here. NaN windows (out-of-range '
        'or zero-filled chunks, or a normalization divide-by-zero) are always dropped -- these are '
        'the closest cuAmpcor equivalent to legacy\'s edge/no-data exclusion, so this recovers most '
        'of that behavior even though it is not a byte-for-byte reconstruction of it.')
    qc.add_argument('--min-snr', type=float, default=None,
                     help='Drop windows with SNR below this threshold.')
    qc.add_argument('--max-cov', type=float, default=None,
                     help='Drop windows where cov_xx or cov_yy exceeds this value (matches legacy\'s '
                          'covariance threshold check).')
    qc.add_argument('--keep-nan', action='store_true',
                     help='Do NOT drop windows with a NaN in offset/snr/cov (dropped by default -- '
                          'a NaN cannot be printed into a fixed-width Fortran-style field anyway, '
                          'and would otherwise corrupt column alignment for that row).')

    parser.add_argument('--header', action='store_true',
                         help='Write a commented header line (prefixed with "#") before the data. '
                              'OFF by default: the legacy file has no header, and some downstream '
                              'GAMMA-side readers may not tolerate one -- only enable this if you have '
                              'verified your specific downstream tool accepts it.')
    parser.add_argument('-o', '--output', type=str, default=None,
                         help='Output text file path (default: <prefix>.ampcorCPU.txt).')

    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)

    if inps.prefix is None and (inps.offset is None or inps.snr is None or inps.cov is None):
        parser.error('Either --prefix or all of --offset/--snr/--cov must be given.')

    inps.offset = inps.offset or f'{inps.prefix}.bip'
    inps.snr = inps.snr or f'{inps.prefix}_snr.bip'
    inps.cov = inps.cov or f'{inps.prefix}_cov.bip'

    haveGeom = [inps.start_down, inps.start_across, inps.skip_down, inps.skip_across,
                inps.win_height, inps.win_width]
    if any(v is not None for v in haveGeom) and not all(v is not None for v in haveGeom):
        parser.error('--start-down/--start-across/--skip-down/--skip-across/--win-height/--win-width '
                      'must all be given together.')
    if not all(v is not None for v in haveGeom):
        parser.error('Window geometry (--start-down/--start-across/--skip-down/--skip-across/'
                      '--win-height/--win-width) is required: without it, azimuth_loc/range_loc '
                      'cannot be reconstructed as real pixel coordinates, and GAMMA needs real '
                      'pixel coordinates, not window indices.')

    if inps.output is None:
        base = inps.prefix or os.path.splitext(inps.offset)[0]
        inps.output = f'{base}.ampcorCPU.txt'

    return inps


def convert(inps):
    offset, width, length = readBip(inps.offset, bands=2, width=inps.width, length=inps.length)
    snr, _, _ = readBip(inps.snr, bands=1, width=width, length=length)
    cov, _, _ = readBip(inps.cov, bands=3, width=width, length=length)

    # cuAmpcor's internal convention: band 0 = down/azimuth, band 1 = across/range.
    # Legacy ampcor.F prints range (across) FIRST, then azimuth (down) -- the columns
    # are swapped relative to cuAmpcor's own array layout. This mapping is correct;
    # do not "simplify" it by writing bands in their raw order.
    downOffset = offset[:, :, 0].astype(np.float64)
    acrossOffset = offset[:, :, 1].astype(np.float64)

    if inps.gross:
        gross, _, _ = readBip(inps.gross, bands=2, width=width, length=length)
        downOffset += gross[:, :, 0]
        acrossOffset += gross[:, :, 1]

    snr = snr[:, :, 0].astype(np.float64)
    covXX = cov[:, :, 0].astype(np.float64)
    covYY = cov[:, :, 1].astype(np.float64)
    covXY = cov[:, :, 2].astype(np.float64)

    # Window CENTER pixel, matching legacy's i_centerxi/i_centeryi = start + (size-1)/2
    azimuthLoc = inps.start_down + inps.win_height // 2 + np.arange(length) * inps.skip_down
    rangeLoc = inps.start_across + inps.win_width // 2 + np.arange(width) * inps.skip_across

    nWritten = 0
    nDroppedSnr = 0
    nDroppedCov = 0
    nDroppedNan = 0
    with open(inps.output, 'w') as f:
        if inps.header:
            f.write('# range_loc range_offset azimuth_loc azimuth_offset SNR cov_xx cov_yy cov_xy\n')
        for i in range(length):
            for j in range(width):
                if not inps.keep_nan:
                    rowVals = (downOffset[i, j], acrossOffset[i, j], snr[i, j],
                               covXX[i, j], covYY[i, j], covXY[i, j])
                    if any(np.isnan(v) or np.isinf(v) for v in rowVals):
                        nDroppedNan += 1
                        continue
                if inps.min_snr is not None and snr[i, j] < inps.min_snr:
                    nDroppedSnr += 1
                    continue
                if inps.max_cov is not None and (covXX[i, j] > inps.max_cov or covYY[i, j] > inps.max_cov):
                    nDroppedCov += 1
                    continue

                fields = [
                    fortranIntField(int(rangeLoc[j]), 7),
                    fortranField(acrossOffset[i, j], 9, 3),
                    fortranIntField(int(azimuthLoc[i]), 7),
                    fortranField(downOffset[i, j], 11, 3),
                    fortranField(snr[i, j], 10, 5),
                    fortranField(covXX[i, j], 10, 6),
                    fortranField(covYY[i, j], 10, 6),
                    fortranField(covXY[i, j], 10, 6),
                ]
                f.write(' ' + ' '.join(fields) + '\n')
                nWritten += 1

    print(f'Wrote {nWritten} windows ({length}x{width} grid) to {inps.output}')
    if nDroppedNan:
        print(f'  dropped {nDroppedNan} windows containing NaN/Inf (use --keep-nan to disable this)')
    if nDroppedSnr:
        print(f'  dropped {nDroppedSnr} windows below --min-snr {inps.min_snr}')
    if nDroppedCov:
        print(f'  dropped {nDroppedCov} windows above --max-cov {inps.max_cov}')
    if inps.keep_nan:
        print('Note: legacy ampcor.F also silently excludes edge/no-data windows; NaN dropping '
              '(disabled here via --keep-nan) is the closest cuAmpcor equivalent to that check.')


def main(iargs=None):
    inps = cmdLineParse(iargs)
    convert(inps)


if __name__ == '__main__':
    main()
