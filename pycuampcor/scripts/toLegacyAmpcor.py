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

The window geometry is read from <prefix>_geometry.json, which cuDenseOffsets writes next to
its outputs, so the start pixel, step and window size are the values the run actually used.
For outputs from older runs without that file, pass the geometry on the command line.

Like the legacy file, the table has one line per *kept* window only: windows dropped by the
NaN/SNR/covariance checks are simply absent, so the line count is usually smaller than
length x width. Do not reshape the table back into a grid; use the location columns.
"""

import argparse
import json
import os
import re
import sys
import numpy as np


ISCE_DATATYPE_TO_NUMPY = {
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
}

GEOMETRY_FORMAT = 'cuampcor-run-geometry'

# command-line option (argparse dest) -> key in <prefix>_geometry.json
GEOMETRY_KEYS = {
    'start_down': 'start_pixel_down',
    'start_across': 'start_pixel_across',
    'skip_down': 'skip_down',
    'skip_across': 'skip_across',
    'win_height': 'window_height',
    'win_width': 'window_width',
    'length': 'number_window_down',
    'width': 'number_window_across',
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
    """Open a BIP raster read-only (memory-mapped), as a (length, width, bands) array.

    Its .xml sidecar, when present, is authoritative: a width/length/bands value given by
    the caller that disagrees with it is an error. The size check alone cannot catch a
    swapped width and length (same product), which would scramble every row silently.
    """
    meta = parse_isce_xml(path + '.xml')
    if meta is not None:
        for name, given, fromXml in (('width', width, meta['width']),
                                     ('length', length, meta['length']),
                                     ('bands', bands, meta['bands'])):
            if given is not None and fromXml is not None and given != fromXml:
                raise ValueError(
                    f'{path}: {name} {given} does not match {name} {fromXml} in {path}.xml. '
                    'The .xml is written by cuDenseOffsets for this exact file: check that the '
                    'file is the right one, or drop the conflicting --width/--length.')
        width = width or meta['width']
        length = length or meta['length']
        dtype = dtype or meta['dtype']

    if width is None or length is None:
        raise ValueError(
            f'Cannot determine width/length for {path} (no .xml sidecar or geometry file found); '
            'pass --width/--length explicitly.')
    # cuAmpcor's offset/snr/cov rasters are always single precision (float/float2/float3
    # in the CUDA source -- there is no double-precision build path), so default here
    # and treat anything else as a user error rather than silently reinterpreting bytes.
    dtype = dtype or np.float32
    if dtype not in (np.float32,):
        raise ValueError(
            f'{path}: dtype {dtype} was requested, but cuAmpcor output rasters are always '
            'float32. Do not pass --double unless you know your build actually differs.')

    expected = width * length * bands
    nbytes = os.path.getsize(path)
    if nbytes != expected * np.dtype(dtype).itemsize:
        raise ValueError(
            f'{path}: expected {expected} elements ({length}x{width}x{bands} {dtype.__name__}), '
            f'got {nbytes // np.dtype(dtype).itemsize}. Check --width/--length.')
    if expected == 0:
        return np.zeros((length, width, bands), dtype=dtype), width, length
    # memory-mapped and read-only: only the windows actually written are paged in, and no
    # full in-memory copy of the raster is ever made
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, width, bands))
    return data, width, length


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


# Legacy ampcor.F format 151: 1x,i7,1x,f9.3,1x,i7,1x,f11.3,1x,f10.5,1x,f10.6,1x,f10.6,1x,f10.6
LINE_FORMAT = ' %7d %9.3f %7d %11.3f %10.5f %10.6f %10.6f %10.6f\n'
LINE_LENGTH = 83  # characters per line, newline included, when every value fits its field
WRITE_BLOCK = 10000  # windows formatted per write call (bounds memory; larger is not faster)


def formatLineFortran(vals):
    """Slow path for a line where at least one value overflows its field: format every
    column with the Fortran-style helpers so the overflowing ones become asterisks."""
    rangeLoc, acrossOff, azimuthLoc, downOff, snr, covXX, covYY, covXY = vals
    fields = [
        fortranIntField(int(rangeLoc), 7),
        fortranField(acrossOff, 9, 3),
        fortranIntField(int(azimuthLoc), 7),
        fortranField(downOff, 11, 3),
        fortranField(snr, 10, 5),
        fortranField(covXX, 10, 6),
        fortranField(covYY, 10, 6),
        fortranField(covXY, 10, 6),
    ]
    return ' ' + ' '.join(fields) + '\n'


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
                              'required for parity unless you ran cuDenseOffsets with --mg 1).\n'
                              'Refused when the geometry file says the run already merged it (--mg 1), '
                              'since it would then be added twice.')
    parser.add_argument('--geometry', type=str, default=None,
                         help='Run geometry file written by cuDenseOffsets (default: <prefix>_geometry.json, '
                              'used automatically when it exists). It supplies the window geometry, grid size '
                              'and mergeGrossOffset of the run, so none of them need to be typed in.')

    parser.add_argument('--width', type=int, default=None, help='Number of windows across (columns). '
                         'Auto-detected from the geometry file or a .xml sidecar if present.')
    parser.add_argument('--length', type=int, default=None, help='Number of windows down (rows). '
                         'Auto-detected from the geometry file or a .xml sidecar if present.')

    geom = parser.add_argument_group(
        'Window geometry',
        'Used to reconstruct azimuth_loc/range_loc pixel coordinates (window CENTER pixel, matching\n'
        'legacy i_centerxi/i_centeryi). Read from the geometry file when there is one; only needed\n'
        'for outputs of older runs. Careful: the start values are the top-left pixel (0-based) of the\n'
        'first window AS THE RUN USED IT, which is not always what was typed into cuDenseOffsets:\n'
        '  - without --startpixeldw/--startpixelac, cuDenseOffsets starts at margin + half search range;\n'
        '  - with --use-center 1, it subtracts window_size//2 from the value given.\n'
        'cuDenseOffsets prints the value it used as "the first pixel in reference image is: (down, across)".')
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


def loadGeometry(path):
    """Read a <prefix>_geometry.json written by cuDenseOffsets."""
    with open(path) as f:
        geo = json.load(f)
    if geo.get('format') != GEOMETRY_FORMAT:
        raise ValueError(f'{path} is not a cuAmpcor run geometry file (format: {geo.get("format")!r}).')
    missing = [k for k in list(GEOMETRY_KEYS.values()) + ['merge_gross_offset'] if k not in geo]
    if missing:
        raise ValueError(f'{path} is missing {", ".join(missing)}.')
    return geo


def cmdLineParse(iargs=None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)

    if inps.prefix is None and (inps.offset is None or inps.snr is None or inps.cov is None):
        parser.error('Either --prefix or all of --offset/--snr/--cov must be given.')

    inps.offset = inps.offset or f'{inps.prefix}.bip'
    inps.snr = inps.snr or f'{inps.prefix}_snr.bip'
    inps.cov = inps.cov or f'{inps.prefix}_cov.bip'
    base = inps.prefix or os.path.splitext(inps.offset)[0]

    # run geometry file: explicit --geometry must exist; the default one is used if present
    geometryPath = inps.geometry or f'{base}_geometry.json'
    inps.merge_gross_offset = None
    if inps.geometry is not None and not os.path.isfile(inps.geometry):
        parser.error(f'--geometry {inps.geometry}: file not found.')
    if os.path.isfile(geometryPath):
        try:
            geo = loadGeometry(geometryPath)
        except (ValueError, json.JSONDecodeError) as err:
            parser.error(str(err))
        for dest, key in GEOMETRY_KEYS.items():
            given, fromFile = getattr(inps, dest), int(geo[key])
            if given is not None and given != fromFile:
                parser.error(f'--{dest.replace("_", "-")} {given} does not match {key} = {fromFile} '
                             f'in {geometryPath}, which records what the run actually used. '
                             'Drop the option, or point --geometry at the right run.')
            setattr(inps, dest, fromFile)
        inps.merge_gross_offset = bool(geo['merge_gross_offset'])
        print(f'Using run geometry from {geometryPath}')

    if inps.gross:
        if inps.merge_gross_offset:
            parser.error(f'--gross given, but {geometryPath} says this run already merged the gross '
                         'offset into the offset (cuDenseOffsets --mg 1). Adding it again would double '
                         'it. Drop --gross.')
        if inps.merge_gross_offset is None:
            print('WARNING: --gross adds the gross offset to the offset. If this run used '
                  'cuDenseOffsets --mg 1, the gross offset is already included and will be counted '
                  'twice; no geometry file was found to check this.', file=sys.stderr)

    haveGeom = [inps.start_down, inps.start_across, inps.skip_down, inps.skip_across,
                inps.win_height, inps.win_width]
    if any(v is not None for v in haveGeom) and not all(v is not None for v in haveGeom):
        parser.error('--start-down/--start-across/--skip-down/--skip-across/--win-height/--win-width '
                      'must all be given together.')
    if not all(v is not None for v in haveGeom):
        parser.error(f'No geometry file ({geometryPath}) was found, and the window geometry '
                      '(--start-down/--start-across/--skip-down/--skip-across/--win-height/--win-width) '
                      'was not given: without it, azimuth_loc/range_loc cannot be reconstructed as real '
                      'pixel coordinates, and GAMMA needs real pixel coordinates, not window indices.')

    if inps.output is None:
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
    # These are float32 views into the memory-mapped files, not copies: a float32 value
    # prints exactly the same whether or not it is first converted to float64.
    downOffset = offset[:, :, 0]
    acrossOffset = offset[:, :, 1]

    if inps.gross:
        gross, _, _ = readBip(inps.gross, bands=2, width=width, length=length)
        # summed in float64, as before this change, so existing outputs are reproduced exactly
        downOffset = downOffset.astype(np.float64) + gross[:, :, 0]
        acrossOffset = acrossOffset.astype(np.float64) + gross[:, :, 1]

    snr = snr[:, :, 0]
    covXX = cov[:, :, 0]
    covYY = cov[:, :, 1]
    covXY = cov[:, :, 2]

    # Window CENTER pixel, matching legacy's i_centerxi/i_centeryi = start + (size-1)/2
    azimuthLoc = inps.start_down + (inps.win_height - 1) // 2 + np.arange(length) * inps.skip_down
    rangeLoc   = inps.start_across + (inps.win_width - 1) // 2 + np.arange(width) * inps.skip_across
    # Decide which windows are written, for the whole grid at once. The order of the
    # checks (NaN/Inf first, then SNR, then covariance) matches the per-window loop this
    # replaced, so the dropped-window counts reported below are unchanged.
    remaining = np.ones((length, width), dtype=bool)
    nDroppedNan = nDroppedSnr = nDroppedCov = 0
    if not inps.keep_nan:
        finite = (np.isfinite(downOffset) & np.isfinite(acrossOffset) & np.isfinite(snr)
                  & np.isfinite(covXX) & np.isfinite(covYY) & np.isfinite(covXY))
        nDroppedNan = int(np.count_nonzero(~finite))
        remaining &= finite
        del finite
    # thresholds are compared in float64 (a temporary per check): comparing a float32 array
    # with a Python float would round the threshold to float32 and could move a window that
    # sits right at the threshold to the other side
    if inps.min_snr is not None:
        drop = remaining & (snr.astype(np.float64) < inps.min_snr)
        nDroppedSnr = int(np.count_nonzero(drop))
        remaining &= ~drop
    if inps.max_cov is not None:
        drop = remaining & ((covXX.astype(np.float64) > inps.max_cov)
                            | (covYY.astype(np.float64) > inps.max_cov))
        nDroppedCov = int(np.count_nonzero(drop))
        remaining &= ~drop

    # Row-major (azimuth line, then range sample) order, same as the legacy output.
    rows, cols = np.nonzero(remaining)
    del remaining
    nWritten = int(rows.size)

    if nWritten:
        # Write to a temporary file and move it into place only once it is complete, so a
        # crash or interruption never leaves a truncated table that looks like a real one.
        tmpPath = inps.output + '.tmp'
        try:
            with open(tmpPath, 'w') as f:
                if inps.header:
                    f.write('# range_loc range_offset azimuth_loc azimuth_offset SNR cov_xx cov_yy cov_xy\n')
                # Write in blocks so the text for a very large grid never sits in memory at once.
                for b in range(0, nWritten, WRITE_BLOCK):
                    r = rows[b:b + WRITE_BLOCK]
                    c = cols[b:b + WRITE_BLOCK]
                    block = zip(rangeLoc[c].tolist(), acrossOffset[r, c].tolist(),
                                azimuthLoc[r].tolist(), downOffset[r, c].tolist(), snr[r, c].tolist(),
                                covXX[r, c].tolist(), covYY[r, c].tolist(), covXY[r, c].tolist())
                    lines = []
                    for vals in block:
                        line = LINE_FORMAT % vals
                        # A value too wide for its column makes the line longer than LINE_LENGTH.
                        # Only those rare lines go through the slower Fortran-style formatter,
                        # which writes asterisks instead of shifting the later columns.
                        if len(line) != LINE_LENGTH:
                            line = formatLineFortran(vals)
                        lines.append(line)
                    f.write(''.join(lines))
            os.replace(tmpPath, inps.output)
        except BaseException:
            if os.path.exists(tmpPath):
                os.remove(tmpPath)
            raise
        print(f'Wrote {nWritten} windows ({length}x{width} grid) to {inps.output}')
    else:
        print(f'No valid windows ({length}x{width} grid); {inps.output} was not written')

    if nDroppedNan:
        print(f'  dropped {nDroppedNan} windows containing NaN/Inf (use --keep-nan to disable this)')
    if nDroppedSnr:
        print(f'  dropped {nDroppedSnr} windows below --min-snr {inps.min_snr}')
    if nDroppedCov:
        print(f'  dropped {nDroppedCov} windows above --max-cov {inps.max_cov}')
    if inps.keep_nan:
        print('Note: legacy ampcor.F also silently excludes edge/no-data windows; NaN dropping '
              '(disabled here via --keep-nan) is the closest cuAmpcor equivalent to that check.')
    return nWritten


def main(iargs=None):
    inps = cmdLineParse(iargs)
    try:
        nWritten = convert(inps)
    except ValueError as err:   # bad or mismatched input files: report without a traceback
        sys.exit(f'ERROR: {err}')
    if nWritten == 0:
        # an empty table almost always means the cuAmpcor run itself failed (e.g. wrong
        # input byte order -> all-NaN SNR); fail loudly so batch jobs don't carry on silently.
        # No output file is created, so a check for its existence also catches this.
        note = (f' (an older {inps.output} from a previous run is still there, untouched)'
                if os.path.exists(inps.output) else '')
        sys.exit(f'ERROR: no valid windows, so {inps.output} was not written{note}; check the '
                 'cuAmpcor outputs (all-NaN SNR or cov=99 everywhere usually means the input SLCs '
                 'were misread).')


if __name__ == '__main__':
    main()
