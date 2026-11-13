#!/usr/bin/env python
"""Compare three cuDenseOffsets runs of the same pair on the same ice windows.

  static_wide : --rr -21 --aa -4,           --sw 32 --sh 8   (manual_offset/)
  narrow      : --rr -21 --aa -5,           --sw 8  --sh 4   (half_sw*)
  model_wide  : gross file from the netcdf, --sw 32 --sh 8   (003193-003368.offmap_1*)

total offset = offmap + gross  (gross = the starting shift of that run).

Run it from the track directory. Edit N, M, SNR_MIN and RUNS below as needed.
"""
import numpy as np

N, M = 1299, 1907
SNR_MIN = 20.0
RUNS = {                                   # half = (azimuth, range) half search
    'static_wide': dict(prefix='manual_offset/003193-003368.offmap_1', half=(8, 32)),
    'narrow':      dict(prefix='half_sw',                              half=(4, 8)),
    'model_wide':  dict(prefix='003193-003368.offmap_1',               half=(8, 32)),
}
AZ, RG = 0, 1
NAMES = ('azimuth', 'range')


def rd(f, *shape):
    return np.fromfile(f, dtype=np.float32).reshape(*shape)


def load_ice():
    try:
        from osgeo import gdal
        a = gdal.Open('icemask/grossOffsets_grossRange.tif').ReadAsArray()
    except Exception as e:
        print('GDAL mask failed (%s); trying the ENVI .off file' % e)
        a = rd('grossOffsets_grossRange.off', N, M)
    return np.isfinite(a)


def pct(x, q=(50, 90, 99)):
    return np.round(np.percentile(x, q), 2)


def main():
    ice = load_ice()
    print('ice windows: %d (%.1f%% of grid)' % (ice.sum(), 100 * ice.mean()))

    D = {}
    for name, r in RUNS.items():
        off = rd(r['prefix'] + '.bip', N, M, 2)
        gro = rd(r['prefix'] + '_gross.bip', N, M, 2)
        snr = rd(r['prefix'] + '_snr.bip', N, M)
        D[name] = dict(off=off, tot=off + gro, snr=snr, half=r['half'])
        print('%-12s gross median on ice (az, rg): %s' %
              (name, np.round(np.median(gro[ice], axis=0), 2)))

    # 1. windows that pass the SNR filter
    print('\n== 1. ice windows with SNR > %g, per run ==' % SNR_MIN)
    for n, d in D.items():
        d['good'] = ice & (d['snr'] > SNR_MIN)
        print('%-12s %8d' % (n, d['good'].sum()))
    common = ice.copy()
    for d in D.values():
        common &= d['good']
    print('common to all three: %d' % common.sum())

    # 2. spread of each run around its own median total offset (common windows)
    print('\n== 2. |total - median(total)| on common windows: p50 / p90 / p99 ==')
    for n, d in D.items():
        for k in (AZ, RG):
            t = d['tot'][..., k][common]
            print('%-12s %-8s median total %+7.2f   %s' %
                  (n, NAMES[k], np.median(t), pct(np.abs(t - np.median(t)))))

    # 3. pile-up on the edge of the search window (the window is too small)
    print("\n== 3. share of each run's own good windows within 0.5 px of the search edge ==")
    for n, d in D.items():
        for k in (AZ, RG):
            o = np.abs(d['off'][..., k][d['good']])
            print('%-12s %-8s half-search %2d : %.3f%%' %
                  (n, NAMES[k], d['half'][k], 100 * (o >= d['half'][k] - 0.5).mean()))

    # 4. agreement between runs (same windows)
    print('\n== 4. difference of total offsets between runs on common windows ==')
    names = list(D)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for k in (AZ, RG):
                df = D[a]['tot'][..., k][common] - D[b]['tot'][..., k][common]
                ad = np.abs(df)
                print('%-12s - %-12s %-8s bias %+6.2f  |diff| p50/p90/p99 %s  >0.3px %.2f%%  >1px %.2f%%' %
                      (a, b, NAMES[k], np.median(df), pct(ad),
                       100 * (ad > 0.3).mean(), 100 * (ad > 1).mean()))

    # 5. what the narrow window loses or changes relative to the wide static run
    print('\n== 5. narrow vs static_wide, over the windows good in static_wide ==')
    g = D['static_wide']['good']
    lost = g & ~D['narrow']['good']
    both = g & D['narrow']['good']
    print('windows good in static_wide : %d' % g.sum())
    print('lost in narrow (SNR <= %g)   : %d (%.2f%%)' % (SNR_MIN, lost.sum(), 100 * lost.sum() / g.sum()))
    for k in (AZ, RG):
        df = np.abs(D['narrow']['tot'][..., k][both] - D['static_wide']['tot'][..., k][both])
        print('%-8s kept windows, |diff| >1 px: %.2f%%, >3 px: %.2f%%' %
              (NAMES[k], 100 * (df > 1).mean(), 100 * (df > 3).mean()))


if __name__ == '__main__':
    main()
