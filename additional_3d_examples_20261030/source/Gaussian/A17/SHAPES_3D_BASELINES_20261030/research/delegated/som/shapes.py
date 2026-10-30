"""Pure NumPy synthetic targets. No physics or inverse-solver imports.

Truth is for input synthesis/offline evaluation only. Run this module to save
pure geometric support counts; it never imports a reconstruction runner.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

TARGETS = {
    'digit3': {'label': 'Volumetric digit 3', 'formula': 'Union of ellipses centered y=+/-0.15: (x/.29)^2+((y-c)/.27)^2<=1, remove inner ellipses (.10,.09), cut x<-.12; abs(z)<=.17.', 'nominal_min_thickness': .18},
    'letter_e': {'label': 'Volumetric letter E', 'formula': 'Vertical beam x in [-.34,-.14], abs(y)<=.35; union three horizontal bars x in [-.34,.30], abs(y-c)<=.10 for c=-.25,0,.25; abs(z)<=.17.', 'nominal_min_thickness': .20},
    'torus': {'label': 'Torus (z-axis hole)', 'formula': '(sqrt(x*x+y*y)-.24)^2+z*z<=.12^2.', 'nominal_min_thickness': .24},
    'double_torus': {'label': 'Double torus (two z-axis holes)', 'formula': 'Union over c=+/-0.20 of (sqrt((x-c)^2+y*y)-.15)^2+z*z<=.09^2. Maximum abs(x)=.44.', 'nominal_min_thickness': .18},
    'cross3d': {'label': 'Three-axis thick cross', 'formula': 'Union of x/y/z rods with longitudinal abs(coord)<=.35 and both transverse abs(coord)<=.10.', 'nominal_min_thickness': .20},
    'offset_shell': {'label': 'Shell with offset core', 'formula': 'Shell: .18<=sqrt(x*x+y*y+z*z)<=.35, chi=.7+.06j; core: (x-.07)^2+y*y+z*z<=.09^2 inside cavity r<.18, chi=1.4+.12j. Core and shell labels disjoint.', 'nominal_min_thickness': .17},
}


def truth_at(name, points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('points must be a finite (N,3) array')
    if name not in TARGETS:
        raise ValueError(f'Unknown target: {name}')
    x, y, z = p.T
    if name == 'digit3':
        outer = np.zeros(len(p), bool)
        inner = np.zeros(len(p), bool)
        for c in (-.15, .15):
            outer |= (x/.29)**2 + ((y-c)/.27)**2 <= 1
            inner |= (x/.10)**2 + ((y-c)/.09)**2 < 1
        mask = outer & ~inner & (x >= -.12) & (np.abs(z) <= .17)
    elif name == 'letter_e':
        mask = (x >= -.34) & (x <= -.14) & (np.abs(y) <= .35)
        for c in (-.25, 0., .25):
            mask |= (x >= -.34) & (x <= .30) & (np.abs(y-c) <= .10)
        mask &= np.abs(z) <= .17
    elif name in ('torus', 'double_torus'):
        mask = np.zeros(len(p), bool)
        for c in ((0.,) if name == 'torus' else (-.20, .20)):
            major, tube = (.24, .12) if name == 'torus' else (.15, .09)
            mask |= (np.sqrt((x-c)**2+y*y)-major)**2+z*z <= tube*tube
    elif name == 'cross3d':
        mask = np.zeros(len(p), bool)
        for axis in range(3):
            transverse = [k for k in range(3) if k != axis]
            mask |= (np.abs(p[:, axis]) <= .35) & np.all(np.abs(p[:, transverse]) <= .10, axis=1)
    else:
        radius2 = x*x+y*y+z*z
        shell = (radius2 <= .35**2) & (radius2 >= .18**2)
        core = ((x-.07)**2+y*y+z*z <= .09**2) & (radius2 < .18**2)
        if np.any(shell & core):
            raise AssertionError('illegal core/shell overlap')
        return np.where(shell, .7+.06j, np.where(core, 1.4+.12j, 0j))
    return np.where(mask, 1.+.08j, 0j)


def support_check(n=12):
    """Count center-sampled voxels, labels, bounds and nominal feature width.

    Nominal thickness is a geometric scale, not a proof that curved boundary
    voxels have two-neighbor thickness everywhere. Small boundary chords are
    inherent in center sampling. Shell thickness .17 is >=2h at n12; core diameter is .18.
    """
    if n < 2:
        raise ValueError('n must be >=2')
    a = -.5+(np.arange(n)+.5)/n
    p = np.stack(np.meshgrid(a, a, a, indexing='ij'), axis=-1).reshape(-1, 3)
    rows = []
    for name, meta in TARGETS.items():
        chi = truth_at(name, p)
        support = chi.real >= .5
        q = p[support]
        width = meta['nominal_min_thickness']*n
        rows.append(dict(target=name, n=n, total_voxels=len(p), support_voxels=int(support.sum()),
                         body_voxels=int(np.count_nonzero(np.isclose(chi, 1+.08j))),
                         shell_voxels=int(np.count_nonzero(np.isclose(chi, .7+.06j))),
                         core_voxels=int(np.count_nonzero(np.isclose(chi, 1.4+.12j))),
                         background_voxels=int(np.count_nonzero(chi == 0)),
                         occupied_x_layers=int(np.unique(q[:,0]).size), occupied_y_layers=int(np.unique(q[:,1]).size),
                         occupied_z_layers=int(np.unique(q[:,2]).size),
                         min_real=float(chi.real.min()), max_real=float(chi.real.max()),
                         max_support_abs_coordinate=float(np.abs(q).max()) if len(q) else '',
                         nominal_min_thickness=meta['nominal_min_thickness'], thickness_in_grid_intervals=width,
                         nominal_thickness_status='AT_LEAST_2_INTERVALS' if width >= 2-1e-12 else 'THICKNESS_LT_2',
                         finite=bool(np.isfinite(chi).all()), nonempty=bool(support.any())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('GEOMETRY_SUPPORT_N12.csv'))
    args = parser.parse_args()
    rows = support_check(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == '__main__':
    main()
