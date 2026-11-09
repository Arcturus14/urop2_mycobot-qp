"""Claude's published global table-grid -> base XY transform (no robot access).

Grid u,w are in 25mm hole steps from S. Z=0 is a declared table height.
This tool does not estimate TCP, measure Z, or determine motion reachability.
"""
import argparse
import json
import math
from pathlib import Path


def to_xyz(u, w, method):
    theta = math.radians(method['theta_degrees'])
    c, s = math.cos(theta), math.sin(theta)
    x0, y0 = method['origin_S_base_xy_mm']
    p = method['pitch_mm']
    return [x0 + p*(c*u-s*w), y0 + p*(s*u+c*w), 0.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('claude_table_method_20261104.json'))
    parser.add_argument('--u', type=float)
    parser.add_argument('--w', type=float)
    parser.add_argument('--bolt', choices=['S','A','B','C','D','E','F'])
    parser.add_argument('--check', action='store_true', help='Check published coordinate examples using rounded inputs')
    args = parser.parse_args()
    method = json.loads(args.config.read_text(encoding='utf-8'))
    if args.check:
        checks = {}
        for name, ref in method['source_reference_targets'].items():
            actual = to_xyz(*ref['uw'], method)
            error = math.dist(actual, ref['xyz_mm'])
            assert error < .2, (name, actual, ref)
            checks[name] = {'calculated_xyz_mm': actual, 'difference_from_rounded_reference_mm': error}
        print(json.dumps({'reference_checks': checks}, ensure_ascii=False))
        return
    if args.bolt:
        if args.u is not None or args.w is not None:
            parser.error('Use --bolt OR --u and --w')
        u, w = method['bolt_grid_uw'][args.bolt]
    else:
        if args.u is None or args.w is None:
            parser.error('Provide --bolt or both --u and --w')
        u, w = args.u, args.w
    if not all(math.isfinite(v) for v in [u,w]):
        parser.error('Grid coordinates must be finite')
    print(json.dumps({'method': method['method_id'], 'grid_uw': [u,w],
                      'xyz_mm': to_xyz(u,w,method), 'convention': method['coordinate_convention'],
                      'status': 'source-calibrated estimate; no motion commanded'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
