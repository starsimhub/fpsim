#!/usr/bin/env python3
"""
Compare calibration parameters between the current code and a saved baseline.

Useful after recalibration to see how much parameters shifted and whether
the changes are within expected tolerances.

Usage:
    # Save current parameters as baseline
    python compare_calib_pars.py --save

    # Compare current vs saved baseline
    python compare_calib_pars.py

    # Compare a specific location
    python compare_calib_pars.py --location kenya

    # Compare with a custom tolerance
    python compare_calib_pars.py --tolerance 0.2
"""

import argparse
import json
import os
import numpy as np
import sciris as sc
import fpsim as fp
import fpsim.defaults as fpd


# %% Configuration

locations = [
    'senegal', 'kenya', 'cotedivoire', 'ethiopia', 'niger',
    'nigeria_kaduna', 'nigeria_kano', 'nigeria_lagos', 'pakistan_sindh',
]

baseline_dir = os.path.join(os.path.dirname(__file__), 'calib_baselines')

# Scalar parameters that are directly calibrated
scalar_keys = [
    'exposure_factor', 'fecundity_low', 'fecundity_high',
    'prob_use_intercept', 'prob_use_trend_par',
]

# Tolerances for flagging changes (relative, as fraction)
default_tolerance = 0.3  # 30% change flagged


# %% Core functions

def get_calib_pars(location):
    """Get the calibration parameters for a location"""
    pars = fpd.get_calib_pars(location)
    return pars if pars else {}


def snapshot_pars(location):
    """Extract a flat comparable dict from calib pars"""
    pars = get_calib_pars(location)
    snap = {}
    for key in scalar_keys:
        if key in pars:
            snap[key] = float(pars[key])

    # Exposure age curve (values only, ages are fixed)
    if 'exposure_age' in pars:
        ea = pars['exposure_age']
        if isinstance(ea, np.ndarray) and ea.ndim == 2:
            snap['exposure_age'] = ea[1].tolist()
        elif isinstance(ea, dict):
            snap['exposure_age'] = list(ea.get('rel_exp', ea.values()))

    # Exposure parity (values only)
    if 'exposure_parity' in pars:
        ep = pars['exposure_parity']
        if isinstance(ep, np.ndarray) and ep.ndim == 2:
            snap['exposure_parity'] = ep[1].tolist()

    # Spacing preference
    if 'spacing_pref' in pars:
        sp = pars['spacing_pref']
        if isinstance(sp, dict) and 'preference' in sp:
            pref = sp['preference']
            snap['spacing_pref'] = pref.tolist() if isinstance(pref, np.ndarray) else list(pref)

    # Other calibrated values
    for key in ['dur_postpartum', 'method_weights']:
        if key in pars:
            val = pars[key]
            if isinstance(val, np.ndarray):
                snap[key] = val.tolist()
            else:
                snap[key] = float(val)

    return snap


def compare_pars(old, new, tolerance=default_tolerance):
    """Compare two parameter snapshots and return changes"""
    changes = []
    all_keys = sorted(set(list(old.keys()) + list(new.keys())))

    for key in all_keys:
        ov = old.get(key)
        nv = new.get(key)

        if ov is None:
            changes.append(dict(key=key, status='ADDED', old=None, new=nv, pct=None))
            continue
        if nv is None:
            changes.append(dict(key=key, status='REMOVED', old=ov, new=None, pct=None))
            continue

        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            if abs(ov) > 1e-9:
                pct = (nv - ov) / abs(ov)
            else:
                pct = 0 if abs(nv) < 1e-9 else float('inf')
            flag = abs(pct) > tolerance
            changes.append(dict(key=key, status='CHANGED' if flag else 'ok', old=ov, new=nv, pct=pct))

        elif isinstance(ov, list) and isinstance(nv, list):
            max_pct = 0
            for o, n in zip(ov, nv):
                if abs(o) > 1e-9:
                    max_pct = max(max_pct, abs((n - o) / o))
            flag = max_pct > tolerance
            changes.append(dict(key=key, status='CHANGED' if flag else 'ok', old='[array]', new='[array]', pct=max_pct))

    return changes


def print_comparison(location, changes, verbose=True):
    """Pretty-print a comparison"""
    flagged = [c for c in changes if c['status'] not in ('ok',)]
    print(f"\n{'='*60}")
    print(f"  {location}: {len(flagged)} parameters flagged out of {len(changes)}")
    print(f"{'='*60}")

    for c in changes:
        if not verbose and c['status'] == 'ok':
            continue
        pct_str = f"{c['pct']:>+7.0%}" if c['pct'] is not None else '    n/a'
        marker = ' ***' if c['status'] not in ('ok',) else ''
        if c['status'] == 'ADDED':
            print(f"  ADDED   {c['key']}")
        elif c['status'] == 'REMOVED':
            print(f"  REMOVED {c['key']}")
        else:
            if isinstance(c['old'], (int, float)):
                print(f"  {c['key']:<30} {c['old']:>10.4f} → {c['new']:>10.4f}  {pct_str}{marker}")
            else:
                print(f"  {c['key']:<30} {c['old']:>10} → {c['new']:>10}  {pct_str}{marker}")


# %% CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--save', action='store_true', help='Save current parameters as baseline')
    parser.add_argument('--location', type=str, default=None, help='Check a single location (default: all)')
    parser.add_argument('--tolerance', type=float, default=default_tolerance, help=f'Flag changes above this relative tolerance (default: {default_tolerance})')
    parser.add_argument('--verbose', action='store_true', help='Show all parameters, not just flagged ones')
    args = parser.parse_args()

    locs = [args.location] if args.location else locations
    os.makedirs(baseline_dir, exist_ok=True)

    if args.save:
        for loc in locs:
            snap = snapshot_pars(loc)
            outfile = os.path.join(baseline_dir, f'{loc}.json')
            with open(outfile, 'w') as f:
                json.dump(snap, f, indent=2)
            print(f"Saved baseline for {loc} ({len(snap)} parameters)")
        return

    # Compare
    n_flagged_total = 0
    for loc in locs:
        baseline_file = os.path.join(baseline_dir, f'{loc}.json')
        if not os.path.exists(baseline_file):
            print(f"\n  {loc}: no baseline found — run with --save first")
            continue

        with open(baseline_file) as f:
            old = json.load(f)
        new = snapshot_pars(loc)
        changes = compare_pars(old, new, tolerance=args.tolerance)
        print_comparison(loc, changes, verbose=args.verbose)
        n_flagged_total += sum(1 for c in changes if c['status'] not in ('ok',))

    if n_flagged_total == 0:
        print(f"\nAll parameters within {args.tolerance:.0%} tolerance.")
    else:
        print(f"\n{n_flagged_total} parameters flagged across {len(locs)} locations.")


if __name__ == '__main__':
    main()
