"""
Regression snapshot tool for comparing FPsim results across versions.

Usage:
    # Generate a snapshot for the current version
    python test_regression.py --save my_snapshot.json

    # Compare two snapshots
    python test_regression.py --compare old.json new.json

    # Run as a pytest (checks that a sim runs and results are plausible)
    pytest test_regression.py
"""

import argparse
import numpy as np
import sciris as sc
import fpsim as fp
import starsim as ss


# %% Configuration

snapshot_pars = dict(
    n_agents = 1000,
    seed     = 0,
    start    = 2000,
    stop     = 2010,
    location = 'kenya',
    verbose  = 0,
)

# Result keys to snapshot — maps to sim.results.fp.<key>
fp_timeseries_keys = [
    'births', 'pregnancies', 'stillbirths', 'miscarriages',
    'abortions', 'total_births', 'infant_deaths',
]

# Cumulative keys (single end-of-sim value)
fp_cumulative_keys = [
    'cum_stillbirths', 'cum_miscarriages', 'cum_abortions',
    'cum_total_births', 'cum_infant_deaths',
]

# People-level summary stats to capture at end of sim
people_stats = [
    'n_pregnancies', 'n_births', 'n_stillbirths', 'n_miscarriages', 'n_abortions', 'parity',
]


# %% Core functions

def make_sim():
    """ Create a deterministic sim for regression testing """
    sim = fp.Sim(**snapshot_pars)
    return sim


def generate_snapshot(sim=None):
    """
    Run a sim and extract a reproducible snapshot of key results.

    Returns a dict with:
        - metadata: version info, parameters
        - timeseries: per-timestep result arrays (as lists)
        - summary: scalar summary statistics
    """
    if sim is None:
        sim = make_sim()
    sim.run()

    fpmod = sim.people.fp
    snapshot = sc.objdict()

    # Metadata
    snapshot.metadata = sc.objdict(
        fpsim_version  = fp.__version__,
        starsim_version = ss.__version__,
        parameters = {k: v for k, v in snapshot_pars.items() if k != 'verbose'},
    )

    # Time series results
    snapshot.timeseries = sc.objdict()
    for key in fp_timeseries_keys:
        try:
            arr = sim.results.fp[key]
            snapshot.timeseries[key] = np.array(arr).tolist()
        except Exception:
            snapshot.timeseries[key] = None

    # Cumulative results
    for key in fp_cumulative_keys:
        try:
            arr = sim.results.fp[key]
            snapshot.timeseries[key] = float(np.array(arr)[-1])
        except Exception:
            snapshot.timeseries[key] = None

    # mCPR if available
    try:
        mcpr = np.array(sim.results.contraception.mcpr)
        snapshot.timeseries['mcpr'] = mcpr.tolist()
    except Exception:
        snapshot.timeseries['mcpr'] = None

    # People-level summary stats (means and sums over ever-pregnant women)
    snapshot.summary = sc.objdict()
    was_preg = (fpmod.n_pregnancies > 0).uids
    for key in people_stats:
        try:
            vals = getattr(fpmod, key)[was_preg]
            snapshot.summary[f'{key}_mean'] = float(np.nanmean(vals)) if len(vals) else 0.0
            snapshot.summary[f'{key}_sum'] = float(np.nansum(vals))
        except AttributeError:
            snapshot.summary[f'{key}_mean'] = None
            snapshot.summary[f'{key}_sum'] = None

    # Population
    snapshot.summary['n_alive'] = int(sim.people.alive.sum())
    snapshot.summary['n_female'] = int(sim.people.female.sum())
    snapshot.summary['n_pregnant'] = int(fpmod.pregnant.sum())

    return snapshot


def compare_snapshots(old, new, rtol=0.0, atol=0.0, verbose=True):
    """
    Compare two snapshots and report differences.

    Args:
        old (dict): baseline snapshot
        new (dict): new snapshot to compare
        rtol (float): relative tolerance for "close enough" (0 = exact match)
        atol (float): absolute tolerance
        verbose (bool): print detailed comparison

    Returns:
        dict: comparison results with diffs
    """
    diffs = sc.objdict()

    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing snapshots")
        print(f"  Old: fpsim {old['metadata']['fpsim_version']}, starsim {old['metadata']['starsim_version']}")
        print(f"  New: fpsim {new['metadata']['fpsim_version']}, starsim {new['metadata']['starsim_version']}")
        print(f"{'='*70}")

    # Compare summaries
    if verbose:
        print(f"\n{'Summary statistics':}")
        print(f"  {'Key':<30s} {'Old':>12s} {'New':>12s} {'Diff':>12s} {'Rel%':>8s}")
        print(f"  {'-'*74}")

    for key in sorted(set(list(old.get('summary', {}).keys()) + list(new.get('summary', {}).keys()))):
        old_val = old.get('summary', {}).get(key)
        new_val = new.get('summary', {}).get(key)
        if old_val is not None and new_val is not None:
            diff = new_val - old_val
            rel = 100 * diff / old_val if old_val != 0 else float('inf') if diff != 0 else 0
            match = np.isclose(old_val, new_val, rtol=rtol, atol=atol)
            marker = '' if match else ' ***'
            diffs[key] = sc.objdict(old=old_val, new=new_val, diff=diff, rel_pct=rel, match=match)
            if verbose:
                print(f"  {key:<30s} {old_val:>12.2f} {new_val:>12.2f} {diff:>+12.2f} {rel:>+7.1f}%{marker}")

    # Compare time series (summary stats of each)
    if verbose:
        print(f"\n{'Time series (total over sim):':}")
        print(f"  {'Key':<30s} {'Old':>12s} {'New':>12s} {'Diff':>12s} {'Rel%':>8s}")
        print(f"  {'-'*74}")

    for key in sorted(set(list(old.get('timeseries', {}).keys()) + list(new.get('timeseries', {}).keys()))):
        old_val = old.get('timeseries', {}).get(key)
        new_val = new.get('timeseries', {}).get(key)
        if old_val is None or new_val is None:
            if verbose:
                print(f"  {key:<30s} {'missing':>12s} {'missing':>12s}")
            continue

        # Scalar cumulative values
        if isinstance(old_val, (int, float)):
            old_sum, new_sum = old_val, new_val
        else:
            old_sum = sum(old_val)
            new_sum = sum(new_val)

        diff = new_sum - old_sum
        rel = 100 * diff / old_sum if old_sum != 0 else float('inf') if diff != 0 else 0
        match = np.isclose(old_sum, new_sum, rtol=rtol, atol=atol)
        marker = '' if match else ' ***'
        diffs[f'ts_{key}_total'] = sc.objdict(old=old_sum, new=new_sum, diff=diff, rel_pct=rel, match=match)
        if verbose:
            print(f"  {key:<30s} {old_sum:>12.2f} {new_sum:>12.2f} {diff:>+12.2f} {rel:>+7.1f}%{marker}")

    # Overall summary
    n_mismatches = sum(1 for v in diffs.values() if not v.match)
    if verbose:
        print(f"\n{'='*70}")
        if rtol == 0 and atol == 0:
            print(f"Exact match comparison: {len(diffs) - n_mismatches}/{len(diffs)} values match")
        else:
            print(f"Approximate comparison (rtol={rtol}, atol={atol}): {len(diffs) - n_mismatches}/{len(diffs)} values match")
        if n_mismatches:
            print(f"  *** = value changed")
        print(f"{'='*70}\n")

    return diffs


# %% Pytest

def test_regression_snapshot():
    """ Test that a snapshot can be generated and results are plausible """
    snap = generate_snapshot()

    # Basic plausibility checks
    s = snap.summary
    assert s.n_alive > 0, 'No agents alive'
    assert s.n_pregnancies_sum > 0, 'No pregnancies occurred'
    assert s.n_births_sum > 0, 'No births occurred'
    assert s.parity_mean > 0, 'Mean parity is zero'

    # Check that pregnancy outcomes balance
    total_outcomes = s.n_births_sum + s.n_stillbirths_sum + s.n_miscarriages_sum + s.n_abortions_sum
    assert total_outcomes >= s.n_pregnancies_sum * 0.9, \
        f'Outcomes ({total_outcomes}) much less than pregnancies ({s.n_pregnancies_sum})'

    # Check time series are non-empty
    assert snap.timeseries.births is not None, 'No births timeseries'
    assert sum(snap.timeseries.births) > 0, 'Zero total births in timeseries'

    return snap


# %% CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FPsim regression snapshot tool')
    parser.add_argument('--save', type=str, help='Generate and save a snapshot to this file')
    parser.add_argument('--compare', nargs=2, metavar=('OLD', 'NEW'), help='Compare two snapshot files')
    parser.add_argument('--rtol', type=float, default=0.0, help='Relative tolerance for comparison (default: exact)')
    args = parser.parse_args()

    if args.save:
        print(f'Generating snapshot...')
        snap = generate_snapshot()
        sc.savejson(args.save, snap, indent=2)
        print(f'Saved to {args.save}')

    elif args.compare:
        old = sc.loadjson(args.compare[0])
        new = sc.loadjson(args.compare[1])
        diffs = compare_snapshots(old, new, rtol=args.rtol)

    else:
        # Default: generate and print
        snap = test_regression_snapshot()
        print('Snapshot generated successfully. Use --save <file> to save it.')
