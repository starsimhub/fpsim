"""
Run calibration for all FPsim locations sequentially using fp.Calibration.
Uses multiprocessing for parallel Optuna workers within each location.
Results are saved to calib_results/ and can be used to update location files.

Usage:
    python calibrate_all.py                  # Run all locations
    python calibrate_all.py --location kenya # Run one location
    python calibrate_all.py --update         # Update location .py files from saved results
"""

import os
import numpy as np
import traceback
import sciris as sc
import fpsim as fp

results_dir = os.path.join(os.path.dirname(__file__), 'calib_results')


def calibrate(locations=None, total_trials=200, n_agents=5000):
    """Run calibration for specified locations"""

    if locations is None:
        locations = [
            'senegal', 'kenya', 'cotedivoire', 'ethiopia', 'niger',
            'nigeria_kaduna', 'nigeria_kano', 'nigeria_lagos', 'pakistan_sindh',
        ]

    calib_pars = dict(
        exposure_factor    = [1.0, 0.5, 3.0],
        prob_use_intercept = [-1.0, -3.0, 0.0],
        prob_use_trend_par = [0.0, -0.15, 0.05],
        fecundity_low      = [0.7, 0.5, 0.9],
        fecundity_high     = [1.5, 1.0, 2.5],
    )

    weights = dict(
        mcpr=3, asfr=3, total_fertility_rate=3, age_first_stats=4,
        ageparity=1, spacing_bins=1, spacing_stats=1,
        pop_size=0.5, pop_growth_rate=0.5, crude_birth_rate=0.5,
        crude_death_rate=0.5, maternal_mortality_ratio=0.5,
        infant_mortality_rate=0.5, method_counts=0,
    )

    os.makedirs(results_dir, exist_ok=True)
    results = {}
    failed = []

    for loc in locations:
        print(f'\n{"="*60}')
        print(f'  CALIBRATING: {loc}')
        print(f'{"="*60}\n')
        try:
            db_name = f'fpsim_calib_{loc}.db'
            calib = fp.Calibration(
                pars=dict(location=loc, n_agents=n_agents, start=1960, verbose=0),
                calib_pars=calib_pars,
                weights=weights,
                fit_exposure_age=True,
                smoothness_weight=0.1,
                total_trials=total_trials,
                n_workers=10,
                name=f'calib_{loc}',
                db_name=db_name,
                storage=f'sqlite:///{db_name}',
                verbose=True,
            )
            calib.calibrate()

            result = dict(
                best_pars=calib.best_pars,
                mismatch=calib.study.best_value,
            )
            results[loc] = result

            # Save per-location result immediately
            outfile = os.path.join(results_dir, f'{loc}.obj')
            sc.saveobj(outfile, result)

            print(f'\nOK {loc} complete (mismatch: {calib.study.best_value:.2f})')
            print(f'Saved to {outfile}')
        except Exception as e:
            print(f'\nFAIL {loc} FAILED: {e}')
            traceback.print_exc()
            failed.append(loc)

    # Summary
    print(f'\n\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'Completed: {len(results)}/{len(locations)}')
    if failed:
        print(f'Failed: {failed}')
    for loc, res in results.items():
        print(f'  {loc}: mismatch={res["mismatch"]:.2f}')

    return results


def update_location_files(locations=None):
    """Update each location's .py file with calibrated parameters from saved results"""

    if locations is None:
        locations = [
            'senegal', 'kenya', 'cotedivoire', 'ethiopia', 'niger',
            'nigeria_kaduna', 'nigeria_kano', 'nigeria_lagos', 'pakistan_sindh',
        ]

    for loc in locations:
        result_file = os.path.join(results_dir, f'{loc}.obj')
        if not os.path.exists(result_file):
            print(f'{loc}: no saved results found at {result_file}')
            continue

        result = sc.loadobj(result_file)
        pars = result['best_pars']
        mismatch = result['mismatch']

        # Build the make_calib_pars function body
        lines = []
        lines.append(f'"""\nSet the parameters for a location-specific FPsim model.\n"""')
        lines.append('import numpy as np')
        lines.append('import fpsim.locations.data_utils as fpld')
        lines.append('')
        lines.append('')
        lines.append('def make_calib_pars():')
        lines.append(f'    """ Make a dictionary of location-specific parameters (mismatch: {mismatch:.2f}) """')
        lines.append('    pars = {}')

        # Scalar parameters
        for key in ['exposure_factor', 'prob_use_intercept', 'prob_use_trend_par',
                     'fecundity_low', 'fecundity_high']:
            if key in pars:
                lines.append(f'    pars[\'{key}\'] = {pars[key]:.4f}')

        # Exposure age
        if 'exposure_age' in pars:
            ea = pars['exposure_age']
            ages_str = ', '.join(f'{a:g}' for a in ea[0])
            vals_str = ', '.join(f'{v:.4f}' for v in ea[1])
            lines.append(f'    pars[\'exposure_age\'] = np.array([[{ages_str}],')
            lines.append(f'                                      [{vals_str}]])')

        # Exposure parity
        if 'exposure_parity' in pars:
            ep = pars['exposure_parity']
            par_str = ', '.join(f'{int(a)}' for a in ep[0])
            val_str = ', '.join(f'{v:.4f}' for v in ep[1])
            lines.append(f'    pars[\'exposure_parity\'] = np.array([[{par_str}],')
            lines.append(f'                                         [{val_str}]])')

        # Spacing preference
        if 'spacing_pref' in pars:
            sp = pars['spacing_pref']
            pref = sp.get('preference', np.ones(19))
            pref_str = ', '.join(f'{v:.4f}' for v in pref)
            lines.append(f'    pars[\'spacing_pref\'] = {{')
            lines.append(f'        \'preference\': np.array([{pref_str}])')
            lines.append(f'    }}')

        lines.append('    return pars')
        lines.append('')
        lines.append('')
        lines.append(f'def dataloader(location=\'{loc}\'):')
        lines.append(f'    return fpld.DataLoader(location=location)')
        lines.append('')

        # Write the file
        loc_dir = os.path.join(os.path.dirname(__file__), loc)
        loc_file = os.path.join(loc_dir, f'{loc}.py')
        with open(loc_file, 'w') as f:
            f.write('\n'.join(lines))

        print(f'{loc}: updated {loc_file} (mismatch: {mismatch:.2f})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--location', type=str, default=None, help='Calibrate a single location')
    parser.add_argument('--update', action='store_true', help='Update location .py files from saved results')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials (default: 200)')
    parser.add_argument('--agents', type=int, default=5000, help='Number of agents (default: 5000)')
    args = parser.parse_args()

    if args.update:
        locs = [args.location] if args.location else None
        update_location_files(locs)
    else:
        locs = [args.location] if args.location else None
        calibrate(locs, total_trials=args.trials, n_agents=args.agents)
