"""
Run calibration for all FPsim locations sequentially using fp.Calibration.
Uses multiprocessing for parallel Optuna workers within each location.
Results are printed as code snippets to paste into each location's .py file.
"""

import numpy as np
import traceback
import fpsim as fp


def main():
    locations = [
        'senegal',
        'kenya',
        'cotedivoire',
        'ethiopia',
        'niger',
        'nigeria_kaduna',
        'nigeria_kano',
        'nigeria_lagos',
        'pakistan_sindh',
    ]

    total_trials = 200
    n_agents = 5000

    # Calibration parameter ranges [best, low, high]
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

            results[loc] = dict(
                best_pars=calib.best_pars,
                mismatch=calib.study.best_value,
            )
            print(f'\nOK {loc} complete (mismatch: {calib.study.best_value:.2f})')
            print(f'Best pars: {calib.best_pars}')
        except Exception as e:
            print(f'\nFAIL {loc} FAILED: {e}')
            traceback.print_exc()
            failed.append(loc)

    print(f'\n\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'Completed: {len(results)}/{len(locations)}')
    if failed:
        print(f'Failed: {failed}')
    for loc, res in results.items():
        print(f'\n{loc}: mismatch={res["mismatch"]:.2f}')
        print(f'  pars: {res["best_pars"]}')


if __name__ == '__main__':
    main()
