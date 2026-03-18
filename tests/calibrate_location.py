"""
Calibrate FPsim parameters for a given location.

Uses Optuna to tune:
  - exposure_factor: scales all pregnancy probabilities (affects ASFR)
  - prob_use_intercept: shifts the logistic model for contraceptive uptake (affects MCPR)
  - fecundity_low / fecundity_high: personal fecundity distribution bounds
  - exposure_age curve: age-specific exposure multipliers (affects ASFR shape)

Usage:
    python calibrate_location.py kenya
    python calibrate_location.py senegal --n-trials 200
"""

import numpy as np
import sciris as sc
import optuna
import fpsim as fp
from fpsim import locations as fplocs

# Weights: emphasize MCPR and ASFR
weights = dict(
    mcpr=3,
    asfr=3,
    total_fertility_rate=3,
    pop_size=0.5,
    pop_growth_rate=0.5,
    crude_birth_rate=0.5,
    crude_death_rate=0.5,
    maternal_mortality_ratio=0.5,
    infant_mortality_rate=0.5,
    ageparity=1,
    method_counts=0,
    spacing_bins=1,
    spacing_stats=1,
    age_first_stats=4,
)

# Exposure age knots: ages at which we set the exposure multiplier
EXPOSURE_KNOT_AGES = [0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50]


def _neutralize_calib_pars(location):
    """
    Temporarily replace the location's make_calib_pars() with one that returns
    only exposure_parity (which we don't calibrate). This prevents the location
    module's calibrated values from overriding trial parameters during sim init
    (sim.py merges calib_pars on top of user pars).
    """
    loc_module = getattr(fplocs, location)
    original = loc_module.make_calib_pars

    def _empty_calib_pars():
        pars = {}
        # Keep exposure_parity since we don't calibrate it
        pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                            [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
        return pars

    loc_module.make_calib_pars = _empty_calib_pars
    return original


def _restore_calib_pars(location, original):
    """Restore the original make_calib_pars function."""
    getattr(fplocs, location).make_calib_pars = original


def run_experiment(location, trial_pars, n_agents=5_000):
    """Run a single experiment with given parameters and return the experiment."""
    pars = sc.mergedicts(dict(location=location, n_agents=n_agents, start=1960, verbose=0), trial_pars)
    exp = fp.Experiment(pars=pars)
    exp.run(weights=weights)
    return exp


def print_comparison(label, exp):
    """Print detailed comparison for an experiment."""
    print(f'\n--- {label} ---')
    print(f'  Total mismatch: {exp.fit.mismatch:.2f}')
    print(f'  Per-key mismatches:')
    for key, val in sorted(exp.fit.mismatches.items(), key=lambda x: -x[1]):
        print(f'    {key:30s}: {val:.2f}')

    for key in ['mcpr', 'asfr', 'total_fertility_rate']:
        if key in exp.fit.pair:
            m = np.array(exp.fit.pair[key].sim)
            d = np.array(exp.fit.pair[key].data)
            mask = (m != 0) | (d != 0)
            m_nz, d_nz = m[mask], d[mask]
            if len(m_nz):
                rmse = float(np.sqrt(np.mean((m_nz - d_nz)**2)))
                print(f'  {key.upper()} RMSE: {rmse:.2f}  '
                      f'(model: {m_nz.mean():.1f}, data: {d_nz.mean():.1f})')


def make_objective(location, n_agents):
    """Create an Optuna objective function for the given location."""
    def objective(trial):
        # Core fertility parameters
        exposure_factor = trial.suggest_float('exposure_factor', 0.8, 5.0)
        fecundity_low = trial.suggest_float('fecundity_low', 0.5, 0.9)
        fecundity_high = trial.suggest_float('fecundity_high', 1.0, 2.5)

        # Contraceptive uptake
        prob_use_intercept = trial.suggest_float('prob_use_intercept', -3.0, -0.5)

        # Exposure age curve — finer knots at younger ages for better age-at-first-birth fit
        exp_age_0    = 1.0   # Fixed (pre-reproductive baseline)
        exp_age_5    = trial.suggest_float('exp_age_5',    0.01, 0.5)   # Pre-reproductive
        exp_age_10   = trial.suggest_float('exp_age_10',   0.01, 0.5)   # Pre-reproductive
        exp_age_12_5 = trial.suggest_float('exp_age_12.5', 0.05, 1.0)   # Early puberty
        exp_age_15   = trial.suggest_float('exp_age_15',   0.3, 2.0)
        exp_age_18   = trial.suggest_float('exp_age_18',   0.3, 2.5)    # Late adolescence
        exp_age_20   = trial.suggest_float('exp_age_20',   0.5, 3.0)
        exp_age_25   = trial.suggest_float('exp_age_25',   0.5, 3.0)
        exp_age_30   = trial.suggest_float('exp_age_30',   0.5, 3.0)
        exp_age_35   = trial.suggest_float('exp_age_35',   0.2, 2.0)
        exp_age_40   = trial.suggest_float('exp_age_40',   0.1, 1.5)
        exp_age_45   = trial.suggest_float('exp_age_45',   0.05, 1.0)
        exp_age_50   = trial.suggest_float('exp_age_50',   0.01, 0.5)

        exposure_age = np.array([
            EXPOSURE_KNOT_AGES,
            [exp_age_0, exp_age_5, exp_age_10, exp_age_12_5, exp_age_15,
             exp_age_18, exp_age_20, exp_age_25, exp_age_30,
             exp_age_35, exp_age_40, exp_age_45, exp_age_50]
        ])

        trial_pars = dict(
            exposure_factor=exposure_factor,
            prob_use_intercept=prob_use_intercept,
            fecundity_low=fecundity_low,
            fecundity_high=fecundity_high,
            exposure_age=exposure_age,
        )

        # Smoothness penalty: penalize large jumps between adjacent exposure_age knots
        exp_vals = exposure_age[1]
        diffs = np.diff(exp_vals)
        smoothness_penalty = 0.5 * np.mean(diffs**2)

        try:
            exp = run_experiment(location, trial_pars, n_agents=n_agents)
            return exp.fit.mismatch + smoothness_penalty
        except Exception as e:
            print(f'  Trial failed: {e}')
            return float('inf')

    return objective


def run_calibration(location, n_trials=100, n_agents=5_000):
    """Run the full calibration for a location."""

    # Show baseline with current calib_pars
    print(f'Running baseline for {location} (current calib_pars)...')
    baseline_exp = run_experiment(location, {}, n_agents=n_agents)
    print_comparison(f'BASELINE ({location})', baseline_exp)

    # Neutralize location's make_calib_pars so trial values take effect
    original_calib = _neutralize_calib_pars(location)
    print(f'(Neutralized {location}.make_calib_pars for calibration)')

    # Run Optuna optimization
    print(f'\nStarting Optuna calibration with {n_trials} trials...')
    study = optuna.create_study(direction='minimize', study_name=f'{location}_calib')
    study.optimize(make_objective(location, n_agents), n_trials=n_trials, n_jobs=-1, show_progress_bar=False)

    # Keep calib_pars neutralized for the final comparison so trial values apply

    # Extract best parameters
    best = study.best_params
    print(f'\nBest trial: #{study.best_trial.number}, mismatch: {study.best_value:.2f}')
    print(f'Best scalar parameters:')
    print(f'  exposure_factor:    {best["exposure_factor"]:.4f}')
    print(f'  prob_use_intercept: {best["prob_use_intercept"]:.4f}')
    print(f'  fecundity_low:      {best["fecundity_low"]:.4f}')
    print(f'  fecundity_high:     {best["fecundity_high"]:.4f}')


    # Reconstruct exposure_age
    exp_age_vals = [1.0]  # age 0
    for age in EXPOSURE_KNOT_AGES[1:]:
        exp_age_vals.append(best[f'exp_age_{age}'])
    exposure_age = np.array([EXPOSURE_KNOT_AGES, exp_age_vals])
    print(f'  exposure_age:')
    for age, val in zip(EXPOSURE_KNOT_AGES, exp_age_vals):
        print(f'    age {age:5.1f}: {val:.4f}')

    # Run final comparison
    best_pars = dict(
        exposure_factor=best['exposure_factor'],
        prob_use_intercept=best['prob_use_intercept'],
        fecundity_low=best['fecundity_low'],
        fecundity_high=best['fecundity_high'],
        exposure_age=exposure_age,
    )

    print('\nRunning final comparison...')
    calibrated_exp = run_experiment(location, best_pars, n_agents=n_agents)
    print_comparison(f'CALIBRATED ({location})', calibrated_exp)

    # Restore original calib_pars
    _restore_calib_pars(location, original_calib)

    # Print the code to apply these parameters
    print('\n' + '='*60)
    print(f'  Code to apply calibrated parameters for {location}')
    print('='*60)
    print(f"""
# In {location}.py make_calib_pars():
def make_calib_pars():
    pars = {{}}
    pars['exposure_factor'] = {best['exposure_factor']:.4f}
    pars['prob_use_intercept'] = {best['prob_use_intercept']:.4f}
    pars['fecundity_low'] = {best['fecundity_low']:.4f}
    pars['fecundity_high'] = {best['fecundity_high']:.4f}
    pars['exposure_age'] = np.array([{EXPOSURE_KNOT_AGES},
                                      {[round(v, 4) for v in exp_age_vals]}])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    return pars
""")

    return study, best_pars


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate FPsim parameters for a location')
    parser.add_argument('location', type=str, help='Location to calibrate (e.g., kenya, senegal)')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of trials (default: 100)')
    parser.add_argument('--n-agents', type=int, default=5_000, help='Number of agents (default: 5000)')
    args = parser.parse_args()

    study, best_pars = run_calibration(args.location, n_trials=args.n_trials, n_agents=args.n_agents)
