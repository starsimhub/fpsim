"""
Run tests on individual parameters.
"""

import os
import numpy as np
import sciris as sc
import fpsim as fp
import pytest
import starsim as ss
import types
import fpsim.defaults as fpd

do_plot = False
sc.options(backend='agg') # Turn off interactive plots


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def test_null(do_plot=do_plot):
    sc.heading('Testing no births, no deaths...')

    pars = fp.all_pars(location='senegal')  # For default pars

    # Set things to zero
    for key in ['exposure_factor']:
        pars[key] = 0

    for key in ['f', 'm']:
        pars['age_mortality'][key] *= 0

    for key in ['age_mortality', 'maternal_mortality', 'infant_mortality']:
        pars[key]['probs'] *= 0

    sim = fp.Sim(pars=pars)
    sim.run()

    # Tests
    n = sim.results.fp.births.sum()
    assert n == 0, f'Expecting 0 births, not {n}'
    n = sim.results.new_deaths.sum()
    assert n == 0, f'Expecting 0 deaths, not {n}'
    ok(f'Births and deaths are 0, as expected')

    return sim


def test_scale():
    sc.heading('Test scale factor')

    # Test settings
    scale = 2

    # Make and run sims
    pars = dict(test=True)
    s1 = fp.Sim(pars=pars)
    s2 = fp.Sim(pars=pars, pop_scale=scale)
    msim = ss.parallel([s1, s2], shrink=False)
    s1, s2 = msim.sims

    # Tests
    orig = s1.results.fp.total_births.sum()
    expected = scale*orig
    actual = s2.results.fp.total_births.sum()
    assert expected == actual, 'Total births should scale exactly with scale factor'
    assert np.array_equal(s1.results.contraception.mcpr, s2.results.contraception.mcpr), 'Scale factor should not change MCPR'
    ok(f'{actual} births = {scale}*{orig} as expected')

    return [s1, s2]


def test_method_changes():
    sc.heading('Test changing methods')

    # # Test adding method
    choice = fp.RandomChoice()
    n = len(choice.methods)
    new_method = fp.Method(
        name='new',
        efficacy=1,
        modern=True,
        dur_use=dict(dist='lognormal', par1=10, par2=3),
        label='New method')
    choice.add_method(new_method)
    s1 = fp.Sim(test=True, contraception_module=choice)
    s1.run()
    assert len(s1.connectors.contraception.methods) == n+1, 'Method was not added'
    ok(f'Methods had expected length after addition ({n+1})')

    # Test remove method
    # Note that this will only work with the RandomChoice module since others have complex switching matrices
    # that depend on specific methods being present
    method_list = [m for m in fp.make_methods().values() if m.label != 'Injectables']
    choice = fp.RandomChoice(methods=method_list)
    s2 = fp.Sim(test=True, contraception_module=choice)
    s2.run()
    assert len(s2.connectors.contraception.methods) == len(method_list), 'Method was not removed'
    ok(f'Methods have expected length after removal ({n})')

    # Test method efficacy
    methods = fp.make_methods()
    for method in methods.values(): method.efficacy = 1  # Make all methods totally effective
    choice = fp.RandomChoice(pars=dict(p_use=1), methods=methods)
    s3 = fp.Sim(test=True, contraception_module=choice)
    s3.run()
    assert s3.results.fp.births.sum() == 0, f'Expecting births to be 0, not {n}'
    ok(f'No births with completely effective contraception, as expected')


# %% Parameter coverage tests

# Parameters that are consumed during __init__ and transformed into other objects
consumed_at_init = {
    'fecundity_low',    # Combined into pars.fecundity (ss.uniform)
    'fecundity_high',   # Combined into pars.fecundity (ss.uniform)
    'debut_age',        # Consumed by _fated_debut distribution in __init__
    'twins_prob',       # Consumed by embryos_per_pregnancy in __init__
}

# Parameters that may be None for some locations (optional)
optional_pars = {'region', 'regional'}

# Known issues: parameters defined but not yet wired into model logic
# TODO: fix these — they are defined in FPPars but never accessed during a sim run
known_unused = {
    'maternal_mortality_factor',  # Defined for calibration but not used in model
}

# Parameters inherited from PregnancyPars used by the parent Pregnancy class
inherited_pars = {
    'fertility_rate', 'rel_fertility', 'rate_units', 'dur_pregnancy',
    'dur_breastfeed', 'p_breastfeed', 'embryos_per_pregnancy', 'rr_ptb',
    'rr_ptb_age', 'p_maternal_death', 'p_survive_maternal_death', 'p_loss',
    'loss_threshold', 'sex_ratio', 'slot_scale', 'min_slots',
    'preterm_threshold', 'very_preterm_threshold', 'burnin', 'trimesters',
    'p_infertile', 'min_age', 'max_age', 'fecundity',
}

# Expected mappings from data files to parameter keys
expected_data_mappings = {
    'bf_stats.csv':              ['dur_breastfeed'],
    'debut_age.csv':             ['debut_age'],
    'sexually_active.csv':       ['sexual_activity'],
    'sexually_active_pp.csv':    ['sexual_activity_pp'],
    'birth_spacing_pref.csv':    ['spacing_pref'],
    'scalar_probs.csv':          ['abortion_prob', 'twins_prob'],
    'maternal_mortality.csv':    ['maternal_mortality'],
    'infant_mortality.csv':      ['infant_mortality'],
    'stillbirths.csv':           ['stillbirth_rate'],
    'lam.csv':                   ['lactational_amenorrhea'],
}


def _get_loaded_par_keys(location):
    """Get the parameter keys that load_fp_data produces"""
    dataloader = fp.get_dataloader(location)
    fp_data = dataloader.load_fp_data(return_data=True)
    return set(fp_data.keys())


def _get_fppars_keys(location=None):
    """Get all non-private parameter keys defined in FPPars"""
    pars = fp.FPPars(location=location)
    return {k for k in pars.keys() if not k.startswith('_')}


def _get_accessed_pars(location):
    """Run a sim and track which pars keys are accessed by monkey-patching"""
    sim = fp.Sim(n_agents=500, seed=0, location=location, verbose=0, test=True)
    sim.init()

    accessed = set()
    pars = sim.people.fp.pars
    original_getitem = type(pars).__getitem__

    def tracking_getitem(self, key, *args, **kwargs):
        if isinstance(key, str):
            accessed.add(key)
        return original_getitem(self, key, *args, **kwargs)

    type(pars).__getitem__ = tracking_getitem
    try:
        sim.run()
    finally:
        type(pars).__getitem__ = original_getitem

    return accessed


def test_data_files_loaded(location='kenya'):
    """Check that expected data files are loaded into parameters"""
    sc.heading('Testing data file loading...')
    loaded_keys = _get_loaded_par_keys(location)

    missing = []
    for csv_file, par_keys in expected_data_mappings.items():
        for key in par_keys:
            if key not in loaded_keys:
                missing.append(f'{csv_file} -> {key}')

    assert not missing, f'Data files not loaded into parameters:\n  ' + '\n  '.join(missing)
    ok(f'All expected data files loaded for {location}')


def test_loaded_pars_have_targets(location='kenya'):
    """Check that every loaded data parameter has a matching FPPars attribute"""
    sc.heading('Testing loaded pars have targets...')
    loaded_keys = _get_loaded_par_keys(location)
    fppars_keys = _get_fppars_keys()
    all_known = fppars_keys | inherited_pars | consumed_at_init

    orphaned = loaded_keys - all_known
    assert not orphaned, (
        f'Loaded data parameters have no matching FPPars attribute:\n  '
        + '\n  '.join(sorted(orphaned))
    )
    ok(f'All loaded parameters have targets in FPPars')


def test_defined_pars_are_used(location='kenya'):
    """Check that every defined FPPars parameter is accessed during a sim run"""
    sc.heading('Testing defined pars are used...')
    accessed = _get_accessed_pars(location)
    fppars_keys = _get_fppars_keys()  # Default keys (location data added at runtime)
    excused = consumed_at_init | inherited_pars

    unused = fppars_keys - accessed - excused - optional_pars - known_unused
    assert not unused, (
        f'FPPars parameters defined but never accessed during sim:\n  '
        + '\n  '.join(sorted(unused))
        + '\n\nIf used by parent Pregnancy class, add to inherited_pars.'
        + '\nIf consumed at __init__, add to consumed_at_init.'
    )
    ok(f'All defined parameters are used during sim run')


def test_no_none_pars(location='kenya'):
    """Check that location data loading fills all None-initialized parameters"""
    sc.heading('Testing no None parameters...')
    # Use the actual loading path: init a sim, then check its fpmod pars
    sim = fp.Sim(n_agents=100, seed=0, location=location, verbose=0, test=True)
    sim.init()
    pars = sim.people.fp.pars
    none_pars = [k for k, v in pars.items() if v is None and k not in optional_pars and not k.startswith('_')]

    assert not none_pars, (
        f'FPPars parameters still None after loading {location} data:\n  '
        + '\n  '.join(sorted(none_pars))
    )
    ok(f'No None parameters after loading {location}')


def test_multiple_locations():
    """Run parameter checks across multiple locations"""
    sc.heading('Testing multiple locations...')
    for location in ['kenya', 'senegal']:
        loaded = _get_loaded_par_keys(location)
        assert len(loaded) > 0, f'No parameters loaded for {location}'

        sim = fp.Sim(n_agents=200, seed=0, location=location, verbose=0, test=True)
        sim.init()
        pars = sim.people.fp.pars
        none_pars = [k for k, v in pars.items() if v is None and k not in optional_pars and not k.startswith('_')]
        assert not none_pars, f'{location}: parameters still None: {none_pars}'

        sim.run()
        ok(f'{location} loads and runs OK')


def test_calib_pars_applied(location='kenya'):
    """
    Check that location calibration parameters are actually applied to the sim.

    Catches the case where make_calib_pars() returns values but they don't
    make it into the sim's effective parameters (e.g., due to merge order bugs
    or parameters being stripped during refactoring).
    """
    sc.heading('Testing calibration parameters are applied...')
    import fpsim.defaults as fpd2

    calib_pars = fpd2.get_calib_pars(location)
    if calib_pars is None:
        ok(f'No calibration parameters for {location} (OK)')
        return

    sim = fp.Sim(n_agents=100, seed=0, location=location, verbose=0, test=True)
    sim.init()
    effective = sim.people.fp.pars

    mismatches = []
    for key, calib_val in calib_pars.items():
        eff_val = effective.get(key, None)
        if eff_val is None:
            mismatches.append(f'{key}: calib value exists but not in effective pars')
            continue

        # For numpy arrays, compare element-wise
        if isinstance(calib_val, np.ndarray):
            if isinstance(eff_val, np.ndarray):
                if not np.allclose(calib_val, eff_val, rtol=0.01):
                    mismatches.append(f'{key}: arrays differ')
            continue

        # For dicts (like spacing_pref), check sub-keys
        if isinstance(calib_val, dict):
            if not isinstance(eff_val, dict):
                continue
            for sub_key, sub_val in calib_val.items():
                eff_sub = eff_val.get(sub_key, None)
                if eff_sub is None:
                    continue
                if isinstance(sub_val, np.ndarray) and isinstance(eff_sub, np.ndarray):
                    if not np.allclose(sub_val, eff_sub, rtol=0.01):
                        mismatches.append(f'{key}.{sub_key}: arrays differ')
            continue

        # For scalars
        try:
            if isinstance(calib_val, (int, float)) and isinstance(eff_val, (int, float)):
                if abs(calib_val - eff_val) > 0.01 * max(abs(calib_val), 1e-9):
                    mismatches.append(f'{key}: calib={calib_val}, effective={eff_val}')
        except (TypeError, ValueError):
            pass

    assert not mismatches, (
        f'Calibration parameters for {location} not applied correctly:\n  '
        + '\n  '.join(mismatches)
    )
    ok(f'Calibration parameters applied correctly for {location}')


if __name__ == '__main__':

    sc.options(backend=None)
    with sc.timer():
        null    = test_null(do_plot=do_plot)
        scale   = test_scale()
        meths   = test_method_changes()

        # Parameter coverage tests
        test_data_files_loaded()
        test_loaded_pars_have_targets()
        test_defined_pars_are_used()
        test_no_none_pars()
        test_calib_pars_applied()
        test_multiple_locations()
