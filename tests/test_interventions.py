"""
Run tests on the interventions.
"""

import sys
from pathlib import Path

import sciris as sc
import starsim as ss
import numpy as np
import pytest

# Ensure we import fpsim from this repo (workspace may contain another fpsim checkout)
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
import fpsim as fp  # noqa: E402
assert str(fp.__file__).startswith(str(_repo_root)), f"Imported fpsim from {fp.__file__}, expected within {_repo_root}"

parallel   = 1 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
# sc.options(backend='agg') # Turn off interactive plots


def make_sim(**kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''
    sim = fp.Sim(test=True, stop=2010, **kwargs)
    return sim


def test_intervention_fn():
    """ Test defining an intervention as a function """
    sc.heading('Testing intervention can be defined as a function...')

    def test_interv(sim):
        if sim.ti == 100:
            print(f'Success on day {sim.ti}')
            sim.intervention_applied = True

    sim = make_sim(interventions=test_interv)
    sim.run()
    assert sim.intervention_applied
    print(f'✓ (functions intervention ok)')

    return sim


def test_change_par():
    ''' Testing that change_par() modifies sim results in expected ways '''
    sc.heading('Testing change_par()...')

    # Define exposure test
    verbose = True
    year = 2002
    ec = 0.01
    cp1 = fp.change_par(par='exposure_factor', years=year, vals=ec, verbose=verbose, name='cp1') # Reduce exposure factor
    cp2 = fp.change_par(par='exposure_factor', years=year+5, vals=['reset'], verbose=verbose, name='cp2')  # Reset exposure factor
    s0 = make_sim(label='Baseline')
    s1 = make_sim(interventions=sc.dcp(cp1), label='Low exposure')
    s2 = make_sim(interventions=[sc.dcp(cp1), cp2], label='Low exposure, reset')

    # Run
    m = ss.parallel(s0, s1, s2, parallel=parallel)
    s0, s1, s2 = m.sims[:] # Replace with run versions

    # Test exposure factor change
    base_births = s0.results.fp.births.sum()
    cp1_births   = s1.results.fp.births.sum()
    cp2_births  = s2.results.fp.births.sum()
    assert s1.pars.fp.exposure_factor == ec, f'change_pars() did not change exposure factor to {ec}'
    assert cp1_births < base_births, f'Reducing exposure factor should reduce births, but {cp1_births} is not less than the baseline of {base_births}'

    assert s2.pars.fp.exposure_factor == 1.0, f'Exposure factor should be reset back to 1.0, but it is {s2["exposure_factor"]}'
    assert cp2_births <= base_births, f'Reducing exposure factor temporarily should reduce births, but {cp2_births} is not less than the baseline of {base_births}'

    return m


def test_plot():
    sc.heading('Testing intervention plotting...')

    cp = fp.change_par(par='exposure_factor', years=2002, vals=2.0) # Reduce exposure factor
    um1 = fp.update_methods(year=2005, eff={'Injectables': 1.0}, name='um1')
    um2 = fp.update_methods(year=2008, p_use=0.5, name='um2')
    um3 = fp.update_methods(year=2010, method_mix=[0.9, 0.1, 0, 0, 0, 0, 0, 0, 0], name='um3')
    sim = make_sim(contraception_module=fp.RandomChoice(), interventions=[cp, um1, um2, um3]).run()

    return sim


def test_change_people_state():
    """ Testing that change_people_state() modifies sim results in expected ways """
    sc.heading('Testing change_people_state()...')

    pars = dict(n_agents=500, start=2000, stop=2010, rand_seed=1, verbose=-1, location='kenya')
    ms = fp.SimpleChoice(location='kenya')

    # Change ever user
    prior_use_lift = fp.change_people_state('fp.ever_used_contra', years=2009, new_val=True, eligibility=np.arange(500), prop=1, annual=False)
    prior_use_gone = fp.change_people_state('fp.ever_used_contra', years=2010, new_val=False, eligibility=np.arange(500), prop=1, annual=False)

    # Make and run sim
    s0 = fp.Sim(pars=pars, contraception_module=sc.dcp(ms), label="Baseline")
    s1 = fp.Sim(pars=pars, contraception_module=sc.dcp(ms), interventions=prior_use_lift, label="All prior_use set to True")
    s2 = fp.Sim(pars=pars, contraception_module=sc.dcp(ms), interventions=prior_use_gone, label="Prior use removed from 500 people")
    msim = ss.parallel(s0, s1, s2)
    s0, s1, s2 = msim.sims

    # Test people state change
    s0_used_contra = np.sum(s0.people.fp.ever_used_contra)
    s1_used_contra = np.sum(s1.people.fp.ever_used_contra)

    s2auids = s2.people.fp.ever_used_contra.auids
    s2subset = s2auids[s2auids < 500]
    s2_500_used_contra = np.sum(s2.people.fp.ever_used_contra[s2subset])

    print(f"Checking change_state CPR trends ... ")
    assert s1_used_contra > s0_used_contra, f'Increasing prior use should increase the number of people with who have used contraception, but {s1_used_contra} is not greater than the baseline of {s0_used_contra}'
    assert s2_500_used_contra == 0, f'Changing people state should set prior use to False for the first 500 agents, but {s2_500_used_contra} is not 0'
    print(f"✓ ({s1_used_contra} > {s0_used_contra})")

    return s0, s1, s2


def _pick_age_group(cm, preferred='18-20'):
    """Pick a stable age group key from method_choice_pars."""
    mcp0 = cm.pars.method_choice_pars[0]
    if preferred in mcp0:
        return preferred
    return next(k for k in mcp0.keys() if k != 'method_idx')


def test_add_method_registers_copy_on_init():
    """Core: when method=None, add_method copies from copy_from during init()."""
    sc.heading('Testing add_method() registers a copied method...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    intv = fp.add_method(year=2001, method=None, method_pars=None, copy_from='impl', verbose=False)

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.init()  # triggers init_pre()
    cm = sim.connectors.contraception

    assert 'impl_copy' in cm.methods
    new_method = cm.methods['impl_copy']
    src_method = cm.get_method('impl')

    # Copy means "same as source, except name may differ"
    assert new_method.name == 'impl_copy'
    assert new_method.efficacy == src_method.efficacy
    assert new_method.modern == src_method.modern
    assert new_method.rel_dur_use == src_method.rel_dur_use
    assert (new_method.dur_use is None) == (src_method.dur_use is None)

    # The FP module's method_mix array must be resized to match the new number of options
    assert sim.connectors.fp.method_mix.shape[0] == cm.n_options

    return sim


def test_add_method_overrides_attributes_via_method_pars():
    """Core: method_pars overrides attributes on the copied (or provided) method."""
    sc.heading('Testing add_method() method_pars overrides...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    method_pars = dict(
        name='new_implant',
        label='New Implant (test)',
        efficacy=0.999,
        modern=True,
        rel_dur_use=1.25,
    )
    intv = fp.add_method(year=2001, method=None, method_pars=method_pars, copy_from='impl', verbose=False)

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.init()
    cm = sim.connectors.contraception

    assert method_pars['name'] in cm.methods
    m = cm.methods[method_pars['name']]
    assert m.label == method_pars['label']
    assert m.efficacy == method_pars['efficacy']
    assert m.modern == method_pars['modern']
    assert m.rel_dur_use == method_pars['rel_dur_use']

    return sim


def test_add_method_partial_method_pars_copies_rest_from_source():
    """Core: partial method_pars should override only provided keys; rest is copied from source."""
    sc.heading('Testing add_method() with partial method_pars...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    new_name = 'partial_method'
    intv = fp.add_method(
        year=2001,
        method=None,
        method_pars=dict(name=new_name, efficacy=0.97),  # intentionally partial
        copy_from='impl',
        verbose=False,
    )

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.init()
    cm = sim.connectors.contraception

    src = cm.get_method('impl')
    assert new_name in cm.methods
    m = cm.methods[new_name]

    # Overridden
    assert m.name == new_name
    assert m.efficacy == 0.97

    # Copied from source
    assert m.label == src.label
    assert m.modern == src.modern
    assert m.rel_dur_use == src.rel_dur_use

    return sim


def test_add_method_activation_splits_switching_probabilities():
    """Core: split_shares splits probability to the new method (postpartum=0)."""
    sc.heading('Testing add_method() activation splits switching probabilities...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    split = 0.3
    new_name = 'impl_new'
    intv = fp.add_method(
        year=2001,
        method=None,
        method_pars=dict(name=new_name),
        copy_from='impl',
        split_shares=split,
        verbose=False,
    )

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.init()
    cm = sim.connectors.contraception
    age = _pick_age_group(cm)

    # Before activation: probability to the new method should be 0; source is unchanged
    p_source_before = cm.get_switching_prob('pill', 'impl', postpartum=0, age_grp=age)
    p_new_before = cm.get_switching_prob('pill', new_name, postpartum=0, age_grp=age)
    assert np.isclose(p_new_before, 0.0)

    sim.run()

    # After activation: the original pill->impl probability is split
    p_source_after = cm.get_switching_prob('pill', 'impl', postpartum=0, age_grp=age)
    p_new_after = cm.get_switching_prob('pill', new_name, postpartum=0, age_grp=age)
    assert np.isclose(p_new_after, p_source_before * split)
    assert np.isclose(p_source_after, p_source_before * (1 - split))

    # Sanity: row still sums to ~1 (renormalized)
    row = cm.get_switching_matrix(0, 'pill')[age]
    assert np.isclose(np.nansum(row), 1.0)

    return sim


def test_add_method_negative_errors():
    """Error handling: invalid inputs should raise clear exceptions."""
    sc.heading('Testing add_method() error handling...')

    # Construction-time validation
    with pytest.raises(ValueError, match='Year must be specified'):
        fp.add_method(year=None, copy_from='impl', verbose=False)

    with pytest.raises(ValueError, match='copy_from must specify'):
        fp.add_method(year=2001, copy_from=None, verbose=False)

    with pytest.raises(TypeError, match='method_pars must be a dict'):
        fp.add_method(year=2001, copy_from='impl', method_pars=['not-a-dict'], verbose=False)

    with pytest.raises(ValueError, match='split_shares must be between 0 and 1'):
        fp.add_method(year=2001, copy_from='impl', split_shares=-0.1, verbose=False)

    with pytest.raises(ValueError, match='split_shares must be between 0 and 1'):
        fp.add_method(year=2001, copy_from='impl', split_shares=1.1, verbose=False)

    # Init-time validation (depends on sim range and method existence)
    pars = dict(n_agents=50, start=2000, stop=2002, verbose=0, location='kenya')

    intv_bad_year = fp.add_method(year=1999, copy_from='impl', verbose=False)
    with pytest.raises(ValueError, match='must be between'):
        fp.Sim(pars=pars, interventions=[intv_bad_year], verbose=0).init()

    intv_bad_method = fp.add_method(year=2001, copy_from='does_not_exist', verbose=False)
    with pytest.raises(ValueError, match='not found'):
        fp.Sim(pars=pars, interventions=[intv_bad_method], verbose=0).init()

    return


@pytest.mark.parametrize('split, expected_source_frac, expected_new_frac', [
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
])
def test_add_method_split_shares_boundaries(split, expected_source_frac, expected_new_frac):
    """Boundary analysis: split_shares at 0 and 1 behaves as expected."""
    sc.heading(f'Testing add_method() split_shares boundary={split}...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    new_name = f'impl_split_{int(split*10)}'
    intv = fp.add_method(
        year=2001,
        method=None,
        method_pars=dict(name=new_name),
        copy_from='impl',
        split_shares=split,
        verbose=False,
    )

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.init()
    cm = sim.connectors.contraception
    age = _pick_age_group(cm)

    p_source_before = cm.get_switching_prob('pill', 'impl', postpartum=0, age_grp=age)
    sim.run()

    p_source_after = cm.get_switching_prob('pill', 'impl', postpartum=0, age_grp=age)
    p_new_after = cm.get_switching_prob('pill', new_name, postpartum=0, age_grp=age)
    assert np.isclose(p_source_after, p_source_before * expected_source_frac)
    assert np.isclose(p_new_after, p_source_before * expected_new_frac)

    return sim


@pytest.mark.parametrize('year_offset', [0, 2])
def test_add_method_year_boundaries(year_offset):
    """Boundary analysis: year at sim start and sim stop should activate without error."""
    sc.heading('Testing add_method() year boundary behavior...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    year = pars['start'] + year_offset  # 2000 or 2002
    new_name = f'year_boundary_{year}'
    intv = fp.add_method(
        year=year,
        method=None,
        method_pars=dict(name=new_name),
        copy_from='impl',
        split_shares=0.0,  # avoid depending on probability exactness here
        verbose=False,
    )

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.run()

    # Intervention should have activated by the end of the run
    assert sim.interventions[0].activated is True
    return sim


if __name__ == '__main__':
    s0 = test_intervention_fn()
    s1 = test_change_par()
    s3 = test_plot()
    s4, s5, s6 = test_change_people_state()
    s7 = test_add_method_registers_copy_on_init()
    s8 = test_add_method_overrides_attributes_via_method_pars()
    s8b = test_add_method_partial_method_pars_copies_rest_from_source()
    s9 = test_add_method_activation_splits_switching_probabilities()
    test_add_method_negative_errors()
    s10 = test_add_method_split_shares_boundaries(0.0, 1.0, 0.0)
    s11 = test_add_method_split_shares_boundaries(1.0, 0.0, 1.0)
    s12 = test_add_method_year_boundaries(0)
    s13 = test_add_method_year_boundaries(2)

    print('Done.')


