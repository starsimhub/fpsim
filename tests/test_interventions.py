"""
Run tests on the interventions.
"""

import sciris as sc
import starsim as ss
import fpsim as fp
import numpy as np

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


def test_add_method():
    """ Test the add_method intervention """
    sc.heading('Testing add_method()...')
    
    pars = dict(n_agents=500, start=2000, stop=2015, verbose=0, location='kenya')
    default_dur = ss.lognorm_ex(mean=2, std=1)
    
    def make_method(name, **kwargs):
        """ Helper to create methods with sensible defaults """
        defaults = dict(label=name.replace('_', ' ').title(), efficacy=0.99, modern=True, dur_use=default_dur)
        defaults.update(kwargs)
        return fp.Method(name=name, **defaults)
    
    def run_intv(intv):
        """ Helper to run a sim with an intervention """
        return fp.Sim(pars=pars, interventions=[intv], verbose=0).run()
    
    def expect_error(intv, msg):
        """ Helper to verify an intervention raises ValueError """
        try:
            run_intv(intv)
            raise AssertionError(msg)
        except ValueError:
            pass
    
    # Test 1: Basic functionality - method added and accessible
    method = make_method('test_method')
    sim = run_intv(fp.add_method(year=2010, method=method, copy_from='impl', verbose=False))
    cm = sim.connectors.contraception
    assert method.name in cm.methods and cm.n_methods > 9
    print(f'  ✓ Basic add_method works')
    
    # Test 2: Late introduction (year before simulation end) works
    run_intv(fp.add_method(year=pars['stop']-1, method=make_method('late'), copy_from='impl', verbose=False))
    print(f'  ✓ Late introduction works')
    
    # Test 3-4: Error cases - invalid year and invalid copy_from
    expect_error(
        fp.add_method(year=pars['start']-5, method=make_method('early'), copy_from='impl', verbose=False),
        'Should have raised ValueError for year before simulation start')
    print(f'  ✓ Invalid year correctly rejected')
    
    expect_error(
        fp.add_method(year=2010, method=make_method('bad'), copy_from='nonexistent', verbose=False),
        'Should have raised ValueError for invalid copy_from')
    print(f'  ✓ Invalid copy_from correctly rejected')
    
    # Test 5-6: Auto-copy behavior - missing properties copied, explicit ones preserved
    # Minimal method: all properties should be copied from source
    minimal = fp.Method(name='minimal', label='Minimal')  # No dur_use, efficacy, or modern
    sim_copy = run_intv(fp.add_method(year=2010, method=minimal, copy_from='impl', verbose=False))
    cm = sim_copy.connectors.contraception
    added, source = cm.methods['minimal'], cm.methods['impl']
    assert added.dur_use is not None, 'dur_use should be auto-copied'
    assert added.efficacy == source.efficacy, f'efficacy should match source: {added.efficacy} != {source.efficacy}'
    assert added.modern == source.modern, f'modern should match source: {added.modern} != {source.modern}'
    print(f'  ✓ Auto-copy of dur_use, efficacy, modern works')
    
    # Explicit method: only dur_use copied, explicit values preserved
    explicit = fp.Method(name='explicit', label='Explicit', efficacy=0.5, modern=False)
    sim_explicit = run_intv(fp.add_method(year=2010, method=explicit, copy_from='impl', verbose=False))
    added2 = sim_explicit.connectors.contraception.methods['explicit']
    assert added2.dur_use is not None and added2.efficacy == 0.5 and added2.modern == False
    print(f'  ✓ Explicit values preserved, only missing values copied')
    
    return sim


if __name__ == '__main__':
    s0 = test_intervention_fn()
    s1 = test_change_par()
    s3 = test_plot()
    s4, s5, s6 = test_change_people_state()
    s7 = test_add_method()

    print('Done.')


