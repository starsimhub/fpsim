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
    age_key = '18-20'  # Choose one age group to check

    # Case 1: Neither 'method' nor 'method_pars' are passed --> copies from source method
    print('  Testing Case 1: Neither method nor method_pars provided (copies from source)...')
    intv1 = fp.add_method(year=2010, copy_from='impl', verbose=False)
    
    # Create a sim to trigger init_pre and verify the method is copied correctly
    sim1 = fp.Sim(pars=pars, interventions=[intv1], verbose=0)
    sim1.init()
    cm1 = sim1.connectors.contraception
    
    # Verify the new method exists - it should be 'impl_copy'
    assert 'impl_copy' in cm1.methods, 'New method should be named impl_copy'
    new_method = cm1.methods['impl_copy']
    source_method = cm1.get_method('impl')
    
    # Verify the new method is the same as the source method, only the name differs
    assert new_method.name == 'impl_copy', f'Method name should be impl_copy, got {new_method.name}'
    assert new_method.efficacy == source_method.efficacy, 'Method efficacy should match source method'
    assert new_method.modern == source_method.modern, 'Method modern should match source method'
    assert new_method.rel_dur_use == source_method.rel_dur_use, 'Method rel_dur_use should match source method'
    
    # Note: dur_use might be a distribution object, so we check it exists rather than exact equality
    assert (new_method.dur_use is not None) == (source_method.dur_use is not None), 'Method dur_use presence should match source method'
    print(f'   ✓ Case 1 passed: method copied from source (same properties, name={new_method.name})')

    # Case 2: method=fp.Method is valid and method_pars=None --> use only values in method
    print('  Testing Case 2: method provided, method_pars=None...')
    
    # Get source method to copy dur_use for the test
    temp_sim = fp.Sim(pars=pars, verbose=0)
    temp_sim.init()
    source_method = temp_sim.connectors.contraception.get_method('impl')
    new_method = fp.Method(
        name='case2_method',
        label='Case 2 Method',
        efficacy=0.995,
        modern=True,
        rel_dur_use=1.2,
        dur_use=source_method.dur_use,  # Copy dur_use from source for test
    )
    intv2 = fp.add_method(year=2010, method=new_method, copy_from='impl', split_shares=0.3, verbose=False)
    sim2 = fp.Sim(pars=pars, interventions=[intv2], verbose=0)
    sim2.run()
    cm2 = sim2.connectors.contraception
    
    assert 'case2_method' in cm2.methods, 'New method should be in contraception methods'
    added_method2 = cm2.methods['case2_method']
    assert added_method2.efficacy == 0.995, f'Method efficacy should be 0.995, got {added_method2.efficacy}'
    assert added_method2.modern == True, f'Method modern should be True, got {added_method2.modern}'
    assert added_method2.rel_dur_use == 1.2, f'Method rel_dur_use should be 1.2, got {added_method2.rel_dur_use}'
    print(f'   ✓ Case 2 passed: method object used as-is')

    print('  Testing Case 3: method provided with method_pars to override...')
    temp_sim3 = fp.Sim(pars=pars, verbose=0)
    temp_sim3.init()
    source_method3 = temp_sim3.connectors.contraception.get_method('impl')
    base_method = fp.Method(
        name='case3_method',
        label='Case 3 Method',
        efficacy=0.90,  # Will be overridden
        modern=False,   # Will be overridden
        rel_dur_use=1.0,  # Will be overridden
        dur_use=source_method3.dur_use,  # Copy dur_use from source for test
    )
    method_pars_overrides = dict(
        efficacy=0.998,  # Override
        modern=True,     # Override
        rel_dur_use=1.5,  # Override
    )
    intv3 = fp.add_method(year=2010, method=base_method, method_pars=method_pars_overrides, copy_from='impl', split_shares=0.3, verbose=False)
    sim3 = fp.Sim(pars=pars, interventions=[intv3], verbose=0)
    sim3.run()
    cm3 = sim3.connectors.contraception
    
    assert 'case3_method' in cm3.methods, 'New method should be in contraception methods'
    added_method3 = cm3.methods['case3_method']
    assert added_method3.efficacy == 0.998, f'Method efficacy should be overridden to 0.998, got {added_method3.efficacy}'
    assert added_method3.modern == True, f'Method modern should be overridden to True, got {added_method3.modern}'
    assert added_method3.rel_dur_use == 1.5, f'Method rel_dur_use should be overridden to 1.5, got {added_method3.rel_dur_use}'
    assert added_method3.name == 'case3_method', f'Method name should remain case3_method, got {added_method3.name}'
    print(f'   ✓ Case 3 passed: method_pars values replaced those in method object')

    # Case 4: method=None and method_pars!=None --> build a new fp.Method from method_pars
    print('  Testing Case 4: method_pars provided without method (build new Method)...')
    # Get source method to copy dur_use for the test
    temp_sim4 = fp.Sim(pars=pars, verbose=0)
    temp_sim4.init()
    source_method4 = temp_sim4.connectors.contraception.get_method('impl')
    method_pars4 = dict(
        name='case4_method',
        label='Case 4 Method',
        efficacy=0.999,
        modern=True,
        rel_dur_use=1.5,
        dur_use=source_method4.dur_use,  # Copy dur_use from source for test
    )
    intv4 = fp.add_method(year=2010, method_pars=method_pars4, copy_from='impl', split_shares=0.3, verbose=False)
    sim4 = fp.Sim(pars=pars, interventions=[intv4], verbose=0)
    sim4.run()
    cm4 = sim4.connectors.contraception
    
    assert method_pars4['name'] in cm4.methods, 'New method should be in contraception methods'
    added_method4 = cm4.methods[method_pars4['name']]
    assert added_method4.efficacy == 0.999, f'Method efficacy should be 0.999, got {added_method4.efficacy}'
    assert added_method4.modern == True, f'Method modern should be True, got {added_method4.modern}'
    assert added_method4.rel_dur_use == 1.5, f'Method rel_dur_use should be 1.5, got {added_method4.rel_dur_use}'
    
    # Check that for a given age key, the method_choice_pars have been updated correctly
    assert np.array_equal(cm4.get_switching_matrix(0, 'case4_method')[age_key], cm4.get_switching_matrix(0, 'impl')[age_key]), 'Method choice switching matrix not updated correctly for new method'
    p1 = cm4.get_switching_prob('pill', 'impl', 0, age_key)
    p2 = cm4.get_switching_prob('pill', 'case4_method', 0, age_key)
    assert np.isclose((p1+p2)*0.7, p1), f'Method choice probabilities not updated correctly for new method'
    assert np.isclose((p1+p2)*0.3, p2), f'Method choice probabilities not updated correctly for new method'
    print(f'   ✓ Case 4 passed: new Method built from method_pars')
    print(f'   ✓ Switching probabilities updated correctly: {p1} -> {p2}')

    # Case 4b: method=None and partial method_pars --> build a new fp.Method from partial method_pars
    print('  Testing Case 4b: partial method_pars provided without method (build new Method with defaults)...')
    # Get source method to copy dur_use for the test
    temp_sim4b = fp.Sim(pars=pars, verbose=0)
    temp_sim4b.init()
    source_method4b = temp_sim4b.connectors.contraception.get_method('impl')
    partial_method_pars = dict(
        name='partial_method',
        efficacy=0.97,
        dur_use=source_method4b.dur_use,  # Copy dur_use from source for test
        # Note: not providing label, modern, rel_dur_use, etc.
    )
    intv4b = fp.add_method(year=2010, method=None, method_pars=partial_method_pars, copy_from='impl', split_shares=0.3, verbose=False)
    sim4b = fp.Sim(pars=pars, interventions=[intv4b], verbose=0)
    sim4b.run()
    cm4b = sim4b.connectors.contraception
    
    assert partial_method_pars['name'] in cm4b.methods, 'New method should be in contraception methods'
    added_method4b = cm4b.methods[partial_method_pars['name']]
    assert added_method4b.name == 'partial_method', f'Method name should be partial_method, got {added_method4b.name}'
    assert added_method4b.efficacy == 0.97, f'Method efficacy should be 0.97, got {added_method4b.efficacy}'
    # When copying from source, label comes from source unless explicitly overridden in method_pars
    # Since we're copying from 'impl' (Implants), label will be 'Implants' unless overridden
    assert added_method4b.label == 'Implants', f'Method label should be copied from source (Implants), got {added_method4b.label}'
    # Other properties (modern, dur_use) are copied from source method
    assert added_method4b.modern == source_method4b.modern, f'Method modern should be copied from source'
    print(f'   ✓ Case 4b passed: new Method built from partial method_pars (efficacy={added_method4b.efficacy}, label={added_method4b.label})')

    print(f'  ✓ All add_method cases work correctly')

    return sim4


if __name__ == '__main__':
    s0 = test_intervention_fn()
    s1 = test_change_par()
    s3 = test_plot()
    s4, s5, s6 = test_change_people_state()
    s7 = test_add_method()

    print('Done.')


