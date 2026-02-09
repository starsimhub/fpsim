"""
Run tests on the interventions.
"""

import sciris as sc
import starsim as ss
import numpy as np
import pytest
import fpsim as fp  # noqa: E402

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


def test_update_methods():
    sc.heading('Testing updates for methods...')

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


def test_copy_method():
    """ Test that using add_method with just copy_from works as intended. """
    sc.heading('Testing that add_method() can be used to directly copy a method...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    intv = fp.add_method(year=2001, copy_from='impl', verbose=False)

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


def test_add_method_pars():
    """Core: method_pars overrides attributes on the copied (or provided) method."""
    sc.heading('Testing add_method() method_pars overrides...')

    pars = dict(n_agents=200, start=2000, stop=2002, verbose=0, location='kenya')
    method_pars = dict(
        name='new_implant',
        label='New Implant (test)',
        efficacy=0.999,
        rel_dur_use=1.25,
    )
    intv = fp.add_method(year=2001, method_pars=method_pars, copy_from='impl', verbose=False)

    sim = fp.Sim(pars=pars, interventions=[intv], verbose=0)
    sim.init()
    cm = sim.connectors.contraception

    # Check that any newly specified values for the method have been set
    assert method_pars['name'] in cm.methods
    m = cm.methods[method_pars['name']]
    assert m.label == method_pars['label']
    assert m.efficacy == method_pars['efficacy']
    assert m.rel_dur_use == method_pars['rel_dur_use']

    # Check that any values for the method that weren't specified have been copied from the source method
    src = cm.get_method('impl')
    assert m.modern == src.modern
    assert np.array_equal(m.dur_use.pars.scale(sim, sim.people.uid), src.dur_use.pars.scale(sim, sim.people.uid))

    return sim


def test_add_method_split():
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


def test_age_restricted_initiation_basic():
    """Test that age-restricted initiation correctly filters by age using enhanced change_initiation."""
    sc.heading('Testing age-restricted initiation (via change_initiation) basic functionality...')
    
    # Use seed for reproducibility; larger n_agents and broader age_range so enough
    # women are eligible and we reliably get at least one user of the test method
    np.random.seed(42)
    pars = dict(n_agents=1200, start=2000, stop=2010, verbose=0, location='kenya')
    
    # Add a new method to target
    add_intv = fp.add_method(
        year=2001,
        method=None,
        method_pars={
            'name': 'test_method',
            'label': 'Test Method',
            'efficacy': 0.95,
            'modern': True,
            'dur_use': ss.lognorm_ex(mean=2, std=1)
        },
        copy_from='pill',
        split_shares=0,
        verbose=False,
        name='add_test_method'
    )
    
    # Age-restricted initiation using enhanced change_initiation
    init_intv = fp.change_initiation(
        years=[2005, 2010],
        age_range=(15, 30),  # Broader range so enough eligible women for robust test
        perc=0.12,
        perc_of_eligible=True,  # Apply to eligible women in age range
        target_method='test_method',
        annual=False,
        verbose=False,
        name='init_test_method'
    )
    
    sim = fp.Sim(pars=pars, interventions=[add_intv, init_intv], verbose=0)
    sim.run()
    
    # Check that some women are using the test method
    cm = sim.connectors.contraception
    test_method_idx = cm.methods['test_method'].idx
    users = sim.people.fp.method == test_method_idx
    
    assert users.sum() > 0, 'No users of test method found'
    
    # Check that users are in the target age range (or were when initiated)
    # This is approximate since people age during simulation
    user_ages = sim.people.age[users]
    # Allow some buffer since people aged during sim (age_range is 15-30)
    assert user_ages.min() >= 14, f'Found user younger than expected: {user_ages.min()}'
    assert user_ages.max() <= 32, f'Found user older than expected: {user_ages.max()}'
    
    print(f'✓ Age-restricted initiation created {users.sum()} users in target age range')
    return sim


def test_age_restricted_initiation_time_varying():
    """Test that change_initiation with final_perc scales up over time."""
    sc.heading('Testing change_initiation with time-varying rates...')
    
    # Use larger population and broader age range to ensure we have eligible women
    pars = dict(n_agents=2000, start=2000, stop=2015, verbose=0, location='kenya')
    
    # Use method_pars approach to avoid distribution copying issues
    add_intv = fp.add_method(
        year=2001, 
        method=None, 
        method_pars={
            'name': 'test_method2',
            'label': 'Test Method 2',
            'efficacy': 0.95,
            'modern': True,
            'dur_use': ss.lognorm_ex(mean=2, std=1)
        },
        copy_from='pill', 
        split_shares=0, 
        verbose=False, 
        name='add_test_method2'
    )
    
    # Scale from 5% to 10% over 10 years (higher rates to ensure some uptake)
    init_intv = fp.change_initiation(
        years=[2005, 2015],
        age_range=(15, 35),  # Broader age range
        perc=0.05,           # Higher initial rate
        final_perc=0.10,     # Higher final rate
        perc_of_eligible=True,
        target_method='test_method2',
        annual=True,
        verbose=False,
    )
    
    sim = fp.Sim(pars=pars, interventions=[add_intv, init_intv], verbose=0)
    sim.run()
    
    cm = sim.connectors.contraception
    test_method_idx = cm.methods['test_method2'].idx
    users = (sim.people.fp.method == test_method_idx).sum()
    
    # Should have some users with time-varying initiation
    assert users > 0, f'No users found with time-varying initiation (had {users} users)'
    
    print(f'✓ Time-varying initiation created {users} users')
    return sim


def test_age_restricted_initiation_errors():
    """Test error handling for age-restricted features in change_initiation."""
    sc.heading('Testing change_initiation error handling...')
    
    pars = dict(n_agents=100, start=2000, stop=2010, verbose=0, location='kenya')
    
    # Invalid age_range (not a tuple of 2)
    with pytest.raises(ValueError, match='age_range must be a tuple'):
        intv = fp.change_initiation(years=[2005, 2010], age_range=(20,), perc=0.01, perc_of_eligible=True)
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    # Invalid target method
    with pytest.raises(ValueError, match='not found'):
        intv = fp.change_initiation(
            years=[2005, 2010], 
            age_range=(0, 20),
            perc=0.01,
            perc_of_eligible=True,
            target_method='nonexistent_method'
        )
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    # Years outside sim range
    with pytest.raises(ValueError, match='before the start'):
        intv = fp.change_initiation(years=[1995, 2005], perc=0.01)
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    print('✓ Error handling works correctly')
    return


def test_method_switching_basic():
    """Test basic method switching functionality."""
    sc.heading('Testing method_switching basic functionality...')
    
    pars = dict(n_agents=1000, start=2000, stop=2010, verbose=0, location='kenya')
    
    # Add a target method
    add_target = fp.add_method(
        year=2001, 
        method=None,
        method_pars={
            'name': 'target_method',
            'label': 'Target Method',
            'efficacy': 0.96,
            'modern': True,
            'dur_use': ss.lognorm_ex(mean=2, std=1)
        },
        copy_from='pill', 
        split_shares=0, 
        verbose=False, 
        name='add_target'
    )
    
    # Switch from pill to target_method
    switch_intv = fp.method_switching(
        year=2005,
        from_methods='pill',
        to_method='target_method',
        switch_prob=0.20,
        annual=False,
        verbose=False,
    )
    
    sim = fp.Sim(pars=pars, interventions=[add_target, switch_intv], verbose=0)
    sim.init()
    
    cm = sim.connectors.contraception
    
    # Get switching probability BEFORE intervention
    pill_idx = cm.methods['pill'].idx
    target_idx = cm.methods['target_method'].idx
    
    # Check in a sample age group and pp state
    age_grp = '<18' if '<18' in cm.pars.method_choice_pars[0] else list(cm.pars.method_choice_pars[0].keys())[1]
    prob_before = cm.get_switching_prob('pill', 'target_method', postpartum=0, age_grp=age_grp)
    
    # Run simulation
    sim.run()
    
    # Get switching probability AFTER intervention
    prob_after = cm.get_switching_prob('pill', 'target_method', postpartum=0, age_grp=age_grp)
    
    # The probability should have increased
    assert prob_after > prob_before, f'Switching probability did not increase: {prob_before} -> {prob_after}'
    
    print(f'✓ Method switching increased probability: {prob_before:.4f} -> {prob_after:.4f}')
    
    return sim


def test_method_switching_multiple_sources():
    """Test switching from multiple source methods."""
    sc.heading('Testing method_switching with multiple sources...')
    
    pars = dict(n_agents=1000, start=2000, stop=2010, verbose=0, location='kenya')
    
    add_target = fp.add_method(
        year=2005, 
        method=None,
        method_pars={
            'name': 'target',
            'label': 'Target Method',
            'efficacy': 0.97,
            'modern': True,
            'dur_use': ss.lognorm_ex(mean=2, std=1)
        },
        copy_from='pill', 
        split_shares=0, 
        verbose=False, 
        name='add_target'
    )
    
    # Switch from existing methods that have users
    switch_intv = fp.method_switching(
        year=2006,
        from_methods=['pill', 'inj'],  # Use methods that exist and have users
        to_method='target',
        switch_prob={'pill': 0.10, 'inj': 0.10},
        annual=False,
        verbose=False,
    )
    
    sim = fp.Sim(pars=pars, interventions=[add_target, switch_intv], verbose=0)
    sim.run()
    
    cm = sim.connectors.contraception
    target_idx = cm.methods['target'].idx
    users = (sim.people.fp.method == target_idx).sum()
    
    assert users > 0, f'No users switched to target method (had {users} users)'
    print(f'✓ Multi-source switching created {users} users')
    
    return sim


def test_method_switching_errors():
    """Test error handling for method_switching."""
    sc.heading('Testing method_switching error handling...')
    
    pars = dict(n_agents=100, start=2000, stop=2010, verbose=0, location='kenya')
    
    # Missing year
    with pytest.raises(ValueError, match='year must be specified'):
        intv = fp.method_switching(year=None, from_methods='pill', to_method='iud', switch_prob=0.1)
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    # Missing switch_prob
    with pytest.raises(ValueError, match='switch_prob must be specified'):
        intv = fp.method_switching(year=2005, from_methods='pill', to_method='iud', switch_prob=None)
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    # Invalid source method
    with pytest.raises(ValueError, match='not found'):
        intv = fp.method_switching(year=2005, from_methods='nonexistent', to_method='pill', switch_prob=0.1)
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    # Invalid target method
    with pytest.raises(ValueError, match='not found'):
        intv = fp.method_switching(year=2005, from_methods='pill', to_method='nonexistent', switch_prob=0.1)
        fp.Sim(pars=pars, interventions=[intv], verbose=0).init()
    
    print('✓ Error handling works correctly')
    return


def test_dmpasc_scenario_integration():
    """Integration test: Run a simplified DMPA-SC scenario.
    (subcutaneous depot medroxyprogesterone acetate
    injectable contraceptive)
    """
    
    sc.heading('Testing DMPA-SC scenario integration...')
    
    np.random.seed(43)  # Reproducibility; ensures initiation/switching yield some users
    pars = dict(n_agents=1200, start=2000, stop=2030, verbose=0, location='kenya')
    
    # Baseline growth
    baseline = fp.change_initiation(years=[2000, 2030], perc=0.02, annual=True)
    
    # Add DMPA-SC method (needs a base duration distribution)
    dmpasc = fp.Method(
        name='dmpasc',
        label='DMPA-SC',
        efficacy=0.983,
        modern=True,
        dur_use=ss.lognorm_ex(mean=2, std=1),  # Base duration
        rel_dur_use=2.0,  # 2x scaling factor
    )
    add_dmpasc = fp.add_method(year=2020, method=dmpasc, copy_from='inj', split_shares=0, verbose=False, name='add_dmpasc')
    
    # Age-restricted initiation using enhanced change_initiation
    initiation = fp.change_initiation(
        years=[2020, 2030],
        age_range=(15, 25),  # Fecund, sexually active ages so enough eligible women
        perc=0.03,
        final_perc=0.08,
        perc_of_eligible=True,
        target_method='dmpasc',
        annual=True,
        verbose=False,
        name='initiation_dmpasc'
    )
    
    # Switching from injectables
    switching = fp.method_switching(
        year=2020,
        from_methods='inj',
        to_method='dmpasc',
        switch_prob=0.10,
        annual=False,
        verbose=False,
    )
    
    interventions = [baseline, add_dmpasc, initiation, switching]
    sim = fp.Sim(pars=pars, interventions=interventions, verbose=0)
    sim.run()
    
    # Verify DMPA-SC has users
    cm = sim.connectors.contraception
    dmpasc_idx = cm.methods['dmpasc'].idx
    users = (sim.people.fp.method == dmpasc_idx).sum()
    
    assert users > 0, 'DMPA-SC has no users in integration test'
    
    # Verify mCPR increased
    mcpr_final = sim.results.contraception.mcpr[-1]
    mcpr_initial = sim.results.contraception.mcpr[0]
    assert mcpr_final > mcpr_initial, 'mCPR did not increase'
    
    print(f'✓ Integration test passed: {users} DMPA-SC users, mCPR: {mcpr_initial:.1%} → {mcpr_final:.1%}')
    
    return sim


if __name__ == '__main__':
    s0 = test_intervention_fn()
    s1 = test_change_par()
    s3 = test_update_methods()
    s4, s5, s6 = test_change_people_state()
    s7 = test_copy_method()
    s8 = test_add_method_pars()
    s9 = test_add_method_split()
    test_add_method_negative_errors()
    s10 = test_add_method_split_shares_boundaries(0.0, 1.0, 0.0)
    s11 = test_add_method_split_shares_boundaries(1.0, 0.0, 1.0)
    s12 = test_add_method_year_boundaries(0)
    s13 = test_add_method_year_boundaries(2)
    
    # New intervention tests
    s14 = test_age_restricted_initiation_basic()
    s15 = test_age_restricted_initiation_time_varying()
    test_age_restricted_initiation_errors()
    s16 = test_method_switching_basic()
    s17 = test_method_switching_multiple_sources()
    test_method_switching_errors()
    s18 = test_dmpasc_scenario_integration()

    print('Done.')


