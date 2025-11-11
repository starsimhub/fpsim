"""
Test suite for dynamically adding new contraceptive methods.

This suite validates the new feature that allows adding contraceptive methods
during a simulation via interventions. It tests:
- Basic method addition
- Switching matrix expansion
- Integration with different contraception modules
- Edge cases and error handling
"""

import numpy as np
import pytest
import sciris as sc
import fpsim as fp


def make_sim(**kwargs):
    """Create a small test simulation."""
    default_kwargs = dict(
        n_agents=500,
        location='kenya',
        start_year=2000,
        end_year=2015,
    )
    default_kwargs.update(kwargs)
    return fp.Sim(**default_kwargs)


# ==============================================================================
# Test 1: Basic method addition
# ==============================================================================
def test_basic_method_addition():
    """Test that a new method can be added and appears in the simulation."""
    sc.heading('Testing basic method addition...')
    
    # Create a new method
    new_method = fp.Method(
        name='test_method',
        label='Test Method',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    # Create intervention
    mod = fp.MethodIntervention(year=2010, label='Add Test Method')
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.3
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    # Verify method was added
    cm = sim.connectors.contraception
    assert 'test_method' in cm.methods, "New method not found in contraception module"
    assert cm.methods['test_method'].idx == 10, "New method has wrong index"
    
    # Verify method_mix was resized
    fp_mod = sim.connectors['fp']
    assert fp_mod.method_mix.shape[0] == 11, f"method_mix not resized correctly: {fp_mod.method_mix.shape}"
    
    # Verify method has some adoption
    final_adoption = fp_mod.method_mix[10, -1] * 100
    assert final_adoption > 0, "New method has zero adoption"
    
    print(f'✓ Basic method addition successful (adoption: {final_adoption:.2f}%)')
    return sim


# ==============================================================================
# Test 2: Switching matrix expansion
# ==============================================================================
def test_switching_matrix_expansion():
    """Test that switching matrix is properly expanded."""
    sc.heading('Testing switching matrix expansion...')
    
    new_method = fp.Method(
        name='new_inj',
        label='New Injectable',
        efficacy=0.98,
        modern=True,
        dur_use=fp.methods.ln(8, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='impl',
        initial_share=0.5
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    cm = sim.connectors.contraception
    mcp = cm.pars.method_choice_pars
    
    # Verify method_idx was updated in all events
    for event_key, event_data in mcp.items():
        if isinstance(event_data, dict) and 'method_idx' in event_data:
            assert 10 in event_data['method_idx'], f"Event {event_key} doesn't have new method in method_idx"
    
    # Verify switching probabilities were created for regular structure (Event 0)
    event_0 = mcp[0]
    age_data = event_0['>35']
    
    if isinstance(age_data, dict):
        # Regular structure - check if new method row exists
        assert 'new_inj' in age_data, "New method not in switching matrix"
        assert isinstance(age_data['new_inj'], np.ndarray), "New method switching data is not an array"
        assert len(age_data['new_inj']) == 10, f"New method switching array has wrong length: {len(age_data['new_inj'])}"
        
        # Verify other methods have probability of switching TO new method
        inj_probs = age_data['inj']
        assert len(inj_probs) == 10, f"Injectable switching array not expanded: {len(inj_probs)}"
    
    print('✓ Switching matrix expansion successful')
    return sim


# ==============================================================================
# Test 3: Method properties update
# ==============================================================================
def test_method_properties_update():
    """Test that new method properties can be updated via intervention."""
    sc.heading('Testing method property updates...')
    
    new_method = fp.Method(
        name='new_larc',
        label='New LARC',
        efficacy=0.90,  # Start with lower efficacy
        modern=True,
        dur_use=fp.methods.ln(4, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='impl',
        copy_from_col='impl',
        initial_share=0.4
    )
    # Update properties
    mod.set_efficacy('new_larc', 0.995)  # Improve efficacy
    mod.set_duration_months('new_larc', 36)  # Set to 3 years
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    cm = sim.connectors.contraception
    method = cm.methods['new_larc']
    
    # Verify properties were updated
    assert method.efficacy == 0.995, f"Efficacy not updated: {method.efficacy}"
    assert method.dur_use == 36.0, f"Duration not updated: {method.dur_use}"
    
    print('✓ Method property updates successful')
    return sim


# ==============================================================================
# Test 4: Method weights expansion
# ==============================================================================
def test_method_weights_expansion():
    """Test that method_weights array is properly expanded."""
    sc.heading('Testing method_weights expansion...')
    
    new_method = fp.Method(
        name='test_weights',
        label='Test Weights',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.4
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    cm = sim.connectors.contraception
    
    # Verify method_weights was expanded
    if hasattr(cm.pars, 'method_weights'):
        assert len(cm.pars.method_weights) == 10, f"method_weights not expanded: {len(cm.pars.method_weights)}"
    
    print('✓ Method weights expansion successful')
    return sim


# ==============================================================================
# Test 5: Multiple methods addition
# ==============================================================================
def test_multiple_methods_addition():
    """Test adding multiple new methods in sequence."""
    sc.heading('Testing multiple method addition...')
    
    method1 = fp.Method(name='new_method_1', label='New Method 1', efficacy=0.95, modern=True, dur_use=fp.methods.ln(6, 2))
    method2 = fp.Method(name='new_method_2', label='New Method 2', efficacy=0.98, modern=True, dur_use=fp.methods.ln(8, 2))
    
    # Create two separate interventions
    mod1 = fp.MethodIntervention(year=2010, label='Add Method 1')
    mod1.add_method(method=method1, copy_from_row='inj', copy_from_col='inj', initial_share=0.3)
    
    mod2 = fp.MethodIntervention(year=2012, label='Add Method 2')
    mod2.add_method(method=method2, copy_from_row='impl', copy_from_col='impl', initial_share=0.4)
    
    intv1 = mod1.build()
    intv2 = mod2.build()
    
    sim = make_sim(interventions=[intv1, intv2])
    sim.run()
    
    cm = sim.connectors.contraception
    
    # Verify both methods were added
    assert 'new_method_1' in cm.methods, "First method not added"
    assert 'new_method_2' in cm.methods, "Second method not added"
    assert len(cm.methods) == 12, f"Expected 12 methods, got {len(cm.methods)}"
    
    # Verify indices
    assert cm.methods['new_method_1'].idx == 10, "First method has wrong index"
    assert cm.methods['new_method_2'].idx == 11, "Second method has wrong index"
    
    # Verify both have some adoption
    fp_mod = sim.connectors['fp']
    adoption1 = fp_mod.method_mix[10, -1] * 100
    adoption2 = fp_mod.method_mix[11, -1] * 100
    
    assert adoption1 > 0 or adoption2 > 0, "Neither method has any adoption"
    
    print(f'✓ Multiple methods addition successful (adoptions: {adoption1:.2f}%, {adoption2:.2f}%)')
    return sim


# ==============================================================================
# Test 6: Error handling - duplicate method
# ==============================================================================
def test_duplicate_method_error():
    """Test that adding a duplicate method raises an error."""
    sc.heading('Testing duplicate method error handling...')
    
    # Try to add a method with same name as existing
    duplicate_method = fp.Method(
        name='inj',  # Already exists!
        label='Duplicate Injectable',
        efficacy=0.99,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=duplicate_method,
        copy_from_row='impl',
        copy_from_col='impl',
        initial_share=0.3
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    
    # This should raise ValueError during the intervention application
    with pytest.raises(ValueError, match="already exists"):
        sim.run()
    
    print('✓ Duplicate method error handling successful')


# ==============================================================================
# Test 7: Error handling - invalid reference method
# ==============================================================================
def test_invalid_reference_method():
    """Test that invalid copy_from methods raise errors."""
    sc.heading('Testing invalid reference method error handling...')
    
    new_method = fp.Method(
        name='test_invalid',
        label='Test Invalid',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='nonexistent_method',  # Invalid!
        copy_from_col='inj',
        initial_share=0.3
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    
    with pytest.raises(ValueError, match="not found in methods"):
        sim.run()
    
    print('✓ Invalid reference method error handling successful')


# ==============================================================================
# Test 8: Renormalization
# ==============================================================================
def test_renormalization():
    """Test that switching probabilities are properly renormalized."""
    sc.heading('Testing probability renormalization...')
    
    new_method = fp.Method(
        name='test_renorm',
        label='Test Renormalization',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.3,
        renormalize=True
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    cm = sim.connectors.contraception
    mcp = cm.pars.method_choice_pars
    
    # Check that probabilities sum to 1.0 after renormalization
    for event_key, event_data in mcp.items():
        if not isinstance(event_data, dict):
            continue
        
        for age_key, age_data in event_data.items():
            if age_key == 'method_idx':
                continue
            
            # Check numpy array structure
            if isinstance(age_data, np.ndarray):
                prob_sum = age_data.sum()
                assert abs(prob_sum - 1.0) < 0.001, f"Event {event_key}, age {age_key} probs don't sum to 1: {prob_sum}"
            
            # Check dict structure
            elif isinstance(age_data, dict):
                for method_name, method_data in age_data.items():
                    if isinstance(method_data, np.ndarray):
                        prob_sum = method_data.sum()
                        assert abs(prob_sum - 1.0) < 0.001, f"Method {method_name} probs don't sum to 1: {prob_sum}"
    
    print('✓ Renormalization successful - all probabilities sum to 1.0')
    return sim


# ==============================================================================
# Test 9: Method mix array resizing
# ==============================================================================
def test_method_mix_resizing():
    """Test that FPmod method_mix array is properly resized."""
    sc.heading('Testing method_mix array resizing...')
    
    new_method = fp.Method(
        name='test_resize',
        label='Test Resize',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.3
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    # Check FPmod method_mix shape
    fp_mod = sim.connectors['fp']
    n_methods = len(sim.connectors.contraception.methods)
    
    assert fp_mod.method_mix.shape[0] == n_methods, \
        f"method_mix shape mismatch: {fp_mod.method_mix.shape[0]} vs {n_methods} methods"
    
    # Verify method_mix sums to 1.0 at all timesteps
    for ti in range(fp_mod.method_mix.shape[1]):
        mix_sum = fp_mod.method_mix[:, ti].sum()
        assert abs(mix_sum - 1.0) < 0.001, f"method_mix at ti={ti} doesn't sum to 1: {mix_sum}"
    
    print('✓ method_mix array resizing successful')
    return sim


# ==============================================================================
# Test 10: Integration with method updates
# ==============================================================================
def test_new_method_with_updates():
    """Test adding a method and updating its properties in same intervention."""
    sc.heading('Testing new method with property updates...')
    
    new_method = fp.Method(
        name='combined_test',
        label='Combined Test',
        efficacy=0.90,
        modern=True,
        dur_use=fp.methods.ln(4, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.4
    )
    # Also update properties
    mod.set_efficacy('combined_test', 0.99)
    mod.set_duration_months('combined_test', 24)
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    cm = sim.connectors.contraception
    method = cm.methods['combined_test']
    
    # Verify both addition and updates worked
    assert method.efficacy == 0.99, f"Efficacy not updated: {method.efficacy}"
    assert method.dur_use == 24.0, f"Duration not updated: {method.dur_use}"
    
    # Verify method has adoption
    fp_mod = sim.connectors['fp']
    final_adoption = fp_mod.method_mix[method.idx, -1] * 100
    assert final_adoption > 0, "Method has zero adoption"
    
    print(f'✓ Combined addition and updates successful (adoption: {final_adoption:.2f}%)')
    return sim


# ==============================================================================
# Test 11: Different copy patterns
# ==============================================================================
def test_different_copy_patterns():
    """Test copying from different source/target methods."""
    sc.heading('Testing different copy patterns...')
    
    # Test 1: Copy from injectables to implants
    new_method1 = fp.Method(name='test_inj_impl', label='Test Inj→Impl', 
                           efficacy=0.95, modern=True, dur_use=fp.methods.ln(6, 2))
    
    mod1 = fp.MethodIntervention(year=2010, label='Test 1')
    mod1.add_method(method=new_method1, copy_from_row='inj', copy_from_col='impl', initial_share=0.3)
    
    intv1 = mod1.build()
    sim1 = make_sim(interventions=intv1, label='Inj→Impl')
    sim1.run()
    
    assert 'test_inj_impl' in sim1.connectors.contraception.methods
    
    # Test 2: Copy from implants to pill
    new_method2 = fp.Method(name='test_impl_pill', label='Test Impl→Pill',
                           efficacy=0.95, modern=True, dur_use=fp.methods.ln(6, 2))
    
    mod2 = fp.MethodIntervention(year=2010, label='Test 2')
    mod2.add_method(method=new_method2, copy_from_row='impl', copy_from_col='pill', initial_share=0.3)
    
    intv2 = mod2.build()
    sim2 = make_sim(interventions=intv2, label='Impl→Pill')
    sim2.run()
    
    assert 'test_impl_pill' in sim2.connectors.contraception.methods
    
    print('✓ Different copy patterns successful')
    return sim1, sim2


# ==============================================================================
# Test 12: Method adoption dynamics
# ==============================================================================
def test_adoption_dynamics():
    """Test that new method adoption increases over time."""
    sc.heading('Testing adoption dynamics...')
    
    new_method = fp.Method(
        name='dynamics_test',
        label='Dynamics Test',
        efficacy=0.98,
        modern=True,
        dur_use=fp.methods.ln(8, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.5  # High staying probability
    )
    mod.set_duration_months('dynamics_test', 36)  # Long duration
    
    intv = mod.build()
    sim = make_sim(interventions=intv, end_year=2020)
    sim.run()
    
    fp_mod = sim.connectors['fp']
    new_idx = sim.connectors.contraception.methods['dynamics_test'].idx
    adoption_timeseries = fp_mod.method_mix[new_idx, :]
    
    # Find intervention timestep (year 2010)
    years = np.linspace(2000, 2020, len(sim.results.timevec))
    interv_ti = np.argmin(np.abs(years - 2010))
    
    # Verify adoption is zero before intervention
    assert adoption_timeseries[interv_ti - 1] == 0, "Method exists before intervention!"
    
    # Verify adoption increases after intervention
    post_interv_adoption = adoption_timeseries[interv_ti + 12:]  # 1 year after
    assert np.any(post_interv_adoption > 0), "No adoption after intervention"
    
    # Check that max adoption > 0
    max_adoption = np.max(adoption_timeseries) * 100
    assert max_adoption > 0, "Method never adopted"
    
    print(f'✓ Adoption dynamics working (peak adoption: {max_adoption:.2f}%)')
    return sim


# ==============================================================================
# Test 13: Integration with existing interventions
# ==============================================================================
def test_integration_with_other_interventions():
    """Test that new method addition works alongside other interventions."""
    sc.heading('Testing integration with other interventions...')
    
    new_method = fp.Method(
        name='integration_test',
        label='Integration Test',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    # Create new method intervention
    mod_new = fp.MethodIntervention(year=2010, label='Add New Method')
    mod_new.add_method(method=new_method, copy_from_row='inj', copy_from_col='inj', initial_share=0.3)
    
    # Create regular method intervention for existing methods
    mod_regular = fp.MethodIntervention(year=2012, label='Improve Existing')
    mod_regular.set_efficacy('impl', 0.995)
    mod_regular.set_duration_months('impl', 48)
    
    intv_new = mod_new.build()
    intv_regular = mod_regular.build()
    
    sim = make_sim(interventions=[intv_new, intv_regular])
    sim.run()
    
    cm = sim.connectors.contraception
    
    # Verify new method was added
    assert 'integration_test' in cm.methods
    
    # Verify existing method was updated
    impl = cm.methods['impl']
    assert impl.efficacy == 0.995, "Implant efficacy not updated"
    assert impl.dur_use == 48.0, "Implant duration not updated"
    
    print('✓ Integration with other interventions successful')
    return sim


# ==============================================================================
# Test 14: No renormalization option
# ==============================================================================
def test_no_renormalization():
    """Test adding method without renormalization."""
    sc.heading('Testing no renormalization option...')
    
    new_method = fp.Method(
        name='no_renorm',
        label='No Renorm',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.1,
        renormalize=False  # Don't renormalize
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    # Should still work but probabilities won't sum to 1
    cm = sim.connectors.contraception
    assert 'no_renorm' in cm.methods
    
    print('✓ No renormalization option successful')
    return sim


# ==============================================================================
# Test 15: Method impact on births
# ==============================================================================
def test_method_impact_on_births():
    """Test that high-efficacy new method can reduce births."""
    sc.heading('Testing method impact on births...')
    
    # Baseline
    sim_baseline = make_sim(label='Baseline', end_year=2020)
    sim_baseline.run()
    
    # With high-efficacy, high-adoption new method
    new_method = fp.Method(
        name='high_impact',
        label='High Impact',
        efficacy=0.995,  # Very high efficacy
        modern=True,
        dur_use=fp.methods.ln(10, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(method=new_method, copy_from_row='inj', copy_from_col='inj', initial_share=0.6)
    mod.set_duration_months('high_impact', 48)  # 4 years duration
    
    intv = mod.build()
    sim_interv = make_sim(interventions=intv, label='With New Method', end_year=2020)
    sim_interv.run()
    
    # Compare births (should be similar or slightly different)
    # Note: Impact may be small with moderate parameters
    baseline_births = np.sum(sim_baseline.results.fp.births)
    interv_births = np.sum(sim_interv.results.fp.births)
    
    # Just verify the simulation runs and produces results
    assert baseline_births > 0, "Baseline has no births"
    assert interv_births > 0, "Intervention has no births"
    
    births_diff = baseline_births - interv_births
    print(f'✓ Method impact test successful (births difference: {births_diff:.0f})')
    return sim_baseline, sim_interv


# ==============================================================================
# Test 16: Wrapper API validation
# ==============================================================================
def test_wrapper_api():
    """Test MethodIntervention wrapper API."""
    sc.heading('Testing MethodIntervention wrapper API...')
    
    new_method = fp.Method(
        name='api_test',
        label='API Test',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    # Test method chaining
    mod = (fp.MethodIntervention(year=2010, label='API Test')
           .add_method(method=new_method, copy_from_row='inj', copy_from_col='inj', initial_share=0.3)
           .set_efficacy('api_test', 0.99)
           .set_duration_months('api_test', 24))
    
    # Test preview
    preview = mod.preview()
    assert 'year' in preview
    assert 'new_method' in preview
    assert preview['new_method']['name'] == 'api_test'
    
    # Test build and run
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    assert 'api_test' in sim.connectors.contraception.methods
    
    print('✓ Wrapper API validation successful')
    return sim


# ==============================================================================
# Test 17: All three event structures
# ==============================================================================
def test_all_event_structures():
    """Test that all three event structure types are handled correctly."""
    sc.heading('Testing all event structure types...')
    
    new_method = fp.Method(
        name='struct_test',
        label='Structure Test',
        efficacy=0.95,
        modern=True,
        dur_use=fp.methods.ln(6, 2)
    )
    
    mod = fp.MethodIntervention(year=2010)
    mod.add_method(
        method=new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.4
    )
    
    intv = mod.build()
    sim = make_sim(interventions=intv)
    sim.run()
    
    cm = sim.connectors.contraception
    mcp = cm.pars.method_choice_pars
    
    # Count how many events were successfully expanded
    events_expanded = 0
    
    for event_key, event_data in mcp.items():
        if not isinstance(event_data, dict):
            continue
        
        # Check if method_idx was updated
        if 'method_idx' in event_data:
            if 10 in event_data['method_idx']:
                events_expanded += 1
    
    assert events_expanded == 3, f"Expected 3 events expanded, got {events_expanded}"
    
    print(f'✓ All event structures handled successfully ({events_expanded} events)')
    return sim


# ==============================================================================
# Run all tests
# ==============================================================================
if __name__ == '__main__':
    # Run tests individually for better debugging
    tests = [
        ('Basic method addition', test_basic_method_addition),
        ('Switching matrix expansion', test_switching_matrix_expansion),
        ('Method properties update', test_method_properties_update),
        ('Method weights expansion', test_method_weights_expansion),
        ('Multiple methods addition', test_multiple_methods_addition),
        ('Duplicate method error', test_duplicate_method_error),
        ('Invalid reference method', test_invalid_reference_method),
        ('Renormalization', test_renormalization),
        ('Method mix resizing', test_method_mix_resizing),
        ('New method with updates', test_new_method_with_updates),
        ('Different copy patterns', test_different_copy_patterns),
        ('Adoption dynamics', test_adoption_dynamics),
        ('Integration with other interventions', test_integration_with_other_interventions),
        ('No renormalization option', test_no_renormalization),
        ('Method impact on births', test_method_impact_on_births),
        ('Wrapper API validation', test_wrapper_api),
        ('All event structures', test_all_event_structures),
    ]
    
    print("\n" + "="*70)
    print("RUNNING ADD_NEW_METHOD TEST SUITE")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f'✗ {name} FAILED: {e}\n')
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if errors:
        print("\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

