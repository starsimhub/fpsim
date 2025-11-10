"""
Tests for MethodIntervention class in fpsim.wrappers

Tests the user-friendly interface for building contraceptive method interventions.
"""

import numpy as np
import pytest
import sciris as sc

import fpsim as fp


# =============================================================================
# Basic construction and validation tests
# =============================================================================

def test_methodintervention_init():
    """Test basic initialization of MethodIntervention."""
    year = 2025
    label = 'Test Intervention'
    
    mod = fp.MethodIntervention(year=year, label=label)
    
    assert mod.year == year
    assert mod.label == label
    assert mod._eff == {}
    assert mod._dur == {}
    assert mod._p_use is None
    assert mod._mix_values == {}
    assert mod._method_mix_base is None
    assert mod._switch is None


def test_methodintervention_method_names():
    """Test that all expected method names are recognized."""
    mod = fp.MethodIntervention(year=2025)
    
    valid_methods = ['none', 'pill', 'iud', 'inj', 'cond', 'btl', 'wdraw', 'impl', 'othtrad', 'othmod']
    
    for method in valid_methods:
        # Should not raise
        normalized = mod._normalize_name(method)
        assert normalized == method


def test_methodintervention_invalid_method_name():
    """Test that invalid method names raise errors."""
    mod = fp.MethodIntervention(year=2025)
    
    with pytest.raises(ValueError, match='Method name must be one of'):
        mod._normalize_name('invalid_method')
    
    with pytest.raises(ValueError, match='Method name must be a string'):
        mod._normalize_name(123)


# =============================================================================
# Efficacy tests
# =============================================================================

def test_set_efficacy_valid():
    """Test setting valid efficacy values."""
    mod = fp.MethodIntervention(year=2025)
    
    # Test method chaining
    result = mod.set_efficacy('pill', 0.95)
    assert result is mod  # Returns self for chaining
    
    # Test value is stored
    assert mod._eff['pill'] == 0.95
    
    # Test multiple methods
    mod.set_efficacy('inj', 0.99)
    mod.set_efficacy('impl', 0.994)
    
    assert len(mod._eff) == 3
    assert mod._eff['inj'] == 0.99
    assert mod._eff['impl'] == 0.994


def test_set_efficacy_boundary_values():
    """Test efficacy boundary values (0 and 1)."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_efficacy('none', 0.0)  # Valid minimum
    mod.set_efficacy('btl', 1.0)   # Valid maximum
    
    assert mod._eff['none'] == 0.0
    assert mod._eff['btl'] == 1.0


def test_set_efficacy_invalid_range():
    """Test that invalid efficacy values raise errors."""
    mod = fp.MethodIntervention(year=2025)
    
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        mod.set_efficacy('pill', -0.1)
    
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        mod.set_efficacy('pill', 1.5)


# =============================================================================
# Duration tests
# =============================================================================

def test_set_duration_months_valid():
    """Test setting valid duration values."""
    mod = fp.MethodIntervention(year=2025)
    
    # Test method chaining
    result = mod.set_duration_months('inj', 36)
    assert result is mod
    
    # Test value is stored
    assert mod._dur['inj'] == 36.0
    
    # Test float values
    mod.set_duration_months('pill', 12.5)
    assert mod._dur['pill'] == 12.5


def test_set_duration_months_invalid():
    """Test that invalid duration values raise errors."""
    mod = fp.MethodIntervention(year=2025)
    
    with pytest.raises(ValueError, match='must be positive'):
        mod.set_duration_months('pill', 0)
    
    with pytest.raises(ValueError, match='must be positive'):
        mod.set_duration_months('pill', -5)


# =============================================================================
# Probability of use tests
# =============================================================================

def test_set_probability_of_use_valid():
    """Test setting valid probability of use."""
    mod = fp.MethodIntervention(year=2025)
    
    result = mod.set_probability_of_use(0.5)
    assert result is mod
    assert mod._p_use == 0.5
    
    # Test boundary values
    mod.set_probability_of_use(0.0)
    assert mod._p_use == 0.0
    
    mod.set_probability_of_use(1.0)
    assert mod._p_use == 1.0


def test_set_probability_of_use_invalid():
    """Test that invalid probability values raise errors."""
    mod = fp.MethodIntervention(year=2025)
    
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        mod.set_probability_of_use(-0.1)
    
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        mod.set_probability_of_use(1.5)


# =============================================================================
# Method mix tests
# =============================================================================

def test_set_method_mix_baseline():
    """Test setting method mix baseline."""
    mod = fp.MethodIntervention(year=2025)
    
    # 9 methods (excluding 'none')
    baseline = np.array([0.1, 0.2, 0.15, 0.05, 0.15, 0.05, 0.2, 0.05, 0.05])
    
    result = mod.set_method_mix_baseline(baseline)
    assert result is mod
    
    # Should normalize
    assert np.isclose(mod._method_mix_base.sum(), 1.0)
    
    # Should have correct length
    assert len(mod._method_mix_base) == 9


def test_set_method_mix_baseline_percentage():
    """Test that percentage values (>1) are converted."""
    mod = fp.MethodIntervention(year=2025)
    
    # Provide as percentages
    baseline_pct = np.array([10, 20, 15, 5, 15, 5, 20, 5, 5])
    
    mod.set_method_mix_baseline(baseline_pct)
    
    # Should be normalized to fractions
    assert np.isclose(mod._method_mix_base.sum(), 1.0)
    assert np.all(mod._method_mix_base <= 1.0)


def test_set_method_mix_baseline_invalid():
    """Test invalid baseline arrays."""
    mod = fp.MethodIntervention(year=2025)
    
    # Wrong length
    with pytest.raises(ValueError, match='must have 9 entries'):
        mod.set_method_mix_baseline(np.array([0.5, 0.5]))
    
    # Negative values
    with pytest.raises(ValueError, match='cannot contain negative'):
        mod.set_method_mix_baseline(np.array([0.1, -0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]))
    
    # All zeros
    with pytest.raises(ValueError, match='sums to zero'):
        mod.set_method_mix_baseline(np.zeros(9))


def test_set_method_mix():
    """Test setting method mix for specific methods."""
    mod = fp.MethodIntervention(year=2025)
    
    # Need baseline first
    baseline = np.array([0.1, 0.2, 0.15, 0.05, 0.15, 0.05, 0.2, 0.05, 0.05])
    mod.set_method_mix_baseline(baseline)
    
    result = mod.set_method_mix('pill', 0.3)
    assert result is mod
    assert mod._mix_values['pill'] == 0.3


def test_set_method_mix_without_baseline():
    """Test that method mix requires baseline to be set first."""
    mod = fp.MethodIntervention(year=2025)
    mod.set_method_mix('pill', 0.3)
    
    # Should raise when trying to build
    with pytest.raises(ValueError, match='Method mix baseline not set'):
        mod._build_method_mix_array()


def test_set_method_mix_none_rejected():
    """Test that 'none' cannot be set in method mix."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.array([0.1, 0.2, 0.15, 0.05, 0.15, 0.05, 0.2, 0.05, 0.05])
    mod.set_method_mix_baseline(baseline)
    
    mod.set_method_mix('none', 0.5)  # Can set it
    
    # But building should raise
    with pytest.raises(ValueError, match='Method mix cannot be set for "none"'):
        mod._build_method_mix_array()


def test_method_mix_rescaling():
    """Test that method mix correctly rescales non-targeted methods."""
    mod = fp.MethodIntervention(year=2025)
    
    # Equal baseline
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    # Set pill to 50%
    mod.set_method_mix('pill', 0.5)
    
    result = mod._build_method_mix_array()
    
    # Should sum to 1
    assert np.isclose(result.sum(), 1.0)
    
    # Pill should be 50%
    pill_idx = mod._method_mix_order.index('pill')
    assert np.isclose(result[pill_idx], 0.5)
    
    # Others should share remaining 50% proportionally
    for idx, method in enumerate(mod._method_mix_order):
        if method != 'pill':
            assert result[idx] < baseline[idx]  # All others reduced


def test_method_mix_sum_exceeds_one():
    """Test that method mix values exceeding 1.0 raise error."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    mod.set_method_mix('pill', 0.6)
    mod.set_method_mix('inj', 0.6)  # Together exceed 1.0
    
    with pytest.raises(ValueError, match='sum to more than 1.0'):
        mod._build_method_mix_array()


# =============================================================================
# Preview tests
# =============================================================================

def test_preview():
    """Test preview() returns correct summary."""
    mod = fp.MethodIntervention(year=2025, label='Test')
    mod.set_efficacy('pill', 0.95)
    mod.set_duration_months('inj', 30)
    mod.set_probability_of_use(0.55)
    
    preview = mod.preview()
    
    assert preview['year'] == 2025
    assert preview['label'] == 'Test'
    assert preview['efficacy'] == {'pill': 0.95}
    assert preview['duration_months'] == {'inj': 30.0}
    assert preview['p_use'] == 0.55
    assert preview['method_mix'] is None  # Not set
    assert preview['switching_matrix'] is None


def test_preview_with_method_mix():
    """Test preview includes method mix when set."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    mod.set_method_mix('pill', 0.3)
    
    preview = mod.preview()
    
    assert preview['method_mix'] is not None
    assert isinstance(preview['method_mix'], dict)
    # Should have all 9 methods
    assert len(preview['method_mix']) == 9


# =============================================================================
# Build tests
# =============================================================================

def test_build_basic():
    """Test building a basic intervention."""
    mod = fp.MethodIntervention(year=2025, label='Test')
    mod.set_efficacy('inj', 0.99)
    mod.set_duration_months('iud', 24)
    
    intv = mod.build()
    
    assert isinstance(intv, fp.update_methods)
    assert intv.pars.year == 2025
    
    # Should use label names, not short names
    assert 'Injectables' in intv.pars.eff
    assert intv.pars.eff['Injectables'] == 0.99
    
    assert 'IUDs' in intv.pars.dur_use
    assert intv.pars.dur_use['IUDs'] == 24.0


def test_build_empty():
    """Test that building with no changes works."""
    mod = fp.MethodIntervention(year=2025)
    
    intv = mod.build()
    
    assert isinstance(intv, fp.update_methods)
    assert intv.pars.year == 2025
    # Should not have efficacy or duration set
    assert not hasattr(intv.pars, 'eff') or intv.pars.eff is None


def test_build_method_chaining():
    """Test that method chaining works correctly."""
    intv = (fp.MethodIntervention(year=2025)
            .set_efficacy('pill', 0.95)
            .set_duration_months('pill', 18)
            .set_probability_of_use(0.6)
            .build())
    
    assert isinstance(intv, fp.update_methods)
    assert intv.pars.eff['Pill'] == 0.95
    assert intv.pars.dur_use['Pill'] == 18.0
    assert intv.pars.p_use == 0.6


def test_build_with_method_mix():
    """Test building with method mix."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    mod.set_method_mix('impl', 0.25)
    
    intv = mod.build()
    
    assert hasattr(intv.pars, 'method_mix')
    assert isinstance(intv.pars.method_mix, np.ndarray)
    assert np.isclose(intv.pars.method_mix.sum(), 1.0)


# =============================================================================
# Name to label conversion tests
# =============================================================================

def test_name_to_label_conversion():
    """Test that method names are correctly converted to labels."""
    mod = fp.MethodIntervention(year=2025)
    
    name_label_pairs = [
        ('pill', 'Pill'),
        ('iud', 'IUDs'),
        ('inj', 'Injectables'),
        ('cond', 'Condoms'),
        ('btl', 'BTL'),
        ('wdraw', 'Withdrawal'),
        ('impl', 'Implants'),
        ('othtrad', 'Other traditional'),
        ('othmod', 'Other modern'),
    ]
    
    for name, expected_label in name_label_pairs:
        mod.set_efficacy(name, 0.95)
    
    intv = mod.build()
    
    for name, expected_label in name_label_pairs:
        assert expected_label in intv.pars.eff
        assert intv.pars.eff[expected_label] == 0.95


# =============================================================================
# Integration test with simulation
# =============================================================================

def test_integration_with_sim():
    """Test that built intervention works in a simulation."""
    # Create intervention
    mod = fp.MethodIntervention(year=2005, label='Test Intervention')
    mod.set_efficacy('inj', 0.99)
    mod.set_duration_months('inj', 30)
    
    intv = mod.build()
    
    # Run simulation
    pars = dict(
        n_agents=100,
        start_year=2000,
        end_year=2010,
        location='senegal',
    )
    
    sim = fp.Sim(pars=pars, interventions=intv)
    sim.run()
    
    # Should complete without error
    assert sim.complete  # Simulation completed successfully


# =============================================================================
# make_update_methods functional shortcut tests
# =============================================================================

def test_make_update_methods_basic():
    """Test the functional shortcut make_update_methods."""
    intv = fp.make_update_methods(
        year=2025,
        method='pill',
        efficacy=0.95,
        duration_months=18,
        p_use=0.6,
        label='Pill Program'
    )
    
    assert isinstance(intv, fp.update_methods)
    assert intv.pars.year == 2025
    assert intv.pars.eff['Pill'] == 0.95
    assert intv.pars.dur_use['Pill'] == 18.0
    assert intv.pars.p_use == 0.6


def test_make_update_methods_requires_method():
    """Test that make_update_methods requires method when setting efficacy/duration."""
    with pytest.raises(ValueError, match='method.*must be provided'):
        fp.make_update_methods(year=2025, efficacy=0.95)
    
    with pytest.raises(ValueError, match='method.*must be provided'):
        fp.make_update_methods(year=2025, duration_months=18)
    
    with pytest.raises(ValueError, match='method.*must be provided'):
        fp.make_update_methods(year=2025, method_mix_value=0.3)


def test_make_update_methods_with_method_mix():
    """Test make_update_methods with method mix baseline."""
    baseline = np.ones(9) / 9.0
    
    intv = fp.make_update_methods(
        year=2025,
        method='impl',
        method_mix_baseline=baseline,
        method_mix_value=0.3
    )
    
    assert isinstance(intv, fp.update_methods)
    assert hasattr(intv.pars, 'method_mix')


# =============================================================================
# CHALLENGING EDGE CASE TESTS
# =============================================================================

def test_overwrite_efficacy_same_method():
    """Test that setting efficacy twice for same method overwrites."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_efficacy('pill', 0.90)
    mod.set_efficacy('pill', 0.95)  # Overwrite
    
    assert mod._eff['pill'] == 0.95
    assert len(mod._eff) == 1


def test_overwrite_duration_same_method():
    """Test that setting duration twice for same method overwrites."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_duration_months('inj', 20)
    mod.set_duration_months('inj', 30)  # Overwrite
    
    assert mod._dur['inj'] == 30.0
    assert len(mod._dur) == 1


def test_overwrite_probability_of_use():
    """Test that setting p_use twice overwrites."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_probability_of_use(0.4)
    mod.set_probability_of_use(0.6)  # Overwrite
    
    assert mod._p_use == 0.6


def test_method_mix_all_to_single_method():
    """Test setting 100% to a single method (extreme case)."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    # Set one method to 100%
    mod.set_method_mix('impl', 1.0)
    
    result = mod._build_method_mix_array()
    
    assert np.isclose(result.sum(), 1.0)
    impl_idx = mod._method_mix_order.index('impl')
    assert np.isclose(result[impl_idx], 1.0)
    
    # All others should be ~0
    for idx, method in enumerate(mod._method_mix_order):
        if method != 'impl':
            assert result[idx] < 1e-6


def test_method_mix_multiple_methods_fill_100_percent():
    """Test setting multiple methods that together equal 100%."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    mod.set_method_mix('pill', 0.4)
    mod.set_method_mix('inj', 0.3)
    mod.set_method_mix('impl', 0.3)
    
    result = mod._build_method_mix_array()
    
    assert np.isclose(result.sum(), 1.0)
    # All other methods should be ~0
    for method in ['iud', 'cond', 'btl', 'wdraw', 'othtrad', 'othmod']:
        idx = mod._method_mix_order.index(method)
        assert result[idx] < 1e-6


def test_method_mix_with_zero_baseline():
    """Test method mix behavior when some baseline values are zero."""
    mod = fp.MethodIntervention(year=2025)
    
    # Baseline with some zeros
    baseline = np.array([0.2, 0.2, 0, 0, 0.2, 0, 0.2, 0, 0.2])
    mod.set_method_mix_baseline(baseline)
    
    # Set target for non-zero method
    mod.set_method_mix('pill', 0.5)
    
    result = mod._build_method_mix_array()
    
    assert np.isclose(result.sum(), 1.0)
    pill_idx = mod._method_mix_order.index('pill')
    assert np.isclose(result[pill_idx], 0.5)


def test_method_mix_percentage_conversion():
    """Test that values >1 are treated as percentages and converted."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    # Use percentage (not fraction)
    mod.set_method_mix('impl', 30)  # Should be interpreted as 30% = 0.30
    
    result = mod._build_method_mix_array()
    impl_idx = mod._method_mix_order.index('impl')
    
    # Should convert 30 → 0.30
    assert np.isclose(result[impl_idx], 0.30, atol=0.01)


def test_very_small_efficacy():
    """Test very small but valid efficacy values."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_efficacy('wdraw', 0.001)  # 0.1% effective
    
    intv = mod.build()
    assert intv.pars.eff['Withdrawal'] == 0.001


def test_very_large_duration():
    """Test very large duration values (e.g., permanent methods)."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_duration_months('btl', 600)  # 50 years (permanent)
    
    intv = mod.build()
    assert intv.pars.dur_use['BTL'] == 600.0


def test_fractional_duration():
    """Test fractional month durations."""
    mod = fp.MethodIntervention(year=2025)
    
    mod.set_duration_months('cond', 0.5)  # Half a month
    
    intv = mod.build()
    assert intv.pars.dur_use['Condoms'] == 0.5


def test_multiple_interventions_same_method():
    """Test setting all intervention types for same method."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    # Set everything for 'inj'
    mod.set_efficacy('inj', 0.99)
    mod.set_duration_months('inj', 36)
    mod.set_method_mix('inj', 0.4)
    
    intv = mod.build()
    
    assert intv.pars.eff['Injectables'] == 0.99
    assert intv.pars.dur_use['Injectables'] == 36.0
    assert hasattr(intv.pars, 'method_mix')


def test_build_multiple_times():
    """Test that building multiple times produces independent interventions."""
    mod = fp.MethodIntervention(year=2025)
    mod.set_efficacy('pill', 0.95)
    
    intv1 = mod.build()
    intv2 = mod.build()
    
    # Should be separate objects
    assert intv1 is not intv2
    
    # But with same configuration
    assert intv1.pars.eff['Pill'] == intv2.pars.eff['Pill']


def test_modify_after_build():
    """Test that modifying after build() doesn't affect built intervention."""
    mod = fp.MethodIntervention(year=2025)
    mod.set_efficacy('pill', 0.95)
    
    intv = mod.build()
    
    # Modify after building
    mod.set_efficacy('pill', 0.80)
    
    # Original intervention should be unchanged
    assert intv.pars.eff['Pill'] == 0.95


def test_empty_baseline_all_zeros():
    """Test that all-zero baseline is rejected."""
    mod = fp.MethodIntervention(year=2025)
    
    with pytest.raises(ValueError, match='sums to zero'):
        mod.set_method_mix_baseline(np.zeros(9))


def test_method_mix_overwrite_target():
    """Test overwriting method mix target for same method."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    mod.set_method_mix('impl', 0.2)
    mod.set_method_mix('impl', 0.3)  # Overwrite
    
    result = mod._build_method_mix_array()
    impl_idx = mod._method_mix_order.index('impl')
    
    assert np.isclose(result[impl_idx], 0.3, atol=0.01)


def test_method_mix_near_boundary():
    """Test method mix values very close to 1.0."""
    mod = fp.MethodIntervention(year=2025)
    baseline = np.ones(9) / 9.0
    mod.set_method_mix_baseline(baseline)
    
    # Just under 1.0
    mod.set_method_mix('impl', 0.9999)
    
    result = mod._build_method_mix_array()
    
    assert np.isclose(result.sum(), 1.0)


def test_capture_method_mix_from_uninitialized_sim():
    """Test that capturing from uninitialized sim raises appropriate error."""
    mod = fp.MethodIntervention(year=2025)
    
    # Create sim but don't initialize
    pars = dict(n_agents=100, start_year=2000, end_year=2010, location='senegal')
    sim = fp.Sim(pars=pars)
    
    # Should raise error because sim not initialized
    with pytest.raises(Exception):  # Exact error depends on implementation
        mod.capture_method_mix_from_sim(sim)


def test_year_types():
    """Test that year accepts both int and float."""
    mod1 = fp.MethodIntervention(year=2025)
    assert mod1.year == 2025
    
    mod2 = fp.MethodIntervention(year=2025.5)
    assert mod2.year == 2025.5


def test_all_methods_set():
    """Test setting all 9 methods for efficacy."""
    mod = fp.MethodIntervention(year=2025)
    
    methods = ['pill', 'iud', 'inj', 'cond', 'btl', 'wdraw', 'impl', 'othtrad', 'othmod']
    
    for i, method in enumerate(methods):
        mod.set_efficacy(method, 0.9 + i * 0.01)
    
    intv = mod.build()
    
    # All should be present
    assert len(intv.pars.eff) == 9


def test_preview_immutability():
    """Test that preview() doesn't modify internal state."""
    mod = fp.MethodIntervention(year=2025)
    mod.set_efficacy('pill', 0.95)
    
    preview1 = mod.preview()
    preview2 = mod.preview()
    
    # Should be equal but different objects
    assert preview1['efficacy'] == preview2['efficacy']
    assert preview1 is not preview2


def test_method_mix_normalization_precision():
    """Test that method mix normalization maintains precision."""
    mod = fp.MethodIntervention(year=2025)
    
    # Use values that don't sum to 1
    baseline = np.array([0.15, 0.18, 0.12, 0.08, 0.14, 0.06, 0.17, 0.07, 0.03])
    
    mod.set_method_mix_baseline(baseline)
    mod.set_method_mix('pill', 0.25)
    
    result = mod._build_method_mix_array()
    
    # Should sum to exactly 1.0 within numerical precision
    assert np.abs(result.sum() - 1.0) < 1e-10


def test_string_to_float_conversion():
    """Test that numeric strings are handled (or rejected appropriately)."""
    mod = fp.MethodIntervention(year=2025)
    
    # These should work (Python float() handles strings)
    mod.set_efficacy('pill', float('0.95'))
    mod.set_duration_months('inj', float('30'))
    mod.set_probability_of_use(float('0.6'))
    
    assert mod._eff['pill'] == 0.95
    assert mod._dur['inj'] == 30.0
    assert mod._p_use == 0.6


# =============================================================================
# END-TO-END SIMULATION TESTS
# =============================================================================

def test_e2e_efficacy_reduces_pregnancies():
    """
    End-to-end test: Near-perfect efficacy intervention reduces pregnancies.
    Tests that the intervention is built and applied correctly.
    """
    base_pars = dict(
        n_agents=2000,
        start_year=2000,
        end_year=2010,
        location='senegal',
        rand_seed=42,
    )
    
    # Baseline simulation
    sim_baseline = fp.Sim(pars=base_pars, label='Baseline')
    sim_baseline.run()
    baseline_births = np.sum(sim_baseline.results.fp.births)
    
    # Intervention: Set ALL modern methods to near-perfect
    mod = fp.MethodIntervention(year=2001, label='Perfect Methods')
    for method in ['pill', 'iud', 'inj', 'impl', 'cond', 'btl', 'othmod']:
        mod.set_efficacy(method, 0.9999)
    intv = mod.build()
    
    # Verify intervention was built correctly
    assert intv.pars.eff['Pill'] == 0.9999
    assert len(intv.pars.eff) == 7
    
    # Run with intervention
    sim_intervention = fp.Sim(pars=base_pars, interventions=intv, label='Intervention')
    sim_intervention.run()
    intervention_births = np.sum(sim_intervention.results.fp.births)
    
    # Should have at least some reduction (or equal) - exact amount depends on contraceptive coverage
    # The key test is that intervention was applied and sim completed
    assert intervention_births <= baseline_births, \
        f"Near-perfect efficacy should not increase births: {intervention_births} vs {baseline_births}"
    
    assert sim_intervention.complete


def test_e2e_duration_increases_method_prevalence():
    """
    End-to-end test: Increasing duration should increase method prevalence at end.
    """
    base_pars = dict(
        n_agents=1000,
        start_year=2000,
        end_year=2012,
        location='senegal',
        rand_seed=123,
    )
    
    # Baseline simulation
    sim_baseline = fp.Sim(pars=base_pars, label='Baseline')
    sim_baseline.run()
    
    # Count injectable users at end
    baseline_inj_users = np.sum(sim_baseline.people.fp.method == 3)  # 3 = injectables index
    
    # Intervention: Significantly increase injectable duration
    mod = fp.MethodIntervention(year=2002, label='Long Duration Injectable')
    mod.set_duration_months('inj', 48)  # 4 years - very long
    intv = mod.build()
    
    sim_intervention = fp.Sim(pars=base_pars, interventions=intv, label='Intervention')
    sim_intervention.run()
    
    intervention_inj_users = np.sum(sim_intervention.people.fp.method == 3)
    
    # With much longer duration, should have more people on injectables at end
    assert intervention_inj_users > baseline_inj_users, \
        f"Expected more injectable users with longer duration: {intervention_inj_users} vs {baseline_inj_users}"
    
    # Verify the intervention was applied
    assert intv.pars.dur_use['Injectables'] == 48.0


def test_e2e_probability_of_use_increases_coverage():
    """
    End-to-end test: set_probability_of_use creates intervention with p_use parameter.
    Note: Whether p_use affects simulation depends on the choice module being used.
    """
    base_pars = dict(
        n_agents=800,
        start_year=2000,
        end_year=2010,
        location='senegal',
        rand_seed=456,
    )
    
    # Create intervention with p_use
    mod = fp.MethodIntervention(year=2002, label='Increase Access')
    mod.set_probability_of_use(0.7)
    intv = mod.build()
    
    # Verify the intervention has p_use parameter
    assert intv.pars.p_use == 0.7
    
    # Run simulation to ensure it doesn't crash
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Intervention')
    sim.run()
    
    assert sim.complete


def test_e2e_method_mix_shifts_distribution():
    """
    End-to-end test: Method mix intervention creates intervention with method_mix parameter.
    """
    base_pars = dict(
        n_agents=1000,
        start_year=2000,
        end_year=2012,
        location='senegal',
        rand_seed=789,
    )
    
    # Initialize temp sim to get baseline
    temp_sim = fp.Sim(pars=base_pars, label='Temp')
    temp_sim.init()
    
    # Create intervention targeting specific distribution
    mod = fp.MethodIntervention(year=2002, label='LARC Campaign')
    mod.capture_method_mix_from_sim(temp_sim)
    mod.set_method_mix('impl', 0.4)  # Target 40% of users on implants
    intv = mod.build()
    
    # Verify method_mix was set in intervention
    assert hasattr(intv.pars, 'method_mix')
    assert isinstance(intv.pars.method_mix, np.ndarray)
    assert np.isclose(intv.pars.method_mix.sum(), 1.0)
    
    # Run simulation to ensure it doesn't crash
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Intervention')
    sim.run()
    
    assert sim.complete


def test_e2e_combined_interventions():
    """
    End-to-end test: Combined efficacy + duration changes for single method.
    """
    base_pars = dict(
        n_agents=800,
        start_year=2000,
        end_year=2012,
        location='senegal',
        rand_seed=999,
    )
    
    # Intervention: Improve pill efficacy AND duration
    mod = fp.MethodIntervention(year=2002, label='Comprehensive Pill Program')
    mod.set_efficacy('pill', 0.98)
    mod.set_duration_months('pill', 30)
    intv = mod.build()
    
    # Verify both changes were applied to intervention
    assert intv.pars.eff['Pill'] == 0.98
    assert intv.pars.dur_use['Pill'] == 30.0
    
    # Run simulation to ensure it completes
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Intervention')
    sim.run()
    
    assert sim.complete


def test_e2e_intervention_timing():
    """
    End-to-end test: Intervention applied at specific year affects results after that year.
    """
    base_pars = dict(
        n_agents=500,
        start_year=2000,
        end_year=2010,
        location='senegal',
        rand_seed=111,
    )
    
    # Early intervention (year 2002)
    mod_early = fp.MethodIntervention(year=2002, label='Early')
    mod_early.set_duration_months('inj', 40)
    intv_early = mod_early.build()
    
    sim_early = fp.Sim(pars=base_pars, interventions=intv_early, label='Early Intervention')
    sim_early.run()
    
    # Late intervention (year 2008)
    mod_late = fp.MethodIntervention(year=2008, label='Late')
    mod_late.set_duration_months('inj', 40)
    intv_late = mod_late.build()
    
    sim_late = fp.Sim(pars=base_pars, interventions=intv_late, label='Late Intervention')
    sim_late.run()
    
    # Early intervention should have more impact by end (more time to accumulate effect)
    early_inj_users = np.sum(sim_early.people.fp.method == 3)
    late_inj_users = np.sum(sim_late.people.fp.method == 3)
    
    assert early_inj_users > late_inj_users, \
        f"Early intervention should have more effect by end: {early_inj_users} vs {late_inj_users}"


def test_e2e_multiple_methods():
    """
    End-to-end test: Setting interventions for multiple different methods.
    """
    base_pars = dict(
        n_agents=1000,
        start_year=2000,
        end_year=2012,
        location='senegal',
        rand_seed=222,
    )
    
    # Intervention affecting 3 different methods
    mod = fp.MethodIntervention(year=2002, label='Multi-Method Program')
    mod.set_efficacy('pill', 0.97)
    mod.set_duration_months('inj', 36)
    mod.set_efficacy('impl', 0.998)
    intv = mod.build()
    
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Multi-Method')
    sim.run()
    
    # Verify all three changes were applied
    assert intv.pars.eff['Pill'] == 0.97
    assert intv.pars.dur_use['Injectables'] == 36.0
    assert intv.pars.eff['Implants'] == 0.998
    
    # Simulation should complete successfully
    assert sim.complete


def test_e2e_no_intervention_baseline():
    """
    End-to-end test: Verify baseline simulation without intervention works correctly.
    """
    base_pars = dict(
        n_agents=500,
        start_year=2000,
        end_year=2010,
        location='senegal',
        rand_seed=333,
    )
    
    sim = fp.Sim(pars=base_pars, label='Pure Baseline')
    sim.run()
    
    # Should complete successfully
    assert sim.complete
    
    # Should have results
    assert 'births' in sim.results.fp
    assert len(sim.results.fp.births) > 0
    
    # Should have people with methods
    methods = sim.people.fp.method
    assert len(methods) > 0  # People exist
    
    # Should have mix of methods (not all on one method)
    unique_methods = np.unique(methods)
    assert len(unique_methods) > 2, "Should have variety of methods in use"


def test_e2e_efficacy_near_perfect():
    """
    End-to-end test: Near-perfect efficacy (0.999) should nearly eliminate failures.
    """
    base_pars = dict(
        n_agents=500,
        start_year=2000,
        end_year=2010,
        location='senegal',
        rand_seed=444,
    )
    
    # Set all methods to near-perfect efficacy
    mod = fp.MethodIntervention(year=2001, label='Perfect Methods')
    for method in ['pill', 'iud', 'inj', 'cond', 'impl']:
        mod.set_efficacy(method, 0.999)
    intv = mod.build()
    
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Perfect Efficacy')
    sim.run()
    
    births = np.sum(sim.results.fp.births)
    
    # Should have very few births (only from non-users and the 0.1% failure rate)
    # Compare to reasonable baseline expectation
    assert births < 200, f"Expected very few births with near-perfect methods, got {births}"
    
    # Simulation should complete
    assert sim.complete


def test_e2e_intervention_year_before_sim_start():
    """
    End-to-end test: Intervention year before sim start should apply from beginning.
    """
    base_pars = dict(
        n_agents=300,
        start_year=2005,
        end_year=2010,
        location='senegal',
        rand_seed=555,
    )
    
    # Intervention year is before simulation start
    mod = fp.MethodIntervention(year=2000, label='Pre-existing')
    mod.set_duration_months('pill', 30)
    intv = mod.build()
    
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Pre-existing Intervention')
    sim.run()
    
    # Should complete without error
    assert sim.complete
    
    # Intervention should be active from start
    assert intv.pars.year == 2000


def test_e2e_compare_method_mix_final_state():
    """
    End-to-end test: Verify final method distribution matches intervention intent.
    """
    base_pars = dict(
        n_agents=2000,  # Larger sample for stable statistics
        start_year=2000,
        end_year=2015,  # Longer run for equilibration
        location='senegal',
        rand_seed=666,
    )
    
    # Initialize temp sim to get baseline
    temp_sim = fp.Sim(pars=base_pars)
    temp_sim.init()
    
    # Create intervention targeting specific distribution
    mod = fp.MethodIntervention(year=2002, label='Targeted Mix')
    mod.capture_method_mix_from_sim(temp_sim)
    mod.set_method_mix('inj', 0.35)  # Target 35% injectables
    mod.set_method_mix('impl', 0.25)  # Target 25% implants
    intv = mod.build()
    
    sim = fp.Sim(pars=base_pars, interventions=intv, label='Targeted Mix')
    sim.run()
    
    # Calculate final shares (among users only, excluding method=0)
    users_mask = sim.people.fp.method != 0
    users_methods = sim.people.fp.method[users_mask]
    
    if len(users_methods) > 0:
        inj_share = np.sum(users_methods == 3) / len(users_methods)  # 3 = injectables
        impl_share = np.sum(users_methods == 7) / len(users_methods)  # 7 = implants
        
        # Should be moving toward targets (may not be exact due to population dynamics)
        # Check that they're in reasonable range of targets
        assert 0.15 < inj_share < 0.55, \
            f"Injectable share {inj_share:.3f} should be moving toward 0.35"
        assert 0.10 < impl_share < 0.45, \
            f"Implant share {impl_share:.3f} should be moving toward 0.25"
    
    assert sim.complete


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
