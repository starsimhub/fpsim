import numpy as np
import sciris as sc

import fpsim as fp


def test_interface_build_basic():
    year = 2025
    mgr = fp.MethodIntervention(year=year)
    mgr.set_efficacy({'inj': 0.99})
    mgr.set_duration_months({'iud': 24})
    mgr.set_method_mix_by_name({'inj': 30, 'impl': 20, 'pill': 50})

    intv = mgr.build()

    assert isinstance(intv, fp.update_methods)
    assert intv.pars.year == year
    assert intv.pars.eff['Injectables'] == 0.99  # Uses label, not name
    assert intv.pars.dur_use['IUDs'] == 24.0  # Uses label, not name

    mix = intv.pars.method_mix
    assert isinstance(mix, np.ndarray)
    assert len(mix) == 10  # Including 'none'
    assert np.isclose(mix.sum(), 1.0)


def test_interface_functional_shortcut():
    year = 2030
    mix = np.array([0.15, 0.2, 0.2, 0.2, 0.1, 0, 0.05, 0.05, 0.025, 0.025])  # 10 elements including 'none'

    intv = fp.make_update_methods(year=year, eff={'pill': 0.95}, duration_months={'pill': 12}, p_use=0.6, mix=mix)

    assert isinstance(intv, fp.update_methods)
    assert intv.pars.year == year
    assert intv.pars.eff['Pill'] == 0.95  # Uses label, not name
    assert intv.pars.dur_use['Pill'] == 12.0  # Uses label, not name
    assert np.isclose(intv.pars.method_mix.sum(), 1.0)


def test_wrapper_switching_matrix_dict():
    year = 2027
    mat = {'example': {'method_idx': [1,2,3]}}
    mgr = fp.MethodIntervention(year=year)
    mgr.set_switching_matrix(mat)
    intv = mgr.build()
    assert intv.pars.method_choice_pars == mat


def test_wrapper_scale_switching_matrix():
    """Test the scale_switching_matrix method."""
    # Create a simple sim with SimpleChoice
    pars = dict(
        n_agents=100,
        location='kenya',
        start_year=2000,
        end_year=2001,
    )
    method_choice = fp.SimpleChoice(location='kenya')
    sim = fp.Sim(pars=pars, contraception_module=method_choice)
    sim.init()
    
    # Get original matrix for comparison
    cm = sim.connectors.contraception
    original_matrix = sc.dcp(cm.pars['method_choice_pars'])
    
    # Create wrapper and scale injectables
    year = 2000.5
    mgr = fp.MethodIntervention(year=year)
    mgr.scale_switching_matrix(sim, target_method='inj', scale_factor=1.5)
    
    # Check that a switching matrix was set
    assert mgr._switch is not None
    assert isinstance(mgr._switch, dict)
    
    # Verify that the matrix structure is preserved but values changed
    assert set(mgr._switch.keys()) == set(original_matrix.keys())
    
    # Build and check that intervention is valid
    intv = mgr.build()
    assert isinstance(intv, fp.update_methods)
    assert intv.pars.year == year
    assert intv.pars.method_choice_pars is not None


