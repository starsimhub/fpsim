"""
Run tests on the Scenarios class.
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
import pytest
from fpsim import defaults as fpd

# Global settings
int_year = 2002 # Year to start the interventions
parallel   = 1 # Whether to run in serial (for debugging)
do_plot  = 0 # Whether to do plotting in interactive mode
do_plot_as = 0 # Whether or not to plot all age-specific channels
default_ages = list(fpd.method_age_map.keys())
sc.options(backend='agg') # Turn off interactive plots


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def make_sims(interventions, contraception_module=None):
    ''' Make simulations with particular interventions '''
    simlist = []
    for intv in interventions:
        pars = dict(location='test')
        new_sim = fp.Sim(pars=pars, interventions=intv, contraception_module=contraception_module)
        simlist.append(new_sim)
    return simlist


def test_update_methods_eff():

    sc.heading('Testing updating method efficacy...')

    method = 'othmod'
    method_label = 'Other modern'
    low_eff = 0.70
    high_eff = 0.95

    # Make interventions that update the efficacies
    um1 = fp.update_methods(year=int_year, eff={method_label:low_eff})
    um2 = fp.update_methods(year=int_year, eff={method_label:high_eff})

    simlist = make_sims([um1, um2])
    msim = ss.MultiSim(sims=simlist)
    msim.run(parallel=parallel)

    low_eff_post_sim = msim.sims[0].connectors.contraception.methods[method].efficacy
    high_eff_post_sim = msim.sims[1].connectors.contraception.methods[method].efficacy

    msg = f"Efficacies did not update correctly"
    assert (high_eff_post_sim==high_eff) & (low_eff_post_sim==low_eff) & (high_eff_post_sim > low_eff_post_sim), msg
    ok(f'High efficacy: {high_eff}, low efficacy: {low_eff}')
    return msim


def test_update_methods():
    """
    Checks that fp.update_methods() function properly updates
    """

    sc.heading('Testing updating method properties...')

    # Make new durations representing longer-lasting IUDs, injectables, and implants
    new_durs = {
        'IUDs': 10,
        'Implants': 10,
    }
    p_use = 0.99

    # Change the method mix
    method_mix = np.array([0.05, 0.3, 0.3, 0.05, 0, 0, 0.3, 0, 0])

    # Make interventions
    no_contra = fp.update_methods(2000, p_use=0.0)
    hi_contr = fp.update_methods(int_year, p_use=p_use, dur_use=new_durs, method_mix=method_mix)

    # Make and run sims
    simlist = make_sims([no_contra, hi_contr], contraception_module=fp.RandomChoice())
    msim = ss.MultiSim(sims=simlist)
    msim.run(parallel=parallel)

    # Test that all the parameters were correctly updated
    assert msim.sims[1].connectors.contraception.pars['p_use'].pars.p == p_use
    assert msim.sims[1].connectors.contraception.methods['iud'].dur_use == 10
    assert np.array_equal(msim.sims[1].connectors.contraception.pars['method_mix'], method_mix)
    ok('Parameters updated correctly')

    # Test that there are fewer births with the new method parameters
    baseline_births = msim.sims[0].results.fp.births.sum()
    scenario_births = msim.sims[1].results.fp.births.sum()

    msg = f'Expected more births with default methods but ({scenario_births} > {baseline_births})'
    assert baseline_births > scenario_births, msg
    ok(f'Changes to method parameters resulted in fewer births, as expected ({scenario_births} < {baseline_births})')

    return msim


# def test_scenarios():
#     def run_scenario(scen, plot=do_plot, plot_as=do_plot_as):
#         '''Runs simple scenario and returns Scenarios object'''
#         scens = fp.Scenarios(location='test', scens=scen, start=int_year)
#         scens.run(parallel=parallel)
#         if plot:
#             scens.plot()
#             scens.plot(to_plot='method')
#             scens.plot_sims()
#         if plot_as:
#             for plot_type in ['age_specific_tfr', 'age_specific_pregnancies', 'age_specific_imr', 'age_specific_mmr', 'age_specific_stillbirths', 'age_specific_births']:
#                 scens.plot(to_plot=plot_type)
#         return scens
#
#     output = {} # dictionary of scenarios output
#
#     base_scenario = run_scenario(fp.make_scen(label="Baseline"))
#     output['Baseline'] = base_scenario
#
#     '''Tests that Scenarios repeats sims corresponding to scenarios added as expected'''
#     high_inj_eff = 0.99
#     scen = fp.make_scen(label='More effective pill', year=int_year, eff={'Pill': high_inj_eff})
#     scens_repeat = fp.Scenarios(location='test', repeats=2, scens=scen, start_year=int_year)
#     scens_repeat.run(parallel=parallel)
#     assert len(scens_repeat.msim.sims) == 2, f"Should be {2} sims in scens object but found {len(scens_repeat.msim.sims)}"
#     ok('Scenarios repeated as expected')
#
#     eff1 = scens_repeat.msim.sims[0].connectors.contraception.methods['pill'].efficacy
#     eff2 = scens_repeat.msim.sims[1].connectors.contraception.methods['pill'].efficacy
#     output['Repeated'] = scens_repeat
#
#     for efficacy in [eff1, eff2]:
#         assert efficacy == high_inj_eff, f"Repeated efficacy scenarios do not match"
#         ok(f'Efficacy of pill is {efficacy}')
#
#     '''Checks that inputting scenarios as list has same result as adding them separately'''
#     low_inj_eff = 0.9
#     scen1 = fp.make_scen(label='More effective pill', year=int_year, eff={'Pill':high_inj_eff})
#     scen2 = fp.make_scen(label='Less effective pill', year=int_year, eff={'Pill':low_inj_eff})
#
#     scens_scenario_list = run_scenario([scen1, scen2])
#
#     eff1 = scens_scenario_list.msim.sims[0].connectors.contraception.methods['pill'].efficacy
#     eff2 = scens_scenario_list.msim.sims[1].connectors.contraception.methods['pill'].efficacy
#
#     output['Scenario_list'] = scens_scenario_list
#
#     assert eff1 == high_inj_eff, f"Efficacy of pill using scenarios list should be {high_inj_eff}, not {eff1}"
#     assert eff2 == low_inj_eff, f"Efficacy of pill using scenarios list should be {low_inj_eff}, not {eff2}"
#     ok(f'First scenario list efficacy is {high_inj_eff}, second scenario list efficacy is {low_inj_eff}')
#
#     '''Checks that we can't add invalid scenarios'''
#     invalid_scen1 = dict(invalid_key='Should fail')
#     invalid_scen2 = dict(dur_use=dict(invalid_key='Also should fail'))
#
#     with pytest.raises(TypeError):
#         invalid_scens1 = fp.Scenarios(location='test')
#         invalid_scens1.add_scen(invalid_scen1)
#
#     with pytest.raises(ValueError):
#         invalid_scens = fp.Scenarios(location='test')
#         invalid_scens.add_scen(invalid_scen2)
#         invalid_scens.run()
#
#     '''Checks that Scenarios.results against sim.results for each component sim'''
#     scenario_df = base_scenario.results.df
#     sim_results = base_scenario.msim.sims[0].results
#
#     def compare_results(scenario_key, sim_key, is_sum=True):
#         scenario_val = scenario_df[scenario_key][0]
#         if is_sum:
#             sim_val = sum(sim_results[sim_key])
#         else:
#             sim_val = np.mean(sim_results[sim_key])
#
#         assert scenario_val == sim_val, f"From sim results {sim_key} is {sim_val} while in scenarios {scenario_key} is {scenario_val}"
#         ok(f"Results for {scenario_key} and {sim_key} match")
#
#     # check sums
#     for keys in [("births", "births"), ("fails", "method_failures_over_year"), ("popsize", "pop_size"),
#                     ("infant_deaths", "infant_deaths_over_year"), ("maternal_deaths", "maternal_deaths_over_year")]:
#         compare_results(keys[0], keys[1])
#
#     # check rates
#     for keys in [("tfr", "tfr_rates"), ("mcpr", "mcpr")]:
#         compare_results(keys[0], keys[1], is_sum=False)


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        msim1  = test_update_methods_eff()
        msim2  = test_update_methods()
        # scenarios = test_scenarios() # returns a dict with schema {name: Scenarios}
