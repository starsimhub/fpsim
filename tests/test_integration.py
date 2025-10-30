"""
Test different components of the simulation behave as expected
"""

# Imports
import sciris as sc
import fpsim as fp
from fpsim import defaults as fpd
from fpsim import methods as fpm
from fpsim import interventions as fpi

import numpy as np
import pylab as pl
import starsim as ss

parallel = 0  # For testing purposes

f24_age_pyramid = np.ndarray(shape=(3, 3), dtype=float)
f24_age_pyramid[0, :] = [0, 0, 0]
f24_age_pyramid[1, :] = [24, 0, 100]
f24_age_pyramid[2, :] = [24.01, 0, 0]

f15_age_pyramid = np.ndarray(shape=(3, 3), dtype=float)
f15_age_pyramid[0, :] = [0, 0, 0]
f15_age_pyramid[1, :] = [15, 0, 100]
f15_age_pyramid[2, :] = [15.01, 0, 0]

f_age_pyramid = np.ndarray(shape=(2,3), dtype=float)
f_age_pyramid[0,:] = [0, 100, 100]
f_age_pyramid[0,:] = [75, 100, 100]


def test_pregnant_women():
    sc.heading('Test pregnancy and birth outcomes... ')

    # start out 24f, no contra, no pregnancy, active, fertile, has_intent
    methods = ss.ndict([
        fpm.Method(name='none', efficacy=0, modern=False, dur_use=fpm.ln(2, 3), label='None'),
        fpm.Method(name='test', efficacy=0.0, modern=True,  dur_use=fpm.ln(2, 3), label='Test'),
    ])
    contra_mod = fpm.RandomChoice(methods=methods)

    # force all to have debuted at age 20
    debut_age = {}
    debut_age['ages']=np.arange(10, 45, dtype=float)
    debut_age['probs']=np.zeros(35, dtype=float)
    debut_age['probs'][10] = 1

    # force all women to have the same fecundity and be sexually active
    # Note: not all agents will be active at t==0 but will be after t==1
    sexual_activity = np.zeros(51, dtype=float)
    sexual_activity[20:30] = 1

    custom_pars = {
        'start_year': 2000,
        'end_year': 2001,
        'n_agents': 1000,
        'test': True,
        'age_pyramid': f24_age_pyramid,
        'debut_age': debut_age,
        'primary_infertility': 0,
        'sexual_activity': sexual_activity,
    }

    sim = fp.Sim(pars=custom_pars, contraception_module=contra_mod)
    sim.init()

    # Override fecundity to maximize pregnancies and minimize variation during test
    sim.people.fp.personal_fecundity[:] = 1

    sim.run()

    n_agents = sim.pars['n_agents']
    fecundity24 = sim.pars.fp['age_fecundity'][24]

    # all women are 24 and can become pregnant, so we can look at percentages to estimate the
    # expected number of pregnancies, stillborns, living infants, and miscarriages/abortions
    # pregnancy rate: 11.2% per month or approx 80% per year
    print(f'Checking pregnancy and birth outcomes from {n_agents} women... ')
    assert (fecundity24*1.1) * n_agents > sim.results.fp.pregnancies[0:12].sum() > (fecundity24*0.9) * n_agents, "Expected number of pregnancies not met"
    print(f'✓ ({sim.results.fp.pregnancies[0:12].sum()} pregnancies, as expected)')

    # stillbirth rate: 2.5% of all pregnancies, but only count completed pregnancies
    assert 0.06 * sim.results.fp.pregnancies[0:3].sum() > sim.results.fp.stillbirths[9:12].sum() > 0.01 * sim.results.fp.pregnancies[0:3].sum() , "Expected number of stillbirths not met"
    print(f'✓ ({sim.results.fp.stillbirths[9:12].sum()} stillbirths, as expected)')

    # miscarriage rate: 10% of all pregnancies. Miscarriages are calculated at end of 1st trimester, so pregnancies from months 0-9
    # have miscarriages from months 3-12.
    pregnancies = sim.results.fp.pregnancies[0:9].sum()
    miscarriages = sim.results.fp.miscarriages[3:12].sum()
    assert 0.15 * pregnancies > miscarriages > 0.05 * pregnancies, "Expected number of miscarriages not met"
    print(f'✓ ({miscarriages} miscarriages, as expected)')

    # abortion rate: 8% of all pregnancies
    # abortions occur at same timestep as conception, so all pregnancies were checked for abortions
    pregnancies = sim.results.fp.pregnancies.sum()
    abortions = sim.results.fp.abortions.sum()
    assert 0.13 * pregnancies > abortions > 0.03 * pregnancies, "Expected number of abortions not met"
    print(f'✓ ({abortions} abortions, as expected)')

    # live birth rate: 79.5% of all pregnancies
    # no premature births so all births are full term
    pregnancies = sim.results.fp.pregnancies[0:3].sum()
    live_births = sim.results.fp.births[9:12].sum()
    assert 0.85 * pregnancies > live_births > 0.75 * pregnancies, "Expected number of live births not met"
    print(f'✓ ({live_births} live births from {pregnancies} pregnancies, as expected)')

    return sim


def test_contraception():
    sc.heading('Test contraception prevents pregnancy... ')

    # start out 24f, on contra, no starting pregnancy, active, fertile, high fecundity.
    # test 1: Assume contraception 100% effective, 100% uptake -> no pregnancies.
    # test 2: contraception use restarts after postpartum period

    methods = ss.ndict([
        fpm.Method(name='none', efficacy=0, modern=False, dur_use=1/12, label='None'),
        fpm.Method(name='test', efficacy=1.0, modern=True,  dur_use=1/12, label='Test'),
    ])
    contra_mod = fpm.RandomChoice(methods=methods, pars={'p_use': 1.0})

    # force all to have debuted at age 20
    debut_age = {}
    debut_age['ages']=np.arange(10, 45, dtype=float)
    debut_age['probs']=np.zeros(35, dtype=float)
    debut_age['probs'][10] = 1.0

    sexual_activity = np.zeros(51, dtype=float)
    sexual_activity[20:30] = 1.0

    # after 12 months, stop using any contraception, so we should see pregnancies after the switch
    p_use_change = fpi.update_methods(year=2001, p_use=0.0, name="stopcontra", label="stop contraception")

    # after 24 months, we should be seeing pregnancies again so we can reenable contraception use rates and check postpartum uptake
    p_use_change2 = fpi.update_methods(year=2002, p_use=0.5, name="restartcontra", label="restart contraception")

    custom_pars = {
        'start_year': 2000,
        'end_year': 2005,
        'n_agents': 1000,
        'location': 'senegal',
        'age_pyramid': f24_age_pyramid,
        'debut_age': debut_age,
        'primary_infertility': 0,
        'sexual_activity': sexual_activity,
    }

    sim = fp.Sim(pars=custom_pars, contraception_module=contra_mod, interventions=[p_use_change, p_use_change2])
    sim.init()
    # Override fecundity to maximize pregnancies and minimize variation during test
    sim.people.fp.personal_fecundity[:] = 1

    sim.run(verbose=1/12)

    print(f'Checking pregnancy and birth outcomes from {sim.pars.n_agents} women... ')
    assert sim.results.fp.pregnancies[0:12].sum() == 0, "Expected no pregnancies"
    assert sim.results.fp.pregnancies[12:].sum() > 0, "Expected pregnancies after contraception switch"
    print(f'✓ (no pregnancies with 100% effective contraception)')

    pp1 = (sim.people.fp.ti_delivery == sim.t.ti).uids
    assert sim.people.fp.on_contra[pp1].sum() == 0, "Expected no contraception use immediately postpartum"
    print(f'✓ (no contraception use postpartum)')
    assert (sim.people.fp.on_contra == True).sum() < sim.pars['n_agents'], "Expected some agents to be off of birth control at any given time"
    print(f'✓ (contraception use rate {sim.people.fp.on_contra.sum()/len(sim.people):.2f}, as expected)')

    return sim


def test_method_selection_dependencies():
    sc.heading('Testing method selection dependencies... ')

    # set up a bunch of identical women, same age, same history
    # manually assign a previous method and short method dur
    # inspect new methods -> should be distributed according to the method mix
    pars = {
        'start_year': 2000,
        'end_year': 2001,
        'n_agents': 1000,
        'primary_infertility': 1,  # make sure no pregnancies!
    }

    cm_pars = dict(
        prob_use_trend_par=0.0,  # no change in use trend, so changes should be driven by other factors
        force_choose=False,
    )
    method = fpm.SimpleChoice(pars=cm_pars, location="kenya")

    cpr = fp.cpr_by_age()
    snapshots = fp.snapshot([1,2])

    # force all to have debuted and sexually active
    debut_age = {
        'ages': np.arange(10, 45, dtype=float),
        'probs': np.ones(35, dtype=float)/35
    }

    pars['debut_age'] = debut_age

    # Note: not all agents will be active at t==0 but will be after t==1
    sexual_activity = np.ones(51, dtype=float)
    pars['sexual_activity'] = sexual_activity

    pars['age_pyramid'] = f15_age_pyramid
    sim1 = fp.Sim(location="kenya", pars=pars, contraception_module=method, analyzers=[cpr, snapshots])
    sim1.init()
    sim1.people.fp.ever_used_contra[:] = True

    # custom init forces age, sex and other person defaults
    # p_use by age: <18: ~.18, 18-20: ~.49, 20-25: ~.54, 25-35: ~.52, 35-50: ~.32
    sim1.run()

    contra_ending_t2 = sim1['analyzers'][1].snapshots['1'].fp.ti_contra == 2
    ending_contra_type = sim1['analyzers'][1].snapshots['1'].fp.method[contra_ending_t2]
    new_contra_type = sim1['analyzers'][1].snapshots['2'].fp.method[contra_ending_t2]

    # Compare methods. All should have changed.
    print(f"Checking agents change methods at ti_contra ... ")
    assert np.all((ending_contra_type != new_contra_type) | ((ending_contra_type == 0) & (new_contra_type == 0))), "expected all agents at ti_contra to change contra method or remain on None"
    print(f'✓ (all agents change methods)')

    return sim1


def test_education_preg():
    sc.heading('Testing that lower fertility rate leads to more education...')

    def make_sim(pregnant=False):
        pars = dict(start=2000, stop=2010, n_agents=1000, test=True)
        sim = fp.Sim(pars=pars)
        sim.init()
        sim.people.age[:] = 15
        sim.people.female[:] = True
        fpppl = sim.people.fp
        if pregnant:
            fpppl.gestation[:] = 1  # Start the counter at 1
            fpppl.dur_pregnancy[:] = 9  # Set pregnancy duration
            fpppl.ti_delivery[:] = 9  # Set time of delivery
            fpppl.ti_pregnant[:] = 0
            fpppl.pregnant[:] = True
            fpppl.method[:] = 0
            fpppl.on_contra[:] = False
            fpppl.ti_contra[:] = 12
        return sim

    sim_base = make_sim()
    sim_preg = make_sim(pregnant=True)
    m = ss.parallel([sim_base, sim_preg], parallel=parallel)
    sim_base, sim_preg = m.sims[:]  # Replace with run versions

    # Check that education has increased
    base_edu = sim_base.results.edu.mean_attainment[-1]
    preg_edu = sim_preg.results.edu.mean_attainment[-1]
    base_births = sum(sim_base.results.fp.births)
    preg_births = sum(sim_preg.results.fp.births)
    assert base_births < preg_births, f'With more pregnancy there should be more births, but {preg_births}<{base_births}'
    assert preg_edu < base_edu, f'With more pregnancy there should be lower education levels, but {preg_edu}>{base_edu}'
    print(f"✓ (Higher teen pregnancy ({preg_births:.0f} vs {base_births:.0f}) -> less education ({preg_edu:.2f} < {base_edu:.2f}))")

    return sim_base, sim_preg


def plot_results(sim):
    # Plots
    fig, axes = pl.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    cind = 0

    # mCPR
    ax = axes[0]
    ax.plot(sim.results.timevec, sim.results.contraception.cpr)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # mCPR by age
    ax = axes[1]
    for alabel in fpd.method_age_map.keys():
        ares = sim.results.cpr_by_age[alabel]
        ax.plot(sim.results.timevec, ares, label=alabel, color=colors[cind])
        cind += 1
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # Plot method mix
    ax = axes[2]
    oc_uids = sim.people.on_contra.uids
    method_props = [len(oc_uids[sim.people.fp.method[oc_uids] == i]) / len(oc_uids) for i in range(1, 10)]
    method_labels = [m.name for m in sim.people.contraception_module.methods.values() if m.label != 'None']
    ax.bar(method_labels, method_props)
    ax.set_ylabel('Proportion among all users')
    ax.set_title('Contraceptive mix')

    sc.figlayout()
    pl.show()


if __name__ == '__main__':

    sc.options(interactive=False)
    s1 = test_pregnant_women()
    s2 = test_contraception()
    s6 = test_method_selection_dependencies()
    s7, s8 = test_education_preg()
    print("All tests passed!")


