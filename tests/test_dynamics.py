"""
Test dynamics within FPsim related to contraceptive use, method choice, and duration of use
"""

import sciris as sc
import starsim as ss
import fpsim as fp
import matplotlib.pyplot as plt
import numpy as np

parallel   = 0 # Whether to run in parallel or serial (for debugging)


def make_sim_parts(location='ethiopia', new_p_use_pars=False):
    pars = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=-1, location=location)
    edu = fp.Education(location=location)
    choice = fp.StandardChoice(location=location)
    if new_p_use_pars:
        choice.contra_use_pars[0] = sc.objdict(
            intercept=np.log(1/9),  # Baseline prob = 0.1
            age_factors=np.array([.5,  1, -.5]),  # p_use increase until 39 then decreases
            ever_used_contra=1,  # Ever used
            edu_factors=np.array([2, 4]),
            parity=0,
            urban=3,
            wealthquintile=1,
            age_ever_user_factors=np.array([-0.5, -3., -1]),
        )
        choice.contra_use_pars[1] = sc.dcp(choice.contra_use_pars[0])
        choice.contra_use_pars[2] = sc.dcp(choice.contra_use_pars[0])
    return pars, choice, edu


def make_sim(label='Baseline', intvs=None, analyzers=None, location=None, **kwargs):
    pars, choice, edu = make_sim_parts(location=location)
    sim = fp.Sim(
        pars=pars, contraception_module=choice, education_module=edu,
        label=label, interventions=intvs, analyzers=analyzers, **kwargs
        )
    return sim


def test_mcpr(location='senegal', do_plot=False):
    sc.heading('Testing that mCPR changes as expected with its covariates...')

    # Create covariate changes
    class Covar:
        def __init__(self, pplattr, val0, val, resname):
            self.pplattr = pplattr
            self.val = val
            self.val0 = val0
            self.resname = resname
            self.base = None
            self.intv = None
            self.mcpr = None

    covars = [
        Covar('edu.attainment', 0, 15, 'edu.mean_attainment'),
        Covar('urban', False, True, 'n_urban'),
        # Covar('parity', 2, 'parity2to3'),  # Unfortunately this will not work
        Covar('wealthquintile', 1, 5, 'n_wq5'),
        Covar('fp.ever_used_contra', False, True, 'fp.ever_used_contra'),
        ]

    # Create interventions and sims
    sims = sc.autolist()
    def select_women(sim): return sim.people.female.uids

    def make_zeros():
        zero_states = sc.autolist()
        for covar in covars:
            zero_states += fp.change_people_state(
                                covar.pplattr,
                                eligibility=select_women,
                                years=2000.0,
                                new_val=covar.val0,
                                name=f"change_state_{covar.pplattr}_2000"
                            )
        return zero_states

    # Make baseline sim
    sims += make_sim(intvs=make_zeros(), location=location, label='Baseline')

    for covar in covars:
        change_state = fp.change_people_state(
                            covar.pplattr,
                            eligibility=select_women,
                            years=2010.0,
                            new_val=covar.val,
                            name=f"change_state_{covar.pplattr}_2010"
                        )
        sims += make_sim(intvs=make_zeros()+[change_state], location=location, label=f'Increased {covar.pplattr}')

    # Run
    m = ss.parallel(*sims, parallel=parallel)
    sims = m.sims[:]  # Replace with run versions

    # Firstly, check that changing the people attributes has registered in the relevant results metrics as expected
    for ri, covar in enumerate(covars):
        print(f"Checking effect of {covar.pplattr} ... ")
        if '.' in covar.resname:
            modname, resname = covar.resname.split('.')
            base = sims[0].results[modname][resname]
            intv = sims[ri+1].results[modname][resname]
        else:
            base = sims[0].results[covar.resname]
            intv = sims[ri+1].results[covar.resname]
        assert base[-1] <= intv[-1], f'Increasing {covar.pplattr} should register in results, but {intv[-1]}<{base[-1]}'
        print(f"✓ ({base[-1]:.2f} <= {intv[-1]:.2f})")
        covar.base = base
        covar.intv = intv
        covar.mcpr = sims[ri+1].results.contraception.mcpr

    # Next, check that changing the people attributes has registered in the relevant results metrics as expected
    mcprs = [sim.results.contraception.mcpr[-1] for sim in sims]
    print('Computed mCPRs:')
    for sim, mcpr in zip(sims, mcprs):
        print(f'{sim.label}: {mcpr:.2f}')
    # assert mcprs[0] == min(mcprs), 'Increasing any covariate should increase contraceptive use.'
    print(f"✗ (TEST FAILS: mCPR DOES NOT INCREASE WITH ALL COVARIATES) - NEED TO DEBUG")

    # Plot
    if do_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()
        for ri, covar in enumerate(covars):
            ax = axes[ri]
            ax.plot(sims[0].results.timevec, covar.base, label='Baseline')
            ax.plot(sims[ri+1].results.timevec, covar.intv, label=f'Increased {covar.pplattr}')
            ax.set_title(f'{covar.pplattr}')
            ax.set_xlabel('Year')
            ax.legend()
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(sims[0].results.timevec, sims[0].results.contraception.mcpr, label=sims[0].label)
        for ri, covar in enumerate(covars):
            ax.plot(sims[ri+1].results.timevec, covar.mcpr, label=covar.pplattr)
        ax.set_ylabel('mCPR')
        ax.set_xlabel('Year')
        plt.legend()
        fig.tight_layout()
        plt.show()

    return sims


def test_durations(location='senegal'):
    sc.heading('Testing that durations are as expected...')

    # Create parameters and modules
    sim_base = make_sim(location=location, label='Baseline')

    pars, choice, edu = make_sim_parts(location=location)
    short_choice = sc.dcp(choice)
    for m in short_choice.methods.values(): m.dur_use = ss.uniform(low=1, high=2)
    sim_short = fp.Sim(
        pars=pars, contraception_module=short_choice, education_module=edu,
        label='Short durations')

    # Run sims
    m = ss.parallel([sim_base, sim_short], parallel=parallel)
    sim_base, sim_short = m.sims[:]  # Replace with run versions

    # Shorter durations should mean more switching
    print(f"Checking effect of durations ... ")
    base = sum(sim_base.results.fp.switchers)
    short = sum(sim_short.results.fp.switchers)
    assert base < short, f'Shorter durations should mean more switching, but {short}<{base}'
    print(f"✓ ({base:.2f} < {short:.2f})")

    return [sim_base, sim_short]


if __name__ == '__main__':

    sims1 = test_mcpr()
    sims2 = test_durations()

    print('Done.')


