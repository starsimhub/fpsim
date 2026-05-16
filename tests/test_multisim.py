"""
Run tests on the multisim object.
"""

import sciris as sc
import fpsim as fp
import starsim as ss

parallel  = True
do_plot = 0
sc.options(backend='agg') # Turn off interactive plots


def test_multisim(do_plot=do_plot):
    ''' Try running a multisim '''
    sc.heading('Testing multisim...')

    # Generate sims in a loop
    sims = []
    for i in range(3):
        exposure = 0.5 + 0.5*i # Run a sweep over exposure
        pars = dict(test=True, exposure_factor=exposure)
        sim = fp.Sim(pars=pars, label=f'Exposure {exposure}')
        sims.append(sim)

    msim = ss.MultiSim(sims)
    msim.run(parallel=parallel)  # Run sims in parallel

    msim.reduce()
    msim.summarize()

    births = msim.results.fp_births
    assert sum(births.low) < sum(births.high), 'Expecting the higher bound of births to be higher than the lower bound'

    if do_plot:
        msim.plot()

    return msim


def devtest_eth_multisim():
    # TODO: deprecate this?
    sc.heading('Testing Ethiopia multisim...')
    sim1 = fp.Sim(location='addis_ababa')
    sim2 = fp.Sim(location='afar')
    sim3 = fp.Sim(location='amhara')
    sim4 = fp.Sim(location='benishangul_gumuz')
    sim5 = fp.Sim(location='dire_dawa')
    sim6 = fp.Sim(location='gambela')
    sim7 = fp.Sim(location='harari')
    sim8 = fp.Sim(location='oromia')
    sim9 = fp.Sim(location='snnpr')
    sim10 = fp.Sim(location='somali')
    sim11 = fp.Sim(location='tigray')
    msim = ss.MultiSim(sims=[sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11])

    msim.run()

    return msim


if __name__ == '__main__':
    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():  # Start timing
        msim = test_multisim()
        msim_eth = devtest_eth_multisim()
