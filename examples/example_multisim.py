'''
Simple example usage for FPsim multisim
'''

import sciris as sc
import fpsim as fp

# Set options
do_plot = True
pars1 = fp.pars(location='amhara')
pars1['n_agents'] = 500 # Small population size
pars1['end_year'] = 2020 # 1961 - 2020 is the normal date range
pars1['exposure_factor'] = 1.0 # Overall scale factor on probability of becoming pregnant

pars2 = fp.pars(location='somali')
pars2['n_agents'] = 500 # Small population size
pars2['end_year'] = 2020 # 1961 - 2020 is the normal date range
pars2['exposure_factor'] = 1.0 # Overall scale factor on probability of becoming pregnant


if __name__ == '__main__':
    sc.tic()

    sim1 = fp.Sim(pars=pars1, label='Amhara')
    sim2 = fp.Sim(pars=pars2, label='Somali')

    msim = fp.MultiSim(sims=[sim1, sim2])

    msim.run()

    if do_plot:
        msim.plot()

    sc.toc()
    print('Done.')
