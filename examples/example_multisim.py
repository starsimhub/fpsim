'''
Simple example usage for FPsim multisim
'''

import sciris as sc
import starsim as ss
import fpsim as fp

# Set options
do_plot = True
def make_pars(location):
    return dict(
        location = location,
        n_agents = 500,         # Small population size
        end_year = 2020,        # 1961 - 2020 is the normal date range
        exposure_factor = 1.0,  # Overall scale factor on probability of becoming pregnant
    )

pars1 = make_pars('kenya')
pars2 = make_pars('senegal')


if __name__ == '__main__':
    sc.tic()

    sim1 = fp.Sim(pars=pars1, label='Kenya')
    sim2 = fp.Sim(pars=pars2, label='Senegal')

    msim = ss.MultiSim(sims=[sim1, sim2])

    msim.run()

    if do_plot:
        msim.plot()

    sc.toc()
    print('Done.')
