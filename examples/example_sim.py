'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim as fp
from fpsim import plotting as plt

# Set options
do_plot = True
pars = dict(location='senegal',
            n_agents = 500, # Small population size
            end_year = 2020, # 1961 - 2020 is the normal date range
            exposure_factor = 1.0, # Overall scale factor on probability of becoming pregnant
            )

sc.tic()
sim = fp.Sim(pars=pars)
sim.run()

if do_plot:
    sim.plot()
    plt.plot_calib(sim)

sc.toc()
print('Done.')
