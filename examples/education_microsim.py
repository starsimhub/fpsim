'''
Simple and small simulation using education analyzer
'''

import sciris as sc
import fpsim as fp
import matplotlib.pyplot as plt

# Set options
do_plot = True
pars = fp.pars(location='kenya')
pars['n_agents'] = 50  # Small population size

sc.tic()
sim = fp.Sim(pars=pars, analyzers=[fp.education_recorder()], education_module=fp.Education())
sim.run()

if do_plot:
    edu_analyzer= sim.get_analyzers()[0]
    # Plot a subset of the available trajectories for female individuals
    for idx in range(max(2, edu_analyzer.max_agents//8)):
        edu_analyzer.plot(index=idx)
    edu_analyzer.plot_waterfall()
    plt.show()

sc.toc()
print('Done.')
