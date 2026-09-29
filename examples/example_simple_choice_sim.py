import sciris as sc
import fpsim as fp

import matplotlib.pyplot as plt


# Set options
do_plot = True
location = 'kenya'
pars = dict(
    location = location,
    n_agents = 200,          # Small population size
    end_year = 2020,         # 1961 - 2020 is the normal date range
    exposure_factor = 1.0,   # Overall scale factor on probability of becoming pregnant
)

method_choice = fp.SimpleChoice(location=location)
sim = fp.Sim(pars=pars, contraception_module=method_choice)

sc.tic()
sim.run()
sc.toc()

if do_plot:
    sim.plot(to_plot='cpr')
    plt.show()

print('Done.')