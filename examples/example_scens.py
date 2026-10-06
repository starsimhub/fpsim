"""
Sample scenarios script, comparing a baseline against an efficacy change and a
parameter change.

Note that fp.Scenarios only supports the scenario types documented on fp.Scenario
(``eff``, ``par`` and custom ``interventions``). To introduce an entirely new
contraceptive method, use the fp.add_method intervention instead -- see
example_add_method.py.
"""

debug = 1

if __name__ == '__main__':

    import fpsim as fp
    import matplotlib.pyplot as plt

    # First number is the full run, second is for debug
    n_agents   = [10_000, 100][debug]
    start_year = [1980, 2010][debug]
    year       = 2030 if not debug else 2015
    end_year   = 2040 if not debug else 2020

    pars = dict(
        location   = 'kenya',
        n_agents   = n_agents,
        start_year = start_year,
        end_year   = end_year,
    )

    # Scenarios are specified by label; see sim.connectors.contraception.methods for options
    s1 = fp.make_scen(
        label = 'More effective injectables',
        pars  = pars,
        eff   = {'Injectables': 0.99},
        year  = year,
    )

    s2 = fp.make_scen(
        label     = 'Halve exposure',
        pars      = pars,
        par       = 'exposure_factor',
        par_years = year,
        par_vals  = 0.5,
    )

    # Create and run scenarios
    scens = fp.Scenarios(repeats=1)
    scens.add_scen(fp.make_scen(label='Baseline', pars=pars))
    scens.add_scen(s1)
    scens.add_scen(s2)
    scens.run()

    # Plot results
    scens.plot()
    scens.plot(key='cpr')
    plt.show()

    print('Done.')
