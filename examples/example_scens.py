# -*- coding: utf-8 -*-
"""

Sample scenarios script using novel method 

Created on Thu Jul 14 11:09:58 2022

@author: michelleob
"""

debug = 1 

if __name__ == '__main__':

    import fpsim as fp
    
    #keep structure below if replicating to create new scenarios; first number is the full run, second number is for debug
    n_agents   = [10_000, 100][debug]
    start_year = [1980, 2010][debug]
    repeats    = [10, 1][debug]
    year       = 2030
    youth_ages = ['<18', '18-20']

    pars = fp.pars(n_agents=n_agents, start_year=start_year, end_year=2040)
    pars.add_method(name='new injectables', eff=0.983)


    #%% Add new method for under 20 agents   
    method = 'new injectables'
    kw = dict(method=method) #kwargs for adjusting initialization/discontinuation
    d_kw = dict(dest=method) #kwargs for adjusting switching

    s1 = fp.make_scen(
            label = 'intro new method, double init',
            year  = year,
            probs = [
                dict(copy_from='Injectables', **kw),
                dict(init_factor=2.0, **kw),
                    ]
                )
    
    s2 = fp.make_scen(
           label = 'intro new method, 20% switching',
           year  = year,
           probs = [
                dict(source = 'Injectables', value = 0.20, **d_kw)
                ])


  #%% Create sims
    scens = fp.Scenarios(pars=pars, repeats=repeats)
    scens.add_scen(label='Baseline')
    scens.add_scen(s1)
    scens.add_scen(s2)

    # Run scenarios
    scens.run(serial=debug)

    # Plot and print results
   
    scens_fig = scens.plot()
    cpr_fig = scens.plot('cpr')
    
    
    ##set color dict for method mix plot
    colors = {
    'None'              : [0.0, 0.0, 0.0],
    'Withdrawal'        : [0.3, 0.3, 0.3],
    'Other traditional' : [0.5, 0.5, 0.5],
    'Condoms'           : [0.7, 0.7, 0.7],
    'Pill'              : [0.3, 0.8, 0.9],
    'Injectables'       : [0.6, 0.4, 0.9],
    'Implants'          : [0.4, 0.2, 0.9],
    'IUDs'              : [0.0, 0.0, 0.9],
    'BTL'               : [0.8, 0.0, 0.0],
    'Other modern'      : [0.8, 0.5, 0.5],
    'new injectables'   : [0.2, 0.8, 0.2],
    }
                        
    meth_fig = scens.plot('method', colors=[color for color in colors.values()])                        
    for fig in [meth_fig]:
                    for ax in fig.axes:
                        ax.set_xlim(left = 2010,   #note that 'historical' method mix is not calibrated
                                    right = 2040)                    
    print('Done.')