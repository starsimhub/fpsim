"""
Test plotting sims
"""

import sciris as sc
import pylab as pl


# par_kwargs = dict(n_agents=1000, start_year=1960, end_year=2020, seed=1, verbose=1)
par_kwargs = dict(n_agents=500, start_year=2000, end_year=2010, seed=1, verbose=-1)
serial = 1  # Whether to run in serial (for debugging)


from test_sim import test_simple_choice


def plot_simple_choice():
    sc.heading('Method choice is based on age & previous method')

    # Make & run sim
    sims = test_simple_choice()

    # Plots
    fig, axes = pl.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    cind = 0

    # mCPR
    ax = axes[0]
    ax.plot(sims[0].results.t, sims[0].results.cpr)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # mCPR by age
    ax = axes[1]
    for alabel, ares in sim['analyzers'].results.items():
        ax.plot(sim.results.t, ares, label=alabel, color=colors[cind])
        cind += 1
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # Plot method mix
    ax = axes[2]
    oc = sim.people.filter(sim.people.on_contra)
    method_props = [sc.safedivide(len(oc.filter(oc.method == i)), len(oc)) for i in range(1, 10)]
    method_labels = [m.name for m in sim.contraception_module.methods.values() if m.label != 'None']
    ax.bar(method_labels, method_props)
    ax.set_ylabel('Proportion among all users')
    ax.set_title('Contraceptive mix')

    sc.figlayout()
    pl.show()

    return sims



if __name__ == '__main__':

    sims = plot_simple_choice()
    print('Done.')
