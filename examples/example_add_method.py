"""
Example script demonstrating the add_method intervention.

This script shows how to add a new contraceptive method to a simulation
at a specific year, with switching behavior copied from an existing method.

The add_method intervention:
- Registers a new method at simulation initialization
- Activates the method at a specified year
- Copies switching probabilities from an existing method
"""

import fpsim as fp
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Run two sims:
    - Baseline (no new method)
    - With a new method introduced at a given year

    Then plot key outcomes and (optionally) compare alternative duration-of-use
    distributions for the *new* method.
    """

    # --- User-editable knobs ---
    do_save = True
    do_show = True
    do_compare_duration_distributions = True  # optional "Part B"

    pars = dict(
        n_agents=2000,
        start=2000,
        stop=2020,
        location='kenya',
        verbose=0,
    )
    introduction_year = 2010

    # --- Part A: add a single new method and compare to baseline ---
    # Define the method we want to introduce
    new_method = fp.Method(
        name='new_inj',                 # internal identifier (must be unique)
        label='New Injectable',         # human-readable name (used in plots)
        csv_name='New Injectable',      # name used if exporting / matching CSVs
        efficacy=0.995,
        modern=True,
        dur_use=ss.lognorm_ex(mean=3, std=1.5),
    )

    # Define the intervention:
    # - "copy_from" is an existing method whose switching behavior is copied
    # - "split_shares" controls how much share is carved out for the new method
    intv = fp.add_method(
        year=introduction_year,
        method=new_method,
        copy_from='inj',
        split_shares=0.99,
        verbose=True,
    )

    # Run baseline vs intervention sims
    sim_baseline = fp.Sim(pars=pars, label='Baseline').run()
    sim_with_method = fp.Sim(pars=pars, interventions=[intv], label='With new method').run()

    fig, stats = add_method_results(
        sim_baseline=sim_baseline,
        sim_with_method=sim_with_method,
        new_method_name=new_method.name,
        introduction_year=introduction_year,
        pars=pars,
        do_save=do_save,
    )

    # --- Part B (optional): compare different duration-of-use distributions ---
    if do_compare_duration_distributions:
        duration_distributions = {
            'Lognormal': ss.lognorm_ex(mean=3, std=1.5),
            'Gamma': ss.gamma(a=4, scale=0.75),
            'Weibull': ss.weibull(c=2.5, scale=3),
            'Exponential': ss.expon(scale=3),
        }

        dist_sims = {'Baseline': sim_baseline}
        for dist_name, dur_dist in duration_distributions.items():
            test_method = fp.Method(
                name=f'dist_{dist_name.lower()[:4]}',
                label=f'{dist_name} duration',
                efficacy=new_method.efficacy,
                modern=True,
                dur_use=dur_dist,
            )
            test_intv = fp.add_method(
                year=introduction_year,
                method=test_method,
                copy_from='impl',
                verbose=False,
            )
            dist_sims[dist_name] = fp.Sim(pars=pars, interventions=[test_intv], label=dist_name).run()

        distribution_comparison(dist_sims, introduction_year, do_save=do_save)
        _, sorted_diffs, _ = summary(dist_sims, sim_baseline, do_save=do_save)

        print("\nDuration distribution impacts (mCPR change from baseline):")
        for label, diff in sorted_diffs.items():
            print(f"  {label:12s}: {diff:+.1f}pp")

    if do_show:
        plt.show()


def _get_years(sim):
    """Convert Starsim time vector to numeric years for plotting."""
    return np.array([float(t) for t in sim.results.timevec.to_float()])

def add_method_results(sim_baseline, sim_with_method, new_method_name, introduction_year, pars, do_save=True):
    """Plot the impact of adding a new contraceptive method."""
    cm_new = sim_with_method.connectors.contraception
    fp_new = sim_with_method.connectors.fp
    new_method_idx = cm_new.methods[new_method_name].idx
    
    years = _get_years(sim_with_method)
    mcpr_baseline = sim_baseline.results.contraception.mcpr * 100
    mcpr_new = sim_with_method.results.contraception.mcpr * 100
    method_mix = fp_new.method_mix
    new_method_usage = method_mix[new_method_idx, :] * 100
    births_baseline = sim_baseline.results.fp.cum_births
    births_new = sim_with_method.results.fp.cum_births
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of Adding a New Contraceptive Method', fontsize=16, fontweight='bold')
    
    # Plot 1: mCPR Comparison
    ax = axes[0, 0]
    ax.plot(years, mcpr_baseline, 'b-', linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(years, mcpr_new, 'r-', linewidth=2, label='With New Method', alpha=0.8)
    ax.axvline(x=introduction_year, color='green', linestyle='--', linewidth=1.5, 
                label=f'Method introduced ({introduction_year})', alpha=0.7)
    ax.fill_between(years, mcpr_baseline, mcpr_new, alpha=0.2, color='green',
                    where=(years >= introduction_year))
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('mCPR (%)', fontsize=11)
    ax.set_title('Modern Contraceptive Prevalence Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pars['start'], pars['stop'])
    
    # Plot 2: Method Mix Over Time
    ax = axes[0, 1]
    method_colors = {
        'none': '#d9d9d9', 'pill': '#e41a1c', 'iud': '#377eb8', 'inj': '#4daf4a',
        'cond': '#984ea3', 'btl': '#ff7f00', 'impl': '#a65628', 'new_inj': '#f781bf',
    }
    bottom = np.zeros(len(years))
    for name, method in cm_new.methods.items():
        if name != 'none':
            usage = method_mix[method.idx, :] * 100
            color = method_colors.get(name, f'C{method.idx}')
            ax.fill_between(years, bottom, bottom + usage, label=method.label, color=color, alpha=0.7)
            bottom += usage
    ax.axvline(x=introduction_year, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Method Usage (%)', fontsize=11)
    ax.set_title('Method Mix Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pars['start'], pars['stop'])
    ax.set_ylim(0, 100)
    
    # Plot 3: New Method Uptake
    ax = axes[1, 0]
    inj_idx = cm_new.methods['inj'].idx
    inj_usage = method_mix[inj_idx, :] * 100
    ax.plot(years, inj_usage, 'g-', linewidth=2, label='Standard Injectable', alpha=0.8)
    ax.plot(years, new_method_usage, color='#f781bf', linewidth=2.5, label='New Injectable', alpha=0.9)
    ax.axvline(x=introduction_year, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    intro_idx = np.argmin(np.abs(years - introduction_year))
    ax.annotate('Method\nIntroduced', xy=(introduction_year, new_method_usage[intro_idx]),
                xytext=(introduction_year + 2, max(new_method_usage) * 0.5),
                fontsize=9, ha='left', arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Usage (%)', fontsize=11)
    ax.set_title('New Injectable Uptake vs Standard', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pars['start'], pars['stop'])
    
    # Plot 4: Cumulative Births Comparison
    ax = axes[1, 1]
    ax.plot(years, births_baseline, 'b-', linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(years, births_new, 'r-', linewidth=2, label='With New Injectable', alpha=0.8)
    ax.axvline(x=introduction_year, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    diff = births_baseline[-1] - births_new[-1]
    pct_diff = (diff / births_baseline[-1]) * 100 if births_baseline[-1] > 0 else 0
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Cumulative Births', fontsize=11)
    ax.set_title(f'Cumulative Births (Diff: {diff:.0f}, {pct_diff:.1f}%)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pars['start'], pars['stop'])
    
    plt.tight_layout()
    if do_save:
        plt.savefig('add_method_results.png', dpi=150, bbox_inches='tight')
    
    stats = {
        'mcpr_baseline': float(mcpr_baseline[-1]),
        'mcpr_new': float(mcpr_new[-1]),
        'new_method_usage': float(new_method_usage[-1]),
        'births_averted': float(diff),
        'births_averted_pct': float(pct_diff),
    }
    return fig, stats


def distribution_comparison(dist_sims, introduction_year, colors=None, do_save=True):
    """Plot comparison of different duration distributions."""
    if colors is None:
        colors = {'Baseline': 'black', 'Lognormal': '#e41a1c', 'Gamma': '#377eb8', 
                    'Weibull': '#4daf4a', 'Exponential': '#984ea3'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of Different Duration Distributions', fontsize=16, fontweight='bold')
    
    # Plot 1: mCPR comparison
    ax = axes[0, 0]
    for label, sim in dist_sims.items():
        mcpr = sim.results.contraception.mcpr * 100
        years_plot = _get_years(sim)
        linestyle = '--' if label == 'Baseline' else '-'
        ax.plot(years_plot, mcpr, color=colors[label], linestyle=linestyle, linewidth=2, label=label, alpha=0.8)
    ax.axvline(x=introduction_year, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('mCPR (%)')
    ax.set_title('Modern CPR by Duration Distribution')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: New method usage over time
    ax = axes[0, 1]
    for label, sim in dist_sims.items():
        if label == 'Baseline':
            continue
        cm = sim.connectors.contraception
        method_name = f'dist_{label.lower()[:4]}'
        if method_name in cm.methods:
            method_idx = cm.methods[method_name].idx
            usage = sim.connectors.fp.method_mix[method_idx, :] * 100
            years_plot = _get_years(sim)
            ax.plot(years_plot, usage, color=colors[label], linewidth=2, label=label, alpha=0.8)
    ax.axvline(x=introduction_year, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('New Method Usage (%)')
    ax.set_title('New Method Adoption by Distribution Type')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative births comparison
    ax = axes[1, 0]
    for label, sim in dist_sims.items():
        births = sim.results.fp.cum_births
        years_plot = _get_years(sim)
        linestyle = '--' if label == 'Baseline' else '-'
        ax.plot(years_plot, births, color=colors[label], linestyle=linestyle, linewidth=2, label=label, alpha=0.8)
    ax.axvline(x=introduction_year, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Births')
    ax.set_title('Cumulative Births by Distribution Type')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final method usage bar chart
    ax = axes[1, 1]
    final_usage = {}
    for label, sim in dist_sims.items():
        if label == 'Baseline':
            continue
        cm = sim.connectors.contraception
        method_name = f'dist_{label.lower()[:4]}'
        if method_name in cm.methods:
            method_idx = cm.methods[method_name].idx
            final_usage[label] = sim.connectors.fp.method_mix[method_idx, -1] * 100
    
    bars = ax.bar(final_usage.keys(), final_usage.values(), 
                    color=[colors[k] for k in final_usage.keys()], alpha=0.8)
    ax.set_ylabel('Final Method Usage (%)')
    ax.set_title('Final Usage by Duration Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, final_usage.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if do_save:
        plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
    
    return fig


def summary(dist_sims, sim_baseline, do_save=True):
    """Plot summary comparison of distribution impacts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Summary: Impact of Duration Distributions', fontsize=14, fontweight='bold')
    
    # mCPR differences
    ax = axes[0]
    mcpr_baseline_end = sim_baseline.results.contraception.mcpr[-1] * 100
    all_diffs = {}
    for label, sim in dist_sims.items():
        if label != 'Baseline':
            mcpr_end = sim.results.contraception.mcpr[-1] * 100
            all_diffs[label] = mcpr_end - mcpr_baseline_end
    
    sorted_diffs = dict(sorted(all_diffs.items(), key=lambda x: x[1], reverse=True))
    bar_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in sorted_diffs.values()]
    ax.barh(list(sorted_diffs.keys()), list(sorted_diffs.values()), color=bar_colors, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('mCPR Change from Baseline (percentage points)')
    ax.set_title('Impact on Modern CPR')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Births averted
    ax = axes[1]
    births_baseline_end = sim_baseline.results.fp.cum_births[-1]
    births_averted = {}
    for label, sim in dist_sims.items():
        if label != 'Baseline':
            births_end = sim.results.fp.cum_births[-1]
            births_averted[label] = births_baseline_end - births_end
    
    sorted_averted = dict(sorted(births_averted.items(), key=lambda x: x[1], reverse=True))
    bar_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in sorted_averted.values()]
    ax.barh(list(sorted_averted.keys()), list(sorted_averted.values()), color=bar_colors, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Births Averted')
    ax.set_title('Impact on Births')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if do_save:
        plt.savefig('summary_comparison.png', dpi=150, bbox_inches='tight')
    
    return fig, sorted_diffs, births_averted


if __name__ == '__main__':
    main()
