"""
Example script demonstrating DMPA-SC intervention scenarios.

This script implements the scenarios defined in GitHub issue #416.

Baseline:
- 2% annual mCPR growth via increased initiation of injectable users
- Simplified implementation: uniform 2% across all ages (method mix stays constant)
- Full spec (not yet implemented): age-specific baseline initiation rates
  (Under 18: 0.17%, Ages 18-19: 1.04%) from DHS contraceptive calendars

Intervention scenarios (add 3-month and 6-month DMPA-SC):
- Scenario 1: 3-month DMPA-SC scale-up from 1.5% (2025) to 5% (2040) in women <20
- Scenario 2: 6-month DMPA-SC from 1% (2030) to 5% (2040) in women <20,
              with 26% switching from 3-month and 20% from traditional methods
- Scenario 3: Combined (identical to Scenario 2)

Placeholders (to be replaced with data-driven values):
- Initiation rates: 1.5%→5% (Scenario 1), 1%→5% (Scenario 2)
- Switching rates: 26% (3mo→6mo), 20% (trad→6mo)
- Duration scaling: rel_dur_use=2.0 (3-month), 2.5 (6-month)
"""

import numpy as np
import matplotlib.pyplot as plt
import starsim as ss
import fpsim as fp


def make_baseline_inj_growth(pars, *, name):
    """
    Create baseline intervention: 2% annual mCPR growth via increased injectable initiation.
    
    Simplified implementation: uniform 2% initiation increase across all ages,
    targeting injectable method. Method mix stays constant (no new products).
    
    Full specification (from requirements table): age-specific baseline initiation
    rates from DHS contraceptive calendars (Under 18: 0.17%, Ages 18-19: 1.04%).
    This simplified version uses uniform 2% for demonstration.
    
    Args:
        pars (dict): Simulation parameters
        name (str): Intervention name
        
    Returns:
        fp.change_initiation: Configured baseline intervention
    """
    return fp.change_initiation(
        years=[pars['start'], pars['stop']],
        perc=0.02,
        annual=True,
        perc_of_eligible=True,
        target_method='inj',
        name=name,
    )


def main(do_save=True, do_show=True):
    """
    Run all DMPA-SC scenarios and generate comparison plots.
    
    Args:
        do_save (bool): Save dashboard PNG to dmpasc_dashboard.png
        do_show (bool): Display plots interactively (blocks on plt.show())
    """
    
    pars = dict(
        n_agents=5000,
        start=2000,
        stop=2045,
        location='kenya',
        rand_seed=1,
        verbose=0,
    )
    
    print("Running DMPA-SC scenarios...")
    print("=" * 60)
    
    scenarios = {}
    print("\n1. Baseline (2% annual injectable growth)...")
    scenarios['Baseline'] = run_baseline(pars)
    
    print("2. Scenario 1 (3-month DMPA-SC)...")
    scenarios['Scenario 1: 3-month DMPA-SC'] = run_scenario_1(pars)
    
    print("3. Scenario 2 (6-month DMPA-SC)...")
    scenarios['Scenario 2: 6-month DMPA-SC'] = run_scenario_2(pars)
    
    print("4. Scenario 3 (Combined, reusing Scenario 2)...")
    scenarios['Scenario 3: Combined'] = scenarios['Scenario 2: 6-month DMPA-SC']
    
    print("\n" + "=" * 60)
    print("Generating plots...")
    
    # Minimal figure: mCPR comparison and total DMPA-SC uptake
    colors = {
        'Baseline': '#000000',
        'Scenario 1: 3-month DMPA-SC': '#e41a1c',
        'Scenario 2: 6-month DMPA-SC': '#377eb8',
        'Scenario 3: Combined': '#4daf4a',
    }
    
    markers = {
        'Baseline': 'o',
        'Scenario 1: 3-month DMPA-SC': 's',
        'Scenario 2: 6-month DMPA-SC': '^',
        'Scenario 3: Combined': 'D',
    }

    fig, (ax_mcpr, ax_dmpasc) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # mCPR: plot non-baseline first, baseline last so it's always visible
    for label, sim in scenarios.items():
        if label == 'Baseline':
            continue
        years = _get_years(sim)
        mcpr = sim.results.contraception.mcpr * 100
        ax_mcpr.plot(
            years, mcpr,
            color=colors.get(label, 'gray'),
            lw=2,
            marker=markers.get(label, 'o'),
            markersize=4,
            markevery=24,
            label=label,
            alpha=0.9
        )

    base = scenarios['Baseline']
    years = _get_years(base)
    mcpr = base.results.contraception.mcpr * 100
    ax_mcpr.plot(
        years, mcpr,
        color=colors['Baseline'],
        ls='--',
        lw=2.25,
        marker='o',
        markersize=3,
        markerfacecolor='white',
        markeredgewidth=0.8,
        markevery=24,
        label='Baseline',
        alpha=0.95,
    )

    for ax in (ax_mcpr, ax_dmpasc):
        ax.axvline(x=2025, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=2030, color='gray', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(pars['start'], pars['stop'])

    ax_mcpr.set_title('Modern Contraceptive Prevalence Rate (mCPR)', fontweight='bold')
    ax_mcpr.set_ylabel('mCPR (%)')
    ax_mcpr.legend(loc='upper left', fontsize=8)

    # Total DMPA-SC uptake: sum all DMPA-SC variants per scenario
    max_y = 0.0
    for label, sim in scenarios.items():
        if label == 'Baseline':
            continue
        years = _get_years(sim)
        cm = sim.connectors.contraception
        fp_conn = sim.connectors.fp

        total = np.zeros_like(years, dtype=float)
        for name, method in cm.methods.items():
            if 'dmpasc' in name.lower() and name != 'none':
                total += fp_conn.method_mix[method.idx, :] * 100

        if len(total):
            max_y = max(max_y, float(np.nanmax(total)))
        ax_dmpasc.plot(
            years, total,
            color=colors.get(label, 'gray'),
            lw=2,
            marker=markers.get(label, 'o'),
            markersize=4,
            markevery=24,
            label=label,
            alpha=0.9
        )

    ax_dmpasc.set_title('Total DMPA-SC uptake (method mix)', fontweight='bold')
    ax_dmpasc.set_ylabel('DMPA-SC usage (%)')
    ax_dmpasc.set_xlabel('Year')
    ax_dmpasc.set_ylim(0, max(0.25, max_y * 1.2))
    ax_dmpasc.legend(loc='upper left', fontsize=8)

    fig.suptitle('DMPA-SC Intervention Scenarios', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if do_save:
        fig.savefig('dmpasc_dashboard.png', dpi=150, bbox_inches='tight')
    
    # Print summary statistics
    print_summary_statistics(scenarios)
    
    if do_show:
        plt.show()
    else:
        plt.close(fig)


def run_baseline(pars):
    """
    Run baseline scenario: 2% annual mCPR growth via increased injectable initiation.
    
    Returns:
        fp.Sim: Completed simulation with baseline interventions.
    """
    intv = make_baseline_inj_growth(pars, name='baseline_inj_growth')
    sim = fp.Sim(pars=pars, interventions=[intv], label='Baseline')
    sim.run()
    return sim


def run_scenario_1(pars):
    """
    Run Scenario 1: 3-month DMPA-SC scale-up in women under 20.
    
    Intervention components:
    - Initiation: gradual increase from 1.5% (2025) to 5% (2040) in women <20
    - Switching: same as baseline (no additional switching)
    - Discontinuation: rel_dur_use=2.0 (placeholder: 2x injectable duration)
    
    Returns:
        fp.Sim: Completed simulation with Scenario 1 interventions.
    """
    baseline_intv = make_baseline_inj_growth(pars, name='baseline_s1')
    
    add_method_intv = fp.add_method(
        year=2025,
        method=None,
        method_pars={
            'name': 'dmpasc3',
            'label': 'DMPA-SC 3-month',
            'dur_use': ss.lognorm_ex(mean=2, std=1),
            'rel_dur_use': 2.0,  # Placeholder: 2x duration vs injectables
        },
        copy_from='inj',
        split_shares=0.0,
        verbose=False,
        name='add_dmpasc3_s1'
    )
    
    initiation_intv = fp.change_initiation(
        years=[2025, 2040],
        age_range=(0, 20),
        perc=0.015,
        final_perc=0.05,
        perc_of_eligible=True,
        target_method='dmpasc3',
        annual=True,
        verbose=False,
        name='init_dmpasc3_s1'
    )
    
    interventions = [baseline_intv, add_method_intv, initiation_intv]
    sim = fp.Sim(pars=pars, interventions=interventions, label='Scenario 1')
    sim.run()
    return sim


def run_scenario_2(pars):
    """
    Run Scenario 2: 6-month DMPA-SC with 3-month scale-up and switching.
    
    Intervention components:
    - Initiation: 3-month from 1.5% (2025)→5% (2040), 6-month from 1% (2030)→5% (2040), both <20
    - Switching: 26% from 3mo→6mo, 20% from traditional→6mo (applied in 2030)
    - Discontinuation: rel_dur_use=2.0 (3-month), 2.5 (6-month) - placeholders
    
    All rates are placeholders to be replaced with data-driven estimates from
    DHS/PMA/clinical trials.
    
    Returns:
        fp.Sim: Completed simulation with Scenario 2 interventions.
    """
    baseline_intv = make_baseline_inj_growth(pars, name='baseline_s2')
    
    add_3month_intv = fp.add_method(
        year=2025,
        method_pars={'name': 'dmpasc3', 'label': 'DMPA-SC 3-month', 
                     'dur_use': ss.lognorm_ex(mean=2, std=1), 'rel_dur_use': 2.0},
        copy_from='inj',
        split_shares=0.0,
        verbose=False,
        name='add_dmpasc3_s2'
    )
    
    add_6month_intv = fp.add_method(
        year=2030,
        method_pars={'name': 'dmpasc6', 'label': 'DMPA-SC 6-month',
                     'dur_use': ss.lognorm_ex(mean=2, std=1), 'rel_dur_use': 2.5},
        copy_from='inj',
        split_shares=0.0,
        verbose=False,
        name='add_dmpasc6_s2'
    )
    
    init_3month_intv = fp.change_initiation(
        years=[2025, 2040], age_range=(0, 20), perc=0.015, final_perc=0.05,
        perc_of_eligible=True, target_method='dmpasc3', annual=True, verbose=False,
        name='init_dmpasc3_s2'
    )
    
    init_6month_intv = fp.change_initiation(
        years=[2030, 2040], age_range=(0, 20), perc=0.01, final_perc=0.05,
        perc_of_eligible=True, target_method='dmpasc6', annual=True, verbose=False,
        name='init_dmpasc6_s2'
    )
    
    switching_intv = fp.method_switching(
        year=2030,
        from_methods=['dmpasc3', 'wdraw', 'othtrad'],
        to_method='dmpasc6',
        switch_prob={'dmpasc3': 0.26, 'wdraw': 0.20, 'othtrad': 0.20},
        annual=False,
        verbose=False,
        name='switch_to_dmpasc6_s2'
    )
    
    interventions = [baseline_intv, add_3month_intv, init_3month_intv,
                     add_6month_intv, init_6month_intv, switching_intv]
    sim = fp.Sim(pars=pars, interventions=interventions, label='Scenario 2')
    sim.run()
    return sim


def _get_years(sim):
    """
    Convert Starsim time vector to numeric years for plotting.
    
    Args:
        sim (fp.Sim): Simulation object
        
    Returns:
        np.ndarray: Array of years as floats
    """
    return np.array([float(t) for t in sim.results.timevec.to_float()])


def print_summary_statistics(scenarios):
    """
    Print summary statistics for all scenarios.
    
    Displays mCPR, change in mCPR vs baseline, births averted, and
    DMPA-SC method uptake by scenario.
    
    Args:
        scenarios (dict): Dictionary of {label: sim} for all scenarios
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Final Timestep)")
    print("=" * 80)
    
    baseline = scenarios['Baseline']
    baseline_mcpr = baseline.results.contraception.mcpr[-1] * 100
    baseline_births = baseline.results.fp.cum_births[-1]
    
    print(f"\n{'Scenario':<40} {'mCPR (%)':<12} {'Δ mCPR':<12} {'Births Averted':<15}")
    print("-" * 80)
    
    for label, sim in scenarios.items():
        mcpr = sim.results.contraception.mcpr[-1] * 100
        births_averted = baseline_births - sim.results.fp.cum_births[-1]
        print(f"{label:<40} {mcpr:>10.2f}  {mcpr - baseline_mcpr:>10.2f}  {births_averted:>13.0f}")
    
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("DMPA-SC METHOD UPTAKE (Final Timestep)")
    print("=" * 80)
    
    for label, sim in scenarios.items():
        if label == 'Baseline':
            continue
        print(f"\n{label}:")
        cm = sim.connectors.contraception
        fp_conn = sim.connectors.fp
        for name, method in cm.methods.items():
            if 'dmpasc' in name.lower() and name != 'none':
                usage_pct = fp_conn.method_mix[method.idx, -1] * 100
                n_users = (sim.people.alive & (sim.people.fp.method == method.idx)).sum()
                print(f"  {method.label:<30} {usage_pct:>6.2f}%  ({n_users:>5} users)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main(do_save=True, do_show=True)
