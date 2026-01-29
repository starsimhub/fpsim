"""
Example script demonstrating DMPA-SC intervention scenarios.

This script implements the scenarios defined in GitHub issue #416:
- Baseline: 2% annual increase in contraception users
- Scenario 1: 3-month DMPA-SC scale-up from 1.5% (2025) to 5% (2040) in women under 20
- Scenario 2: 6-month DMPA-SC scale-up from 1% (2030) to 5% (2040) in women under 20,
              with 26% switching from 3-month and 20% from traditional methods
- Scenario 3: Combined scenarios
- Placeholder: Simplified single DMPA-SC with 5% initial initiation, 2% annual,
               10% switching from injectables, 5% from traditional, 2x duration
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import fpsim as fp


def main():
    """Run all DMPA-SC scenarios and generate comparison plots."""
    
    # Configuration
    do_save = True
    do_show = True
    
    pars = dict(
        n_agents=5000,
        start=2000,
        stop=2045,
        location='kenya',
        verbose=0,
    )
    
    print("Running DMPA-SC intervention scenarios...")
    print("=" * 60)
    
    # Run all scenarios
    scenarios = {}
    
    print("\n1. Running baseline scenario (2% annual growth)...")
    scenarios['Baseline'] = run_baseline(pars)
    
    print("2. Running Scenario 1 (3-month DMPA-SC)...")
    scenarios['Scenario 1: 3-month DMPA-SC'] = run_scenario_1(pars)
    
    print("3. Running Scenario 2 (6-month DMPA-SC)...")
    scenarios['Scenario 2: 6-month DMPA-SC'] = run_scenario_2(pars)
    
    print("4. Running Scenario 3 (Combined)...")
    scenarios['Scenario 3: Combined'] = run_scenario_3(pars)
    
    print("5. Running Placeholder scenario...")
    scenarios['Placeholder: Simple DMPA-SC'] = run_placeholder(pars)
    
    print("\n" + "=" * 60)
    print("All scenarios completed. Generating plots...")
    
    # Generate a single "dashboard" figure with all plots as subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=4,
        height_ratios=[1.2, 1.0, 1.0, 1.0, 0.05],
        hspace=0.45,
        wspace=0.35,
    )

    # Row 0: Scenario comparison spans full width
    ax_mcpr = fig.add_subplot(gs[0, :])

    # Row 1: Births (2 panels, each spans 2 columns)
    ax_births_cum = fig.add_subplot(gs[1, 0:2])
    ax_births_averted = fig.add_subplot(gs[1, 2:4])

    # Rows 2-3: Method uptake (left 2 cols) and age-specific uptake (right 2 cols)
    method_axes = np.array([
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])],
    ])
    age_axes = np.array([
        [fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3])],
        [fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[3, 3])],
    ])

    plot_scenario_comparison(scenarios, pars, ax=ax_mcpr)
    plot_births_comparison(scenarios, pars, axes=(ax_births_cum, ax_births_averted))
    plot_method_uptake(scenarios, pars, axes=method_axes)
    plot_age_specific_uptake(scenarios, pars, axes=age_axes)

    fig.suptitle('DMPA-SC Intervention Scenarios (Dashboard)', fontsize=16, fontweight='bold', y=0.995)

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
    Baseline scenario: 2% annual increase in contraception users.
    
    Uses the change_initiation intervention with multiplicative annual growth.
    """
    intv = fp.change_initiation(
        years=[2000, 2045],
        perc=0.02,
        annual=True,
        name='baseline_growth'
    )
    
    sim = fp.Sim(pars=pars, interventions=[intv], label='Baseline')
    sim.run()
    return sim


def run_scenario_1(pars):
    """
    Scenario 1: 3-month DMPA-SC scale-up.
    
    - Introduce 3-month DMPA-SC in 2025
    - Scale from 1.5% to 5% (2025-2040) in women under 20
    - Same efficacy as injectables
    - 2x longer duration (via rel_dur_use)
    """
    # Baseline growth
    baseline_intv = fp.change_initiation(
        years=[2000, 2045],
        perc=0.02,
        annual=True,
        name='baseline_s1'
    )
    
    # Add 3-month DMPA-SC method in 2025
    # Use lognormal distribution with 2x duration via rel_dur_use
    add_method_intv = fp.add_method(
        year=2025,
        method=None,  # Will copy from source
        method_pars={
            'name': 'dmpasc3',
            'label': 'DMPA-SC 3-month',
            'csv_name': 'DMPA-SC 3-month',
            'dur_use': ss.lognorm_ex(mean=2, std=1),  # Base distribution
            'rel_dur_use': 2.0,  # 2x scaling factor
        },
        copy_from='inj',
        split_shares=0.0,  # Don't automatically split shares
        verbose=False,
        name='add_dmpasc3_s1'
    )
    
    # Age-restricted initiation with scale-up using enhanced change_initiation
    initiation_intv = fp.change_initiation(
        years=[2025, 2040],
        age_range=(0, 20),
        perc=0.015,        # 1.5% initial
        final_perc=0.05,   # 5% final (enables linear scale-up)
        perc_of_eligible=True,  # Apply to eligible women, not current users
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
    Scenario 2: 6-month DMPA-SC scale-up with switching.
    
    - Introduce both 3-month (2025) and 6-month (2030) DMPA-SC
    - 6-month scales from 1% to 5% (2030-2040) in women under 20
    - 26% of 3-month DMPA-SC users switch to 6-month
    - 20% of traditional method users switch to 6-month
    """
    # Baseline growth
    baseline_intv = fp.change_initiation(
        years=[2000, 2045],
        perc=0.02,
        annual=True,
        name='baseline_s2'
    )
    
    # Add 3-month DMPA-SC in 2025
    add_3month_intv = fp.add_method(
        year=2025,
        method=None,
        method_pars={
            'name': 'dmpasc3',
            'label': 'DMPA-SC 3-month',
            'csv_name': 'DMPA-SC 3-month',
            'dur_use': ss.lognorm_ex(mean=2, std=1),
            'rel_dur_use': 2.0,
        },
        copy_from='inj',
        split_shares=0.0,
        verbose=False,
        name='add_dmpasc3_s2'
    )
    
    # Add 6-month DMPA-SC in 2030
    add_6month_intv = fp.add_method(
        year=2030,
        method=None,
        method_pars={
            'name': 'dmpasc6',
            'label': 'DMPA-SC 6-month',
            'csv_name': 'DMPA-SC 6-month',
            'dur_use': ss.lognorm_ex(mean=2, std=1),
            'rel_dur_use': 2.5,  # Even longer than 3-month
        },
        copy_from='inj',
        split_shares=0.0,
        verbose=False,
        name='add_dmpasc6_s2'
    )
    
    # 3-month initiation
    init_3month_intv = fp.change_initiation(
        years=[2025, 2040],
        age_range=(0, 20),
        perc=0.015,
        final_perc=0.05,
        perc_of_eligible=True,
        target_method='dmpasc3',
        annual=True,
        verbose=False,
        name='init_dmpasc3_s2'
    )
    
    # 6-month initiation
    init_6month_intv = fp.change_initiation(
        years=[2030, 2040],
        age_range=(0, 20),
        perc=0.01,         # 1%
        final_perc=0.05,   # 5%
        perc_of_eligible=True,
        target_method='dmpasc6',
        annual=True,
        verbose=False,
        name='init_dmpasc6_s2'
    )
    
    # Switching from 3-month and traditional to 6-month
    switching_intv = fp.method_switching(
        year=2030,
        from_methods=['dmpasc3', 'wdraw', 'othtrad'],
        to_method='dmpasc6',
        switch_prob={'dmpasc3': 0.26, 'wdraw': 0.20, 'othtrad': 0.20},
        annual=False,
        verbose=False,
        name='switch_to_dmpasc6_s2'
    )
    
    interventions = [
        baseline_intv,
        add_3month_intv, init_3month_intv,
        add_6month_intv, init_6month_intv,
        switching_intv
    ]
    
    sim = fp.Sim(pars=pars, interventions=interventions, label='Scenario 2')
    sim.run()
    return sim


def run_scenario_3(pars):
    """
    Scenario 3: Combined scenario.
    
    This is identical to Scenario 2 (includes both 3-month and 6-month with switching).
    """
    return run_scenario_2(pars)


def run_placeholder(pars):
    """
    Placeholder scenario: Simplified single DMPA-SC implementation.
    
    - Single DMPA-SC method
    - 5% initiation in year 1 (2025), then 2% annually
    - 10% of injectable users switch
    - 5% of traditional method users switch
    - 2x longer duration
    """
    # Baseline growth
    baseline_intv = fp.change_initiation(
        years=[2000, 2045],
        perc=0.02,
        annual=True,
        name='baseline_placeholder'
    )
    
    # Add DMPA-SC method in 2025
    add_method_intv = fp.add_method(
        year=2025,
        method=None,
        method_pars={
            'name': 'dmpasc',
            'label': 'DMPA-SC',
            'csv_name': 'DMPA-SC',
            'dur_use': ss.lognorm_ex(mean=2, std=1),
            'rel_dur_use': 2.0,
        },
        copy_from='inj',
        split_shares=0.0,
        verbose=False,
        name='add_dmpasc_placeholder'
    )
    
    # Placeholder requirement: 5% initiation in the year of the intervention,
    # then 2% each year after. Since the sim runs monthly (dt=1/12 years),
    # define the "intervention year" as [2025.0, 2025.9166...] (12 monthly steps).
    init_intv_year1 = fp.change_initiation(
        years=[2025, 2025 + (11/12)],
        age_range=(0, 100),  # All ages
        perc=0.05,           # 5% annual initiation rate during intervention year
        perc_of_eligible=True,
        target_method='dmpasc',
        annual=True,
        verbose=False,
        name='init_dmpasc_placeholder_year1'
    )

    init_intv_after = fp.change_initiation(
        years=[2026, 2045],
        age_range=(0, 100),  # All ages
        perc=0.02,           # 2% annual initiation rate after intervention year
        perc_of_eligible=True,
        target_method='dmpasc',
        annual=True,
        verbose=False,
        name='init_dmpasc_placeholder_after'
    )
    
    # Switching from injectables and traditional methods
    switching_intv = fp.method_switching(
        year=2025,
        from_methods=['inj', 'wdraw', 'othtrad'],
        to_method='dmpasc',
        switch_prob={'inj': 0.10, 'wdraw': 0.05, 'othtrad': 0.05},
        annual=False,
        verbose=False,
        name='switch_to_dmpasc_placeholder'
    )
    
    interventions = [baseline_intv, add_method_intv, init_intv_year1, init_intv_after, switching_intv]
    
    sim = fp.Sim(pars=pars, interventions=interventions, label='Placeholder')
    sim.run()
    return sim


def _get_years(sim):
    """Convert Starsim time vector to numeric years for plotting."""
    return np.array([float(t) for t in sim.results.timevec.to_float()])


def plot_scenario_comparison(scenarios, pars, ax=None):
    """Plot mCPR comparison across all scenarios."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'Baseline': '#000000',
        'Scenario 1: 3-month DMPA-SC': '#e41a1c',
        'Scenario 2: 6-month DMPA-SC': '#377eb8',
        'Scenario 3: Combined': '#4daf4a',
        'Placeholder: Simple DMPA-SC': '#984ea3',
    }
    
    for label, sim in scenarios.items():
        years = _get_years(sim)
        mcpr = sim.results.contraception.mcpr * 100
        color = colors.get(label, 'gray')
        linestyle = '--' if label == 'Baseline' else '-'
        linewidth = 1.5 if label == 'Baseline' else 2
        ax.plot(years, mcpr, color=color, linestyle=linestyle, 
                linewidth=linewidth, label=label, alpha=0.9)
    
    # Mark intervention years
    ax.axvline(x=2025, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=2030, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(2025, ax.get_ylim()[1] * 0.95, '2025', ha='center', fontsize=9, color='gray')
    ax.text(2030, ax.get_ylim()[1] * 0.95, '2030', ha='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('mCPR (%)', fontsize=12)
    ax.set_title('Modern Contraceptive Prevalence Rate: Scenario Comparison', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pars['start'], pars['stop'])
    
    return ax


def plot_method_uptake(scenarios, pars, axes=None):
    """Plot DMPA-SC method uptake over time for each scenario (4 subplots)."""
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = np.array(axes).flatten()
    
    scenario_list = [
        'Scenario 1: 3-month DMPA-SC',
        'Scenario 2: 6-month DMPA-SC',
        'Scenario 3: Combined',
        'Placeholder: Simple DMPA-SC',
    ]
    
    for idx, scenario_name in enumerate(scenario_list):
        ax = axes_flat[idx]
        sim = scenarios[scenario_name]
        years = _get_years(sim)
        cm = sim.connectors.contraception
        fp_conn = sim.connectors.fp
        
        # Find DMPA-SC methods
        dmpasc_methods = {}
        for name, method in cm.methods.items():
            if 'dmpasc' in name.lower() and name != 'none':
                dmpasc_methods[name] = method
        
        # Plot each DMPA-SC method
        colors_map = {'dmpasc3': '#e41a1c', 'dmpasc6': '#377eb8', 'dmpasc': '#984ea3'}
        for name, method in dmpasc_methods.items():
            usage = fp_conn.method_mix[method.idx, :] * 100
            color = colors_map.get(name, 'gray')
            ax.plot(years, usage, color=color, linewidth=2.5, 
                   label=method.label, alpha=0.9)
        
        # Also plot standard injectable for comparison
        if 'inj' in cm.methods:
            inj_usage = fp_conn.method_mix[cm.methods['inj'].idx, :] * 100
            ax.plot(years, inj_usage, color='green', linewidth=1.5, 
                   linestyle='--', label='Standard Injectable', alpha=0.7)
        
        ax.axvline(x=2025, color='gray', linestyle=':', alpha=0.5)
        if 'Scenario 2' in scenario_name or 'Scenario 3' in scenario_name:
            ax.axvline(x=2030, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Usage (%)', fontsize=11)
        ax.set_title(scenario_name, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(pars['start'], pars['stop'])
    
    return axes


def plot_births_comparison(scenarios, pars, axes=None):
    """Plot cumulative births comparison (2 subplots)."""
    if axes is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        ax1, ax2 = axes
    
    colors = {
        'Baseline': '#000000',
        'Scenario 1: 3-month DMPA-SC': '#e41a1c',
        'Scenario 2: 6-month DMPA-SC': '#377eb8',
        'Scenario 3: Combined': '#4daf4a',
        'Placeholder: Simple DMPA-SC': '#984ea3',
    }
    
    baseline_sim = scenarios['Baseline']
    baseline_births = baseline_sim.results.fp.cum_births
    baseline_years = _get_years(baseline_sim)
    
    # Plot 1: Cumulative births
    for label, sim in scenarios.items():
        years = _get_years(sim)
        births = sim.results.fp.cum_births
        color = colors.get(label, 'gray')
        linestyle = '--' if label == 'Baseline' else '-'
        linewidth = 1.5 if label == 'Baseline' else 2
        ax1.plot(years, births, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label, alpha=0.9)
    
    ax1.axvline(x=2025, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=2030, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Cumulative Births', fontsize=12)
    ax1.set_title('Cumulative Births by Scenario', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(pars['start'], pars['stop'])
    
    # Plot 2: Births averted relative to baseline
    for label, sim in scenarios.items():
        if label == 'Baseline':
            continue
        years = _get_years(sim)
        births = sim.results.fp.cum_births
        births_averted = baseline_births - births
        color = colors.get(label, 'gray')
        ax2.plot(years, births_averted, color=color, linewidth=2, 
                label=label, alpha=0.9)
    
    ax2.axvline(x=2025, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=2030, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Births Averted (vs Baseline)', fontsize=12)
    ax2.set_title('Births Averted: Intervention vs Baseline', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(pars['start'], pars['stop'])
    
    return (ax1, ax2)


def plot_age_specific_uptake(scenarios, pars, axes=None):
    """Plot age-specific DMPA-SC uptake at final timestep (4 subplots)."""
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = np.array(axes).flatten()
    
    scenario_list = [
        'Scenario 1: 3-month DMPA-SC',
        'Scenario 2: 6-month DMPA-SC',
        'Scenario 3: Combined',
        'Placeholder: Simple DMPA-SC',
    ]
    
    age_bins = [0, 18, 20, 25, 35, 50]
    age_labels = ['<18', '18-20', '20-25', '25-35', '35-50']
    
    for idx, scenario_name in enumerate(scenario_list):
        ax = axes_flat[idx]
        sim = scenarios[scenario_name]
        ppl = sim.people
        cm = sim.connectors.contraception
        
        # Find DMPA-SC methods
        dmpasc_methods = {}
        for name, method in cm.methods.items():
            if 'dmpasc' in name.lower() and name != 'none':
                dmpasc_methods[name] = method
        
        # Calculate age-specific usage
        for name, method in dmpasc_methods.items():
            usage_by_age = []
            for i in range(len(age_bins) - 1):
                min_age = age_bins[i]
                max_age = age_bins[i + 1]
                
                # Women in this age group
                in_age_group = (ppl.alive & ppl.female & 
                               (ppl.age >= min_age) & (ppl.age < max_age))
                n_in_age = in_age_group.sum()
                
                # Women using this method in this age group
                using_method = in_age_group & (ppl.fp.method == method.idx)
                n_using = using_method.sum()
                
                pct = (n_using / n_in_age * 100) if n_in_age > 0 else 0
                usage_by_age.append(pct)
            
            # Offset bars slightly per method so multiple methods don't fully overlap
            x = np.arange(len(age_labels))
            width = 0.35
            if len(dmpasc_methods) > 1:
                # Deterministic offset based on sorted method keys
                keys_sorted = sorted(dmpasc_methods.keys())
                m_i = keys_sorted.index(name)
                offset = (m_i - (len(keys_sorted) - 1) / 2) * width
            else:
                offset = 0.0
            ax.bar(x + offset, usage_by_age, width=width, label=method.label, alpha=0.7)
        
        ax.set_xlabel('Age Group', fontsize=11)
        ax.set_ylabel('Usage (%)', fontsize=11)
        ax.set_title(f'{scenario_name}\n(Final timestep)', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(len(age_labels)))
        ax.set_xticklabels(age_labels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    return axes


def print_summary_statistics(scenarios):
    """Print summary statistics for all scenarios."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Final Timestep)")
    print("=" * 80)
    
    baseline_sim = scenarios['Baseline']
    baseline_mcpr = baseline_sim.results.contraception.mcpr[-1] * 100
    baseline_births = baseline_sim.results.fp.cum_births[-1]
    
    print(f"\n{'Scenario':<40} {'mCPR (%)':<12} {'Δ mCPR':<12} {'Births Averted':<15}")
    print("-" * 80)
    
    for label, sim in scenarios.items():
        mcpr = sim.results.contraception.mcpr[-1] * 100
        delta_mcpr = mcpr - baseline_mcpr
        births = sim.results.fp.cum_births[-1]
        births_averted = baseline_births - births
        
        print(f"{label:<40} {mcpr:>10.2f}  {delta_mcpr:>10.2f}  {births_averted:>13.0f}")
    
    print("-" * 80)
    
    # DMPA-SC uptake statistics
    print("\n" + "=" * 80)
    print("DMPA-SC METHOD UPTAKE (Final Timestep)")
    print("=" * 80)
    
    for label, sim in scenarios.items():
        if label == 'Baseline':
            continue
        
        print(f"\n{label}:")
        cm = sim.connectors.contraception
        fp_conn = sim.connectors.fp
        ppl = sim.people
        
        for name, method in cm.methods.items():
            if 'dmpasc' in name.lower() and name != 'none':
                usage_pct = fp_conn.method_mix[method.idx, -1] * 100
                n_users = (ppl.alive & (ppl.fp.method == method.idx)).sum()
                print(f"  {method.label:<30} {usage_pct:>6.2f}%  ({n_users:>5} users)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
