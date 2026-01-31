"""
Example script demonstrating all scenarios achievable with the add_method intervention.

This script shows six distinct use cases:
0. Basic introduction - Simplest example: new method + market share (core concept)
1. Brand new method - Fully custom Method object
2. Quick clone - Simple copy of existing method
3. Clone + override - Copy with property modifications
4. Market splitting - New method takes share from existing
5. Custom duration - Different duration-of-use distributions

Each scenario is implemented and can be run independently or together.
"""

import fpsim as fp
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Common simulation parameters
# =============================================================================
PARS = dict(
    n_agents=2000,
    start=2000,
    stop=2020,
    location='kenya',
    verbose=0,
)
INTRODUCTION_YEAR = 2010


# =============================================================================
# SCENARIO 0: Basic Introduction (Core Concept)
# =============================================================================
def scenario_basic_introduction():
    """
    The simplest example of introducing a new contraceptive method.
    
    This is the "hello world" of add_method - it demonstrates the core concept:
    1. Define a new method with basic properties
    2. Introduce it at a specific year
    3. Copy switching behavior from an existing method
    4. Optionally take market share from that method
    
    Use case: A new injectable becomes available that competes with existing
    injectables, taking 50% of their market share.
    """
    print("\n" + "="*70)
    print("SCENARIO 0: Basic Introduction (Core Concept)")
    print("="*70)
    
    # Step 1: Define the new method
    new_injectable = fp.Method(
        name='new_inj',
        label='New Injectable',
        efficacy=0.995,
        modern=True,
        dur_use=ss.lognorm_ex(mean=3, std=1.5),
    )
    
    # Step 2: Create the intervention
    intv = fp.add_method(
        year=INTRODUCTION_YEAR,       # When to introduce
        method=new_injectable,        # The new method
        copy_from='inj',              # Copy switching behavior from injectables
        split_shares=0.50,            # Take 50% of injectable market share
        verbose=True,
    )
    
    # Step 3: Run simulations
    sim_baseline = fp.Sim(pars=PARS, label='Baseline').run()
    sim_with_method = fp.Sim(pars=PARS, interventions=[intv], label='With New Injectable').run()
    
    # Report results
    mcpr_baseline = sim_baseline.results.contraception.mcpr[-1] * 100
    mcpr_new = sim_with_method.results.contraception.mcpr[-1] * 100
    
    print(f"\nResults:")
    print(f"  Baseline mCPR: {mcpr_baseline:.1f}%")
    print(f"  With new method mCPR: {mcpr_new:.1f}%")
    print(f"  Change: {mcpr_new - mcpr_baseline:+.1f} percentage points")
    
    return sim_baseline, sim_with_method, 'new_inj'


# =============================================================================
# SCENARIO 1: Brand New Method
# =============================================================================
def scenario_brand_new_method():
    """
    Create a completely new contraceptive method with all custom parameters.
    
    Use case: Modeling the introduction of a novel contraceptive technology
    (e.g., a new long-acting injectable, hormonal gel, or contraceptive ring).
    
    Parameters used:
        - method: ✓ (required - full Method object)
        - method_pars: optional (not used here)
        - copy_from: required (for switching behavior)
        - split_shares: optional (not used here)
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Brand New Method")
    print("="*70)
    
    # Define a completely new contraceptive method
    new_method = fp.Method(
        name='hormonal_ring',
        label='Hormonal Ring',
        csv_name='Hormonal Ring',
        efficacy=0.991,  # 99.1% effective
        modern=True,
        dur_use=ss.lognorm_ex(mean=1, std=0.5),  # ~1 year average use
    )
    
    # Create intervention - copies switching behavior from IUD
    intv = fp.add_method(
        year=INTRODUCTION_YEAR,
        method=new_method,
        copy_from='iud',  # Similar user profile to IUD users
        verbose=True,
    )
    
    # Run simulation
    sim_baseline = fp.Sim(pars=PARS, label='Baseline').run()
    sim_with_method = fp.Sim(pars=PARS, interventions=[intv], label='With Hormonal Ring').run()
    
    # Report results
    mcpr_baseline = sim_baseline.results.contraception.mcpr[-1] * 100
    mcpr_new = sim_with_method.results.contraception.mcpr[-1] * 100
    
    print(f"\nResults:")
    print(f"  Baseline mCPR: {mcpr_baseline:.1f}%")
    print(f"  With new method mCPR: {mcpr_new:.1f}%")
    print(f"  Change: {mcpr_new - mcpr_baseline:+.1f} percentage points")
    
    return sim_baseline, sim_with_method, 'hormonal_ring'


# =============================================================================
# SCENARIO 2: Quick Clone
# =============================================================================
def scenario_quick_clone():
    """
    Create a copy of an existing method with minimal configuration.
    
    Use case: Setting up a baseline for further modifications, or modeling
    a generic version of an existing contraceptive becoming available.
    
    Parameters used:
        - method: None (will clone from source)
        - method_pars: ✓ (must provide fresh dur_use to avoid distribution state issues)
        - copy_from: required (source method to clone)
        - split_shares: optional (not used here)
    
    Note: The cloned method will be named '{source}_copy' automatically unless
          overridden in method_pars.
    Note: Always provide a fresh dur_use distribution to avoid state issues
          from cloned distributions.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Quick Clone")
    print("="*70)
    
    # Clone implants with a fresh duration distribution
    # The cloned method inherits efficacy, modern flag, etc. from source
    intv = fp.add_method(
        year=INTRODUCTION_YEAR,
        copy_from='impl',  # Clone implants
        method_pars={
            'dur_use': ss.lognorm_ex(mean=3, std=1),  # Fresh distribution (~3 years)
        },
        verbose=True,
    )
    
    # Run simulation
    sim_baseline = fp.Sim(pars=PARS, label='Baseline').run()
    sim_with_clone = fp.Sim(pars=PARS, interventions=[intv], label='With Implant Clone').run()
    
    # Report results
    mcpr_baseline = sim_baseline.results.contraception.mcpr[-1] * 100
    mcpr_new = sim_with_clone.results.contraception.mcpr[-1] * 100
    
    print(f"\nResults:")
    print(f"  Baseline mCPR: {mcpr_baseline:.1f}%")
    print(f"  With cloned method mCPR: {mcpr_new:.1f}%")
    print(f"  Change: {mcpr_new - mcpr_baseline:+.1f} percentage points")
    
    return sim_baseline, sim_with_clone, 'impl_copy'


# =============================================================================
# SCENARIO 3: Clone + Override
# =============================================================================
def scenario_clone_override():
    """
    Clone an existing method and override specific properties.
    
    Use case: Modeling an improved version of an existing method (e.g., 
    a new formulation of injectables with better efficacy, or a longer-lasting
    implant variant).
    
    Parameters used:
        - method: None (will clone from source)
        - method_pars: ✓ (dictionary of properties to override)
        - copy_from: required (source method to clone)
        - split_shares: optional (not used here)
    
    Note: When cloning methods with callback-based dur_use, provide explicit dur_use.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Clone + Override")
    print("="*70)
    
    # Clone injectables but improve efficacy, rename, and set explicit duration
    intv = fp.add_method(
        year=INTRODUCTION_YEAR,
        copy_from='inj',  # Start with injectables
        method_pars={
            'name': 'inj_improved',
            'label': 'Improved Injectable',
            'efficacy': 0.998,  # Improved from ~0.97 to 0.998
            'dur_use': ss.lognorm_ex(mean=0.5, std=0.2),  # ~6 months average use
        },
        verbose=True,
    )
    
    # Run simulation
    sim_baseline = fp.Sim(pars=PARS, label='Baseline').run()
    sim_with_improved = fp.Sim(pars=PARS, interventions=[intv], label='With Improved Injectable').run()
    
    # Report results
    mcpr_baseline = sim_baseline.results.contraception.mcpr[-1] * 100
    mcpr_new = sim_with_improved.results.contraception.mcpr[-1] * 100
    
    print(f"\nResults:")
    print(f"  Baseline mCPR: {mcpr_baseline:.1f}%")
    print(f"  With improved method mCPR: {mcpr_new:.1f}%")
    print(f"  Change: {mcpr_new - mcpr_baseline:+.1f} percentage points")
    
    return sim_baseline, sim_with_improved, 'inj_improved'


# =============================================================================
# SCENARIO 4: Market Splitting
# =============================================================================
def scenario_market_splitting():
    """
    Introduce a new method that takes market share from an existing method.
    
    Use case: Modeling the introduction of a self-injectable (DMPA-SC) that
    draws users away from clinic-administered injectables, or a new implant
    variant that competes with existing implants.
    
    Parameters used:
        - method: optional (can provide full Method or clone)
        - method_pars: optional (can override properties)
        - copy_from: required (source method for switching AND market share)
        - split_shares: ✓ (fraction of source method users who switch)
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Market Splitting")
    print("="*70)
    
    # Define DMPA-SC (self-injectable) that takes 40% share from injectables
    dmpasc = fp.Method(
        name='dmpasc',
        label='DMPA-SC (Self-Injectable)',
        csv_name='DMPA-SC',
        efficacy=0.997,
        modern=True,
        dur_use=ss.lognorm_ex(mean=0.25, std=0.1),  # 3-month cycle
    )
    
    intv = fp.add_method(
        year=INTRODUCTION_YEAR,
        method=dmpasc,
        copy_from='inj',
        split_shares=0.40,  # 40% of injectable users switch to DMPA-SC
        verbose=True,
    )
    
    # Run simulation
    sim_baseline = fp.Sim(pars=PARS, label='Baseline').run()
    sim_with_dmpasc = fp.Sim(pars=PARS, interventions=[intv], label='With DMPA-SC').run()
    
    # Report results
    mcpr_baseline = sim_baseline.results.contraception.mcpr[-1] * 100
    mcpr_new = sim_with_dmpasc.results.contraception.mcpr[-1] * 100
    
    # Check method-specific usage
    cm = sim_with_dmpasc.connectors.contraception
    fp_mod = sim_with_dmpasc.connectors.fp
    
    inj_idx = cm.methods['inj'].idx
    dmpasc_idx = cm.methods['dmpasc'].idx
    
    inj_usage = fp_mod.method_mix[inj_idx, -1] * 100
    dmpasc_usage = fp_mod.method_mix[dmpasc_idx, -1] * 100
    
    print(f"\nResults:")
    print(f"  Baseline mCPR: {mcpr_baseline:.1f}%")
    print(f"  With DMPA-SC mCPR: {mcpr_new:.1f}%")
    print(f"  Change: {mcpr_new - mcpr_baseline:+.1f} percentage points")
    print(f"\n  Method-specific usage (end of simulation):")
    print(f"    Standard Injectable: {inj_usage:.1f}%")
    print(f"    DMPA-SC: {dmpasc_usage:.1f}%")
    
    return sim_baseline, sim_with_dmpasc, 'dmpasc'


# =============================================================================
# SCENARIO 5: Custom Duration Distribution
# =============================================================================
def scenario_custom_duration():
    """
    Create methods with different duration-of-use distributions.
    
    Use case: Exploring how different continuation patterns affect 
    contraceptive prevalence and demographic outcomes. Different distributions
    model different user retention scenarios.
    
    Parameters used:
        - method: optional (can provide full Method)
        - method_pars: ✓ (must include dur_use)
        - copy_from: required (for switching behavior)
        - split_shares: optional (not used here)
    
    Duration distributions compared:
        - Lognormal: Right-skewed, most users discontinue early
        - Gamma: Flexible shape, moderate skew
        - Weibull: Models "wear-out" or increasing discontinuation
        - Exponential: Constant discontinuation rate (memoryless)
    """
    print("\n" + "="*70)
    print("SCENARIO 5: Custom Duration Distributions")
    print("="*70)
    
    # Define different duration distributions
    duration_configs = {
        'Lognormal': {
            'name': 'impl_lognorm',
            'label': 'Implant (Lognormal)',
            'dur_use': ss.lognorm_ex(mean=3, std=1.5),
        },
        'Gamma': {
            'name': 'impl_gamma',
            'label': 'Implant (Gamma)',
            'dur_use': ss.gamma(a=4, scale=0.75),  # mean=3
        },
        'Weibull': {
            'name': 'impl_weibull',
            'label': 'Implant (Weibull)',
            'dur_use': ss.weibull(c=2.5, scale=3.4),  # ~mean 3
        },
        'Exponential': {
            'name': 'impl_exp',
            'label': 'Implant (Exponential)',
            'dur_use': ss.expon(scale=3),  # mean=3
        },
    }
    
    # Run baseline
    sim_baseline = fp.Sim(pars=PARS, label='Baseline').run()
    
    # Run each duration variant
    sims = {'Baseline': sim_baseline}
    
    for dist_name, config in duration_configs.items():
        print(f"\n  Running {dist_name} distribution...")
        
        intv = fp.add_method(
            year=INTRODUCTION_YEAR,
            copy_from='impl',
            method_pars=config,
            verbose=False,
        )
        
        sim = fp.Sim(pars=PARS, interventions=[intv], label=dist_name).run()
        sims[dist_name] = sim
    
    # Report results
    print(f"\nResults:")
    mcpr_baseline = sim_baseline.results.contraception.mcpr[-1] * 100
    print(f"  Baseline mCPR: {mcpr_baseline:.1f}%")
    
    for dist_name, sim in sims.items():
        if dist_name != 'Baseline':
            mcpr = sim.results.contraception.mcpr[-1] * 100
            diff = mcpr - mcpr_baseline
            print(f"  {dist_name:12s} mCPR: {mcpr:.1f}% ({diff:+.1f}pp)")
    
    return sims


# =============================================================================
# Visualization
# =============================================================================
def plot_scenario_comparison(sim_baseline, sim_with_method, new_method_name, scenario_title):
    """Plot comparison between baseline and intervention scenario."""
    
    def get_years(sim):
        return np.array([float(t) for t in sim.results.timevec.to_float()])
    
    years = get_years(sim_with_method)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Scenario: {scenario_title}', fontsize=14, fontweight='bold')
    
    # Plot 1: mCPR comparison
    ax = axes[0]
    mcpr_baseline = sim_baseline.results.contraception.mcpr * 100
    mcpr_new = sim_with_method.results.contraception.mcpr * 100
    ax.plot(years, mcpr_baseline, 'b-', linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(years, mcpr_new, 'r-', linewidth=2, label='With New Method', alpha=0.8)
    ax.axvline(x=INTRODUCTION_YEAR, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('mCPR (%)')
    ax.set_title('Modern CPR Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: New method usage
    ax = axes[1]
    cm = sim_with_method.connectors.contraception
    fp_mod = sim_with_method.connectors.fp
    
    if new_method_name in cm.methods:
        method_idx = cm.methods[new_method_name].idx
        usage = fp_mod.method_mix[method_idx, :] * 100
        ax.plot(years, usage, 'g-', linewidth=2, label=new_method_name)
        ax.axvline(x=INTRODUCTION_YEAR, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Usage (%)')
    ax.set_title(f'New Method Usage: {new_method_name}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative births
    ax = axes[2]
    births_baseline = sim_baseline.results.fp.cum_births
    births_new = sim_with_method.results.fp.cum_births
    ax.plot(years, births_baseline, 'b-', linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(years, births_new, 'r-', linewidth=2, label='With New Method', alpha=0.8)
    ax.axvline(x=INTRODUCTION_YEAR, color='green', linestyle='--', alpha=0.7)
    diff = births_baseline[-1] - births_new[-1]
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Births')
    ax.set_title(f'Births (Averted: {diff:.0f})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_duration_comparison(sims):
    """Plot comparison of different duration distributions."""
    
    def get_years(sim):
        return np.array([float(t) for t in sim.results.timevec.to_float()])
    
    colors = {
        'Baseline': 'black',
        'Lognormal': '#e41a1c',
        'Gamma': '#377eb8',
        'Weibull': '#4daf4a',
        'Exponential': '#984ea3',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Scenario 5: Custom Duration Distributions', fontsize=14, fontweight='bold')
    
    # Plot 1: mCPR comparison
    ax = axes[0]
    for label, sim in sims.items():
        years = get_years(sim)
        mcpr = sim.results.contraception.mcpr * 100
        linestyle = '--' if label == 'Baseline' else '-'
        ax.plot(years, mcpr, color=colors[label], linestyle=linestyle, 
                linewidth=2, label=label, alpha=0.8)
    ax.axvline(x=INTRODUCTION_YEAR, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('mCPR (%)')
    ax.set_title('Modern CPR by Duration Distribution')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final mCPR bar chart
    ax = axes[1]
    baseline_mcpr = sims['Baseline'].results.contraception.mcpr[-1] * 100
    diffs = {}
    for label, sim in sims.items():
        if label != 'Baseline':
            mcpr = sim.results.contraception.mcpr[-1] * 100
            diffs[label] = mcpr - baseline_mcpr
    
    bar_colors = [colors[k] for k in diffs.keys()]
    bars = ax.bar(diffs.keys(), diffs.values(), color=bar_colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('mCPR Change from Baseline (pp)')
    ax.set_title('Impact by Duration Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, diffs.values()):
        ypos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.2f}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main execution
# =============================================================================
def main():
    """Run all scenarios and generate visualizations."""
    
    print("\n" + "#"*70)
    print("# ADD_METHOD INTERVENTION: ALL SCENARIOS")
    print("#"*70)
    
    do_plot = True
    do_save = True
    do_show = True
    
    # Run Scenario 0: Basic Introduction (Core Concept)
    sim_base0, sim_new0, method0 = scenario_basic_introduction()
    
    # Run Scenario 1: Brand New Method
    sim_base1, sim_new1, method1 = scenario_brand_new_method()
    
    # Run Scenario 2: Quick Clone
    sim_base2, sim_new2, method2 = scenario_quick_clone()
    
    # Run Scenario 3: Clone + Override
    sim_base3, sim_new3, method3 = scenario_clone_override()
    
    # Run Scenario 4: Market Splitting
    sim_base4, sim_new4, method4 = scenario_market_splitting()
    
    # Run Scenario 5: Custom Duration
    sims5 = scenario_custom_duration()
    
    # Generate plots
    if do_plot:
        print("\n" + "="*70)
        print("Generating plots...")
        print("="*70)
        
        fig0 = plot_scenario_comparison(sim_base0, sim_new0, method0, 
                                        "0: Basic Introduction (New Injectable)")
        fig1 = plot_scenario_comparison(sim_base1, sim_new1, method1, 
                                        "1: Brand New Method (Hormonal Ring)")
        fig2 = plot_scenario_comparison(sim_base2, sim_new2, method2, 
                                        "2: Quick Clone (Implant Copy)")
        fig3 = plot_scenario_comparison(sim_base3, sim_new3, method3, 
                                        "3: Clone + Override (Improved Injectable)")
        fig4 = plot_scenario_comparison(sim_base4, sim_new4, method4, 
                                        "4: Market Splitting (DMPA-SC)")
        fig5 = plot_duration_comparison(sims5)
        
        if do_save:
            fig0.savefig('scenario0_basic_introduction.png', dpi=150, bbox_inches='tight')
            fig1.savefig('scenario1_brand_new_method.png', dpi=150, bbox_inches='tight')
            fig2.savefig('scenario2_quick_clone.png', dpi=150, bbox_inches='tight')
            fig3.savefig('scenario3_clone_override.png', dpi=150, bbox_inches='tight')
            fig4.savefig('scenario4_market_splitting.png', dpi=150, bbox_inches='tight')
            fig5.savefig('scenario5_custom_duration.png', dpi=150, bbox_inches='tight')
            print("Plots saved!")
        
        if do_show:
            plt.show()
    
    print("\n" + "#"*70)
    print("# ALL SCENARIOS COMPLETE")
    print("#"*70)
    
    return {
        'scenario0': (sim_base0, sim_new0),
        'scenario1': (sim_base1, sim_new1),
        'scenario2': (sim_base2, sim_new2),
        'scenario3': (sim_base3, sim_new3),
        'scenario4': (sim_base4, sim_new4),
        'scenario5': sims5,
    }


if __name__ == '__main__':
    results = main()
