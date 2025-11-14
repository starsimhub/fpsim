"""
Showcase: Adding New Contraceptive Methods with Visualizations

This example demonstrates the new capability to dynamically add contraceptive 
methods during a simulation, with comprehensive plots showing:
1. Method mix changes over time
2. CPR/mCPR trends
3. Adoption of new methods
4. Comparison with baseline

Examples included:
- Example 1: Adding a single new method (MY-NEW-METHOD) in 2010
- Example 2: Adding TWO new methods at different times:
    * SC-DMPA (self-injectable) in 2010
    * Contraceptive Ring in 2015

NOTES:
- run_example can be set to 'one', 'two', or 'both' to run the corresponding example
- the figures will be saved to the figures directory
- the figures will be saved with a unique prefix
- the figures will be saved with the following format:
    - add_method_<example_number>_<example_name>.png
    - add_method_<example_number>_<example_name>.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import fpsim as fp
from plots import plot_comparison_full, plot_injectable_methods_comparison, plot_method_mix_evolution, plot_new_method_adoption, plot_method_comparison_bar, plot_cpr_comparison, plot_births_comparison, create_summary_figure, print_summary_statistics, generate_figure_prefix

# Configuration
location = 'kenya'
n_agents = 10000  # Larger population for clearer trends
start_year = 2000
end_year = 2020
intervention_year = 2010      

run_example = 'one' 

def default_pars():
    """Baseline parameters."""
    return dict(
        n_agents=n_agents,
        location=location,
        start_year=start_year,
        end_year=end_year,
    )

def run_baseline():
    """Run baseline simulation without new method."""
    print("Running baseline simulation...")
    pars = default_pars()
    sim = fp.Sim(pars=pars, label='Baseline')
    sim.run()
    return sim

def run_with_new_method():
    """Run simulation with new MY-NEW-METHOD method with comprehensive promotion."""
    print("Running intervention simulation...")
    
    pars = default_pars()
    
    # STEP 1: Define the new method with better baseline properties
    my_new_method = fp.Method(
        name='my_new_method',
        label='MY-NEW-METHOD',
        efficacy=0.80,  # Higher efficacy - well-designed product
        modern=False,
        dur_use=fp.methods.ln(6, 2.5),  # Better baseline duration
        csv_name='MY-NEW-METHOD'
    )
    # STEP 2: Define the intervention
    mod = fp.MethodIntervention(year=intervention_year, label='MY-NEW-METHOD Comprehensive Program')
    # STEP 3: Call the feature that adds the new contraceptive method to the intervention
    mod.add_method(
        method=my_new_method,     
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.40  # 40% staying probability !
    )

    # [Optional] Set the duration and efficacy of the new method
    mod.set_duration_months('my_new_method', 12)  
    # [Optional] Set the efficacy of the new method
    mod.set_efficacy('my_new_method', 0.995)

    # STEP 4: Build the intervention
    intv = mod.build()
    # STEP 5: Run the simulation
    sim = fp.Sim(pars=pars, interventions=intv, label='With MY-NEW-METHOD Program')
    sim.run()
    
    # STEP 6: Return the simulation object for further analysis (Or plot)
    return sim

def run_with_two_new_methods():
    """
    Run simulation demonstrating adding TWO new contraceptive methods.
    
    This example shows how to add multiple methods at the same time or at different
    time points. Here we add:
    1. SC-DMPA (self-injectable) in 2010 - similar to regular injectables
    2. Contraceptive Ring in 2015 - similar to pills (short-acting)
    
    This demonstrates:
    - Adding multiple methods in different intervention years
    - Methods with different characteristics (injectable vs. barrier-like)
    - Phased rollout of new products
    """
    print("Running simulation with TWO new methods...")
    
    pars = default_pars()
    
    # ========== FIRST NEW METHOD: SC-DMPA (2010) ==========
    # Define SC-DMPA - a self-injectable similar to regular injectables
    sc_dmpa = fp.Method(
        name='sc_dmpa',
        label='SC-DMPA',
        efficacy=0.94,  # Similar to regular injectables
        modern=True,
        dur_use=fp.methods.ln(4, 2.5),  # Slightly shorter than regular DMPA
        csv_name='SC-DMPA'
    )
    
    # Create first intervention for SC-DMPA
    intv1 = fp.MethodIntervention(year=2010, label='Introduce SC-DMPA')
    intv1.add_method(
        method=sc_dmpa,
        copy_from_row='inj',  # Copy switching patterns from injectables
        copy_from_col='inj',
        initial_share=0.25  # 25% staying probability
    )
    # Optional: Improve its properties after initial introduction
    intv1.set_duration_months('sc_dmpa', 6)  # Better continuation than baseline
    intv1.set_efficacy('sc_dmpa', 0.96)
    
    # ========== SECOND NEW METHOD: Contraceptive Ring (2015) ==========
    # Define contraceptive ring - a monthly vaginal ring
    ring = fp.Method(
        name='ring',
        label='Contraceptive Ring',
        efficacy=0.91,  # Similar to pills
        modern=True,
        dur_use=fp.methods.ln(12, 3),  # Moderate duration
        csv_name='Ring'
    )
    
    # Create second intervention for ring
    intv2 = fp.MethodIntervention(year=2015, label='Introduce Contraceptive Ring')
    intv2.add_method(
        method=ring,
        copy_from_row='pill',  # Copy switching patterns from pills
        copy_from_col='pill',  # (both user-controlled, short-acting)
        initial_share=0.20  # 20% staying probability
    )
    # Optional: Set specific properties
    intv2.set_duration_months('ring', 18)  # Better continuation than pills
    intv2.set_efficacy('ring', 0.93)
    
    # Build both interventions
    intervention_list = [intv1.build(), intv2.build()]
    
    # Run simulation with BOTH interventions
    sim = fp.Sim(
        pars=pars, 
        interventions=intervention_list,
        label='With Two New Methods (SC-DMPA + Ring)'
    )
    sim.run()
    
    print(f"  ✓ SC-DMPA introduced in 2010")
    print(f"  ✓ Contraceptive Ring introduced in 2015")
    
    return sim


if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("SHOWCASE: Adding New Contraceptive Methods to FPsim")
    print("="*70)
    
    example_choice = run_example
    
    # region: ========== EXAMPLE 1: Single New Method ==========
    if example_choice in ['one', '1', 'single']:
        print(f"\nSimulation Setup:")
        print(f"  Location: {location.title()} Population: {n_agents:,} agents")
        print(f"  Period: {start_year}-{end_year}")
        print(f"  Intervention: Comprehensive MY-NEW-METHOD Program ({intervention_year})")
        print(f"  Figures will be saved to: {figures_dir}")
        
        # Run simulations
        s0 = run_baseline()
        s1 = run_with_new_method()
        
        # Generate unique prefix for this run
        prefix = generate_figure_prefix(s1, s0, prefix_type='method')
        print(f"  Figure prefix: '{prefix}'")
        
        plot_comparison_full(s0, s1)    

        # Create all plots - focus on method-specific metrics
        plot_injectable_methods_comparison(
            s0,s1, start_year, end_year,
            location, intervention_year=intervention_year,
            save_path=figures_dir / f'{prefix}add_method_injectables.png',
        )    
        plot_cpr_comparison(s0, s1, start_year, end_year, intervention_year, location, 
                          save_path=figures_dir / f'{prefix}add_method_cpr.png')
        plot_method_mix_evolution(s1, start_year, end_year, intervention_year, location, 
                                 save_path=figures_dir / f'{prefix}add_method_mix.png')
        plot_new_method_adoption(s1, start_year, end_year, intervention_year, location, 
                                save_path=figures_dir / f'{prefix}add_method_adoption.png')
        plot_method_comparison_bar(s0, s1, start_year, end_year, intervention_year, location, 
                                  save_path=figures_dir / f'{prefix}add_method_bar.png')
        plot_births_comparison(s0, s1, start_year, end_year, intervention_year, location, 
                             save_path=figures_dir / f'{prefix}add_method_births.png')
        create_summary_figure(s0, s1, start_year, end_year, intervention_year, location, 
                            save_path=figures_dir / f'{prefix}add_method_summary.png')
        
        # Print statistics
        print_summary_statistics(s0, s1, start_year, end_year, intervention_year)
        
        print(f"\n{'='*70}")
        print(f"All figures saved to: {figures_dir}")
        print(f"  with prefix: '{prefix}'")
        print(f"{'='*70}")
    #endregion 
    
    # region: ========== EXAMPLE 2: Two New Methods ==========
    elif example_choice in ['two', '2', 'multiple']:
        print(f"\nSimulation Setup:")
        print(f"  Location: {location.title()} Population: {n_agents:,} agents")
        print(f"  Period: {start_year}-{end_year}")
        print(f"  Interventions:")
        print(f"    - SC-DMPA introduced in 2010")
        print(f"    - Contraceptive Ring introduced in 2015")
        print(f"  Figures will be saved to: {figures_dir}")
        
        # Run simulations
        s0 = run_baseline()
        s1 = run_with_two_new_methods()
        
        # Generate unique prefix for this run
        prefix = generate_figure_prefix(s1, s0, prefix_type='method')
        print(f"  Figure prefix: '{prefix}'")
        
        # Create comparison plots
        plot_comparison_full(s0, s1, title=f'{location.title()}: Baseline vs Two New Methods')
        plot_cpr_comparison(s0, s1, start_year, end_year, 2010, location, 
                          save_path=figures_dir / f'{prefix}add_method_cpr.png')
        plot_method_mix_evolution(s1, start_year, end_year, 2010, location, 
                                 save_path=figures_dir / f'{prefix}add_method_mix.png')
        plot_method_comparison_bar(s0, s1, start_year, end_year, 2010, location, 
                                  save_path=figures_dir / f'{prefix}add_method_bar.png')
        plot_births_comparison(s0, s1, start_year, end_year, 2010, location, 
                             save_path=figures_dir / f'{prefix}add_method_births.png')
        create_summary_figure(s0, s1, start_year, end_year, 2010, location, 
                            save_path=figures_dir / f'{prefix}add_method_summary.png')
        
        # Print statistics
        print_summary_statistics(s0, s1, start_year, end_year, 2010)
        
        print(f"\n{'='*70}")
        print(f"All figures saved to: {figures_dir}")
        print(f"  with prefix: '{prefix}'")
        print(f"To see adoption of individual methods, check the method mix plots.")
        print(f"{'='*70}")
    # endregion: ========== EXAMPLE 2: Two New Methods ==========
    
    # region: ========== EXAMPLE 3: Run Both Examples ==========
    elif example_choice in ['both', 'all']:
        print(f"\nRunning BOTH examples...\n")
        
        print("--- EXAMPLE 1: Single New Method ---")
        s0 = run_baseline()
        s1 = run_with_new_method()
        
        # Generate prefix for single method
        prefix1 = generate_figure_prefix(s1, s0, prefix_type='method')
        print(f"  Single method prefix: '{prefix1}'")
        
        print("\n--- EXAMPLE 2: Two New Methods ---")
        two_methods_sim = run_with_two_new_methods()
        
        # Generate prefix for two methods
        prefix2 = generate_figure_prefix(two_methods_sim, s0, prefix_type='method')
        print(f"  Two methods prefix: '{prefix2}'")
        
        print(f"\nGenerating comparison plots...")
        
        # Single method plots
        plot_comparison_full(s0, s1, 
                           title=f'{location.title()}: Baseline vs Single New Method')
        create_summary_figure(s0, s1, start_year, end_year, 
                            intervention_year, location, 
                            save_path=figures_dir / f'{prefix1}add_method_summary.png')
        
        # Two methods plots
        plot_comparison_full(s0, two_methods_sim, 
                           title=f'{location.title()}: Baseline vs Two New Methods')
        create_summary_figure(s0, two_methods_sim, start_year, end_year, 
                            2010, location, 
                            save_path=figures_dir / f'{prefix2}add_method_summary.png')
        
        print(f"\n{'='*70}")
        print(f"All figures saved to: {figures_dir}")
        print(f"  Single method figures: '{prefix1}...'")
        print(f"  Two methods figures: '{prefix2}...'")
        print(f"{'='*70}")
    # endregion 
    else:
        print(f"\nUsage: python example_add_new_method.py [one|two|both]")
        print(f"  one  - Run single new method example (default)")
        print(f"  two  - Run two new methods example")
        print(f"  both - Run both examples for comparison")
        print(f"{'='*70}")
    

