"""
Showcase: Adding New Contraceptive Methods with Visualizations

This example demonstrates the new capability to dynamically add contraceptive 
methods during a simulation, with comprehensive plots showing:
1. Method mix changes over time
2. CPR/mCPR trends
3. Adoption of new methods
4. Comparison with baseline
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import fpsim as fp
from plots import plot_comparison_full, plot_injectable_methods_comparison, plot_method_mix_evolution, plot_new_method_adoption, plot_method_comparison_bar, plot_births_comparison, create_summary_figure, print_summary_statistics

# Configuration
location = 'kenya'
n_agents = 10000  # Larger population for clearer trends
start_year = 2000
end_year = 2020
intervention_year = 2010      

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


if __name__ == '__main__':
    import os
    from pathlib import Path
    
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("SHOWCASE: Adding New Contraceptive Methods to FPsim")
    print("="*70)
    print(f"\nSimulation Setup:")
    print(f"  Location: {location.title()} Population: {n_agents:,} agents")
    print(f"  Period: {start_year}-{end_year}")
    print(f"  Intervention: Comprehensive MY-NEW-METHOD Program ({intervention_year})")
    print(f"  Figures will be saved to: {figures_dir}")
    
    # Run simulations
    baseline_sim = run_baseline()
    intervention_sim = run_with_new_method()
    plot_comparison_full(baseline_sim, intervention_sim)    

    # Create all plots - focus on method-specific metrics
    plot_injectable_methods_comparison(
        baseline_sim,intervention_sim, start_year, end_year,
        location, intervention_year=intervention_year,
        save_path=figures_dir / 'add_method_injectables.png',
    )    
    plot_method_mix_evolution(intervention_sim, start_year, end_year, location, save_path=figures_dir / 'add_method_mix.png')
    plot_new_method_adoption(intervention_sim, start_year, end_year, location, save_path=figures_dir / 'add_method_adoption.png')
    plot_method_comparison_bar(baseline_sim, intervention_sim, start_year, end_year, location, save_path=figures_dir / 'add_method_bar.png')
    plot_births_comparison(baseline_sim, intervention_sim, start_year, end_year, location, save_path=figures_dir / 'add_method_births.png')
    create_summary_figure(baseline_sim, intervention_sim, start_year, end_year, location, save_path=figures_dir / 'add_method_summary.png')
    
    # Print statistics
    print_summary_statistics(baseline_sim, intervention_sim, start_year, end_year, location )
    
    print(f"\n{'='*70}")
    print(f"All figures saved to: {figures_dir}")
    print(f"{'='*70}")
    

