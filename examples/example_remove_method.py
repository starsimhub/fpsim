"""
Example demonstrating the remove_method functionality in FPsim.

This example shows how to:
1. Run a baseline simulation with all methods available
2. Remove a specific method (implants) during the simulation
3. Compare the method mix before and after removal
4. Generate comprehensive visualizations of method removal impact
"""

import numpy as np
import sciris as sc
import fpsim as fp
from pathlib import Path
from plots import (plot_method_removal_impact, plot_method_redistribution, 
                   plot_removed_method_timeline, create_removal_summary_figure,
                   plot_comparison_full)

# Set random seed for reproducibility
np.random.seed(1)

# Basic parameters
pars = dict(
    n_agents=5_000,
    start_year=2000,
    end_year=2030,
    verbose=0,
)

def run_baseline():
    """Run a baseline simulation with all methods available."""
    print("=" * 60)
    print("Running BASELINE simulation (all methods available)")
    print("=" * 60)
    
    sim = fp.Sim(pars=pars, location='senegal')
    sim.run()
    
    # Print final method mix
    print("\nFinal method mix (2030):")
    cm = sim.connectors.contraception
    for method_name, method in cm.methods.items():
        if method_name == 'none':
            continue
        n_users = np.sum(sim.people.fp.method == method.idx)
        pct = 100 * n_users / len(sim.people)
        print(f"  {method.label:20s}: {n_users:4d} users ({pct:5.2f}%)")
    
    return sim

def run_with_removal():
    """Run a simulation that removes implants in 2015."""
    print("\n" + "=" * 60)
    print("Running simulation with IMPLANT REMOVAL in 2015")
    print("  (Users reassigned to Injectables)")
    print("=" * 60)
    
    # Create intervention to remove implants
    intv = fp.MethodIntervention(year=2015, label='Remove Implants')
    intv.remove_method('impl', reassign_to='inj')
    
    # Preview the intervention
    print("\nIntervention preview:")
    preview = intv.preview()
    for key, value in preview.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # Build and run
    intervention = intv.build()
    sim = fp.Sim(pars=pars, location='senegal', interventions=intervention)
    sim.run()
    
    # Print method mix at key timepoints
    print("\nMethod mix at year 2030:")
    cm = sim.connectors.contraception
    
    # Get the methods and their names (excluding 'none')
    for method_name, method in cm.methods.items():
        if method_name == 'none':
            continue
        n_users = np.sum(sim.people.fp.method == method.idx)
        pct = 100 * n_users / len(sim.people)
        print(f"  {method.label:20s}: {n_users:4d} users ({pct:5.2f}%)")
    
    print(f"\nNote: Implants were removed in 2015")
    print(f"Remaining methods: {[m for m in cm.methods.keys() if m != 'none']}")
    
    return sim

def run_multiple_removals():
    """Run a simulation that removes a less-used method."""
    print("\n" + "=" * 60)
    print("Running simulation with METHOD REMOVAL")
    print("  - Remove Other Modern in 2015 → reassign to Pill")
    print("=" * 60)
    
    # Create intervention to remove method
    intv = fp.MethodIntervention(year=2015, label='Remove Other Modern')
    intv.remove_method('othmod', reassign_to='pill')
    
    # Build and run
    intervention = intv.build()
    sim = fp.Sim(pars=pars, location='senegal', interventions=intervention)
    sim.run()
    
    print("\nFinal method mix (2030):")
    cm = sim.connectors.contraception
    for method_name, method in cm.methods.items():
        if method_name == 'none':
            continue
        n_users = np.sum(sim.people.fp.method == method.idx)
        pct = 100 * n_users / len(sim.people)
        print(f"  {method.label:20s}: {n_users:4d} users ({pct:5.2f}%)")
    
    print(f"\nTotal methods available: {cm.n_methods}")
    print(f"Methods still in simulation: {list(cm.methods.keys())}")
    
    return sim

def run_direct_removal():
    """Demonstrate direct removal via connector (without intervention)."""
    print("\n" + "=" * 60)
    print("DIRECT REMOVAL: Calling remove_method() directly")
    print("  (This removes the method immediately at sim initialization)")
    print("=" * 60)
    
    sim = fp.Sim(pars=pars, location='senegal')
    sim.init()
    
    # Get initial method count
    cm = sim.connectors.contraception
    initial_methods = len(cm.methods)
    print(f"\nInitial methods: {initial_methods}")
    print(f"Methods: {list(cm.methods.keys())}")
    
    # Remove IUDs directly
    print("\nRemoving IUDs (reassigning to Injectables)...")
    cm.remove_method('IUDs', reassign_to='Injectables')
    
    # Check after removal
    final_methods = len(cm.methods)
    print(f"\nAfter removal: {final_methods} methods")
    print(f"Methods: {list(cm.methods.keys())}")
    print(f"Successfully removed {initial_methods - final_methods} method(s)")
    
    # Now run the simulation
    sim.run()
    
    print("\nFinal method mix (2030):")
    for method_name, method in cm.methods.items():
        if method_name == 'none':
            continue
        n_users = np.sum(sim.people.fp.method == method.idx)
        pct = 100 * n_users / len(sim.people)
        print(f"  {method.label:20s}: {n_users:4d} users ({pct:5.2f}%)")
    
    return sim

if __name__ == '__main__':
    # Run all examples
    print("\n" + "=" * 60)
    print("FPSIM REMOVE_METHOD EXAMPLES")
    print("=" * 60)
    
    # Example 1: Baseline
    baseline_sim = run_baseline()
    
    # Example 2: Single removal via intervention
    removal_sim = run_with_removal()
    
    # Example 3: Direct removal
    direct_removal_sim = run_direct_removal()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Methods can be removed during simulation via interventions")
    print("2. Users are automatically reassigned to specified alternative methods")
    print("3. Methods can be removed via MethodIntervention wrapper or directly")
    print("4. Switching matrices are automatically adjusted when methods are removed")
    print("5. Method indices are automatically re-mapped after removal")
    print("\nNote: Some methods with complex switching patterns may require")
    print("      additional calibration data after removal for optimal results.")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Get simulation parameters
    location = 'senegal'
    start_year = pars['start_year']
    end_year = pars['end_year']
    removal_year = 2015  # Year when implants were removed
    
    # Generate comprehensive plots
    print("\nGenerating comparison plots...")
    plot_comparison_full(baseline_sim, removal_sim, show_figure=False, save_figure=True,
                        filename=f"{location}_remove_method_comparison_full.png",
                        title=f"Method Removal Impact in {location.title()}")
    
    print("Generating removal impact plots...")
    plot_method_removal_impact(baseline_sim, removal_sim, start_year, end_year, 
                              removal_year, location,
                              save_path=figures_dir / f'{location}_removal_impact.png')
    
    print("Generating redistribution analysis...")
    plot_method_redistribution(baseline_sim, removal_sim, removal_year, location,
                              save_path=figures_dir / f'{location}_removal_redistribution.png')
    
    print("Generating timeline visualization...")
    plot_removed_method_timeline(baseline_sim, removal_sim, start_year, end_year,
                                removal_year, location,
                                save_path=figures_dir / f'{location}_removal_timeline.png')
    
    print("Generating comprehensive summary figure...")
    create_removal_summary_figure(baseline_sim, removal_sim, start_year, end_year,
                                 removal_year, location,
                                 save_path=figures_dir / f'{location}_removal_summary.png')
    
    print(f"\n✓ All visualizations saved to: {figures_dir}/")
    print(f"  - {location}_remove_method_comparison_full.png")
    print(f"  - {location}_removal_impact.png")
    print(f"  - {location}_removal_redistribution.png")
    print(f"  - {location}_removal_timeline.png")
    print(f"  - {location}_removal_summary.png")

