"""
Examples demonstrating how to build and apply interventions with
`fpsim.MethodIntervention`.

Each example highlights a different capability of the interface while using the
same core simulation parameters for comparison:

    0. Baseline (no intervention)
    1. A very simple use case - improve method efficacy
    2. Method mix and duration update - LARC promotion program
    3. Efficacy and continuation changes - comprehensive injectable program
    4. Switching matrix scaling - improving method switching patterns

All examples use Kenya location with StandardChoice module (the default). 
They demonstrate interventions that work with all contraception modules.

Method names provided to the interface must match the canonical short names used
inside fpsim:

    ['none', 'pill', 'iud', 'inj', 'cond', 'btl', 'wdraw', 'impl', 'othtrad', 'othmod']

The interface converts these names to the labels expected by the core
`update_methods` intervention when `build()` is called.

IMPORTANT NOTE about `set_probability_of_use()`:
- This method only works with RandomChoice contraception module
- StandardChoice/SimpleChoice calculate probability from individual attributes
  (age, education, wealth, parity) and cannot be overridden with a simple p_use value
- For those modules, use efficacy, duration, or method mix interventions instead
"""

from __future__ import annotations

import numpy as np
import sciris as sc

import fpsim as fp
from plots import plot_comparison_full, plot_method_bincount, plot_method_mix_simple, plot_pregnancies_per_year

location = 'senegal'
usecases = [0, 1]
plot_compare = [[0,1]]
print('*'*40)
baseline_sim = None

# region: Utils & Default simulation parameters
def _default_pars(n_agents=5_000, location=location, start_year=2000, end_year=2015):
    """Baseline parameters shared by all example simulations."""
    return dict(
        n_agents=n_agents,
        location=location,
        start_year=start_year,
        end_year=end_year,
        exposure_factor=1.0,
    )

def _print_key_results(sim: fp.Sim, label: str):
    """Print a compact summary of key FP indicators."""
    res_fp = sim.results.fp
    cpr_series = res_fp.get('cpr')
    mcpr_series = res_fp.get('mcpr')
    births_series = res_fp.get('total_births', res_fp.get('births'))

    cpr_val = float(cpr_series[-1]) if cpr_series is not None else float('nan')
    mcpr_val = float(mcpr_series[-1]) if mcpr_series is not None else float('nan')
    births_val = float(np.sum(births_series)) if births_series is not None else float('nan')
    print(f'[{label}] CPR={cpr_val:0.3f}, mCPR={mcpr_val:0.3f}, total births={births_val:0.0f}')

# endregion: Helper functions

# region: 0. Baseline simulation
def run_baseline(
    label='Baseline (no intervention)',
):
    """Run a baseline simulation without interventions for comparison."""
    pars = _default_pars()
    sim = fp.Sim(pars=pars, label=label)
    sim.run()
    _print_key_results(sim, label)
    global baseline_sim
    baseline_sim = sim
    return sim

# endregion: Baseline simulation

# region: 1. Simple use case: improve method efficacy
def run_simple_usecase(
    year_apply=2007.0,
    label='A very simple use case',
):
    """
    Minimal example: improve contraceptive efficacy for injectables.
    
    This demonstrates a simple, realistic intervention: improving injectable
    contraceptive effectiveness from baseline ~97% to 99% through better quality
    products and user education.
    
    This works with all contraception modules (StandardChoice, SimpleChoice, RandomChoice).
    """
    pars = _default_pars()
    
    mod = fp.MethodIntervention(year=year_apply, label=label)
    mod.set_efficacy('inj', 0.99)  # Improve injectable efficacy to 99%
    
    print(f'\nPreview of configured payload ({label}):')
    sc.pp(mod.preview())
    
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv, label=label)
    sim.run()
    _print_key_results(sim, label)
    # if baseline_sim is provided, plot the method mix
    if baseline_sim is not None:
        plot_method_bincount(baseline_sim, sim, title=label, filename=f"{location}_plot_method_bincount_{label}.png", save_figure=True)
    return sim

# endregion: Very simple use case

# region: 2. Method mix and duration update  
def run_with_method_mix_adjustment(
    year_apply=2007.0,
    label='run_with_method_mix_adjustment',
):
    """
    Showcase `set_method_mix` together with `set_duration_months`, adjusting a
    single method (`impl`) while preserving the baseline mix for other methods.
    
    This demonstrates a comprehensive LARC promotion intervention: increasing
    both the share of women choosing implants AND improving continuation rates.
    
    Works with all contraception modules (StandardChoice, SimpleChoice, RandomChoice).
    
    The baseline method mix is captured automatically by passing `baseline_sim`
    to the first `set_method_mix` call.
    """
    pars = _default_pars()

    mod = fp.MethodIntervention(year=year_apply, label=label)
    
    # Capture the existing method mix using the convenience baseline_sim parameter
    baseline_sim = fp.Sim(pars=pars, label='Tmp method mix baseline (mix case)')
    baseline_sim.init()
    mod.set_method_mix('impl', 0.20, baseline_sim=baseline_sim, print_method_mix=True)  # Target 20% implant share
    del baseline_sim

    mod.set_duration_months('impl', 48)  # Improve continuation to 4 years

    print(f'\nPreview of configured payload ({label}):')
    sc.pp(mod.preview())
    
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv, label=label)
    sim.run()
    _print_key_results(sim, label)
    if baseline_sim is not None:
        plot_method_bincount(baseline_sim, sim, title=label, filename=f"{location}_run_with_method_mix_adjustment_{label}.png", save_figure=True)
    
    return sim

# region: 3. Efficacy and duration changes
def run_with_efficacy_and_duration_changes(
    year_apply=2007.0,
    label='run_with_efficacy_and_duration_changes',
):
    """
    Highlight `set_efficacy` and `set_duration_months` capabilities while
    modifying a single method (`inj`).
    """
    pars = _default_pars()
    mod = fp.MethodIntervention(year=year_apply, label=label)
    mod.set_efficacy('inj', 0.992)
    mod.set_duration_months('inj', 3)     # 3 years + 4 months
    
    print(f'\nPreview of configured payload ({label}):')
    sc.pp(mod.preview())
    
    intv = mod.build()
    sim = fp.Sim(pars=pars, 
                 interventions=intv, 
                 label=label)
    sim.run()
    _print_key_results(sim, label)
    return sim

# REGION: Efficacy and duration changes

# region: 4. Switching matrix scaling
def run_with_switching_matrix_scaling(
    year_apply=2007.0,
    label='run_with_switching_matrix_scaling',
):
    """
    Demonstrate `scale_switching_matrix` for modules that honor switching matrices.

    We initialize a temporary simulation to access the location-derived switching
    matrix, scale the probability of moving into injectables, and then run a fresh
    simulation with the modified matrix.
    """
    pars = _default_pars()
    temp_sim = fp.Sim(pars=pars, label='Tmp init for matrix (switch case)')
    temp_sim.init()

    mod = fp.MethodIntervention(year=year_apply, label=label)
    mod.scale_switching_matrix(temp_sim, target_method='inj', scale_factor=1.2)
    mod.set_efficacy('inj', 0.985)
    mod.set_duration_months('iud', 45)
    
    print(f'\nPreview of configured payload ({label}):')
    sc.pp(mod.preview())
    
    intv = mod.build()

    sim = fp.Sim(pars=pars, 
                 interventions=intv, 
                 label=label)
    sim.run()
    _print_key_results(sim, label)
    return sim

# REGION: Switching matrix scaling

# region: MAIN FUNCTION
if __name__ == '__main__':

    # Map case numbers to functions
    cases = {
        0: ('Baseline', run_baseline),
        1: ('Simple: Efficacy Improvement', run_simple_usecase),
        2: ('Method Mix and Duration Update', run_with_method_mix_adjustment),
        3: ('Efficacy and Duration Changes', run_with_efficacy_and_duration_changes),
        4: ('Switching Matrix Scaling', run_with_switching_matrix_scaling),
    }
    
    # Parse which cases to run from command line
    if len(usecases) > 0:
        to_run = usecases
    else:
        to_run = list(cases.keys())
    
    # Store simulation results
    sims = {}
    
    for case_num in to_run:
        if case_num not in cases:
            print(f"Warning: Case {case_num} not found, skipping")
            continue
        name, func = cases[case_num]
        print(f"\n"+"---"*10 + f" Running Case {case_num}: {name} " + "---"*10 + "\n") 
        sims[case_num] = func()
    
    for plot_pair in plot_compare:
        name1 = cases[plot_pair[0]][0]
        name2 = cases[plot_pair[1]][0]
        title = f"{location}: {name1} vs. {name2}"
        plot_comparison_full(sims[plot_pair[0]], sims[plot_pair[1]], title=title, filename=f"{location}_plot_comparison_{plot_pair[0]}_{plot_pair[1]}.png", save_figure=True)
        plot_method_bincount(sims[plot_pair[0]], sims[plot_pair[1]], title=title, filename=f"{location}_plot_method_bincount_{plot_pair[0]}_{plot_pair[1]}.png", save_figure=True)
        plot_pregnancies_per_year(sims[plot_pair[0]], sims[plot_pair[1]], filename=f"{location}_plot_pregnancies_per_year_{plot_pair[0]}_{plot_pair[1]}.png", save_figure=True)
        
#endregion: MAIN FUNCTION