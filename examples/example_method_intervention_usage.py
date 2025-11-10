"""
Examples demonstrating how to build and apply interventions with
`fpsim.MethodIntervention`.

Each example highlights a different capability of the interface while using the
same core simulation parameters for comparison:

    1. Baseline (no intervention)
    2. Method mix and probability update (single method, RandomChoice module)
    3. Efficacy and continuation change (single method, RandomChoice module)
    4. Switching matrix scaling (single method, SimpleChoice module)

Method names provided to the interface must match the canonical short names used
inside fpsim:

    ['none', 'pill', 'iud', 'inj', 'cond', 'btl', 'wdraw', 'impl', 'othtrad', 'othmod']

The interface converts these names to the labels expected by the core
`update_methods` intervention when `build()` is called.
"""

from __future__ import annotations

import numpy as np
import sciris as sc

import fpsim as fp
from plots import plot_comparison_full

usecases = [1, 2, 3]
plot_compare = [[1,2], [1,3], [2,3]]

# region: Helper functions
def _default_pars(n_agents=5_000, location='kenya', start_year=2000, end_year=2015):
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

# region: Baseline simulation
def run_baseline(
    label='Baseline (no intervention)',
):
    """Run a baseline simulation without interventions for comparison."""
    pars = _default_pars()
    sim = fp.Sim(pars=pars, label=label)
    sim.run()
    _print_key_results(sim, label)
    return sim

# endregion: Baseline simulation

# region: Method mix and probability update
def run_with_method_mix_adjustment(
    year_apply=2007.0,
    label='run_with_method_mix_adjustment',
):
    """
    Showcase `set_method_mix` together with `set_probability_of_use`, adjusting a
    single method (`pill`) while preserving the baseline mix for other methods.

    RandomChoice does not use a switching matrix, so we focus on method mix and
    global p_use adjustments.
    """
    pars = _default_pars()

    # Capture the existing method mix so only the provided values change
    temp_sim = fp.Sim(pars=pars, label='Tmp method mix baseline (mix case)')
    temp_sim.init()

    mod = fp.MethodIntervention(year=year_apply, label=label)
    mod.capture_method_mix_from_sim(temp_sim)
    del temp_sim

    mod.set_probability_of_use(0.58)
    mod.set_method_mix('pill', 0.25)  # Target pill share (others rescaled automatically)

    print(f'\nPreview of configured payload ({label}):')
    sc.pp(mod.preview())
    
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv, label=label)
    sim.run()
    _print_key_results(sim, label)
    return sim

# endregion: Method mix and probability update

# region: Efficacy and duration changes
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

# endregion: Efficacy and duration changes

# region: Switching matrix scaling
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

# endregion: Switching matrix scaling

# region: Main function
if __name__ == '__main__':

    # Map case numbers to functions
    cases = {
        1: ('Baseline', run_baseline),
        2: ('Method Mix and Probability Update', run_with_method_mix_adjustment),
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
        title = f"{name1} .vs. {name2}"
        plot_comparison_full(sims[plot_pair[0]], sims[plot_pair[1]], title=title, filename=f"plot_comparison_{plot_pair[0]}_{plot_pair[1]}.png")
